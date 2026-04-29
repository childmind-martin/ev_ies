from __future__ import annotations

import importlib.util
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from test_TD3 import (
    ALERT_COLUMNS,
    ALERT_TOL,
    CONFIG_COLUMNS,
    DAILY_SUMMARY_COLUMNS,
    STEP_DETAIL_COLUMNS,
    build_alert_rows,
    build_step_row,
    configure_stdio,
    export_test_workbook,
    save_csv,
)


YEARLY_CSV_PATH = Path(r"D:\wangye\chengxu\3.30\yearly_data_sci.csv")
YEARLY_EV_PATH = Path(r"D:\wangye\chengxu\3.30\ev_data_s123_bilevel.npy")
MODEL_PATH = Path("./models/ppo_sb3_direct/best/best_model.zip")

RESULT_DIR = Path("./results/ppo_sb3_direct_test")
SUMMARY_CSV = RESULT_DIR / "daily_summary.csv"
TIMESERIES_CSV = RESULT_DIR / "timeseries_detail.csv"
CONFIG_CSV = RESULT_DIR / "test_config.csv"
TEST_EXPORT_XLSX = RESULT_DIR / "ppo_test_export.xlsx"
RUNTIME_SUMMARY_JSON = RESULT_DIR / "runtime_summary.json"

SEED = 42
VAL_DAYS_PER_MONTH = 2
REWARD_SCALE = 1e-5
REQUESTED_DEVICE = "auto"


def resolve_excel_engine() -> str:
    for module_name, engine_name in (("openpyxl", "openpyxl"), ("xlsxwriter", "xlsxwriter")):
        if importlib.util.find_spec(module_name) is not None:
            return engine_name
    raise ModuleNotFoundError("Missing Excel writer dependency. Please install openpyxl or xlsxwriter.")


def preflight_check() -> str:
    print(f"[startup] Python executable: {sys.executable}")
    print(f"[startup] Python version: {sys.version.split()[0]}")

    if sys.version_info[:2] >= (3, 14) or sys.version_info.releaselevel != "final":
        raise RuntimeError(
            "Current interpreter is Python 3.14+ or a non-final release. "
            "Run this project with the tf_env Python 3.10 environment."
        )

    required_paths = [YEARLY_CSV_PATH, YEARLY_EV_PATH, MODEL_PATH]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_paths)}")

    excel_engine = resolve_excel_engine()
    required_modules = ["numpy", "pandas", "gymnasium", "torch", "stable_baselines3"]
    missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing_modules:
        raise ModuleNotFoundError(
            "Missing required packages: "
            f"{', '.join(missing_modules)}. "
            "Install numpy pandas gymnasium torch stable-baselines3 openpyxl."
        )
    return excel_engine


def import_runtime_modules():
    print("[startup] Importing PPO test dependencies...")
    import numpy as np
    import pandas as pd
    import torch as th
    from stable_baselines3 import PPO

    from park_ies_env import IESConfig, ParkIESEnv
    from yearly_case_env import YearlyEVProvider
    from yearly_csv_loader import YearlyCSVDataLoader

    print("[startup] PPO test dependencies imported.")
    return {
        "np": np,
        "pd": pd,
        "th": th,
        "PPO": PPO,
        "IESConfig": IESConfig,
        "ParkIESEnv": ParkIESEnv,
        "YearlyCSVDataLoader": YearlyCSVDataLoader,
        "YearlyEVProvider": YearlyEVProvider,
    }


def set_random_seed(seed: int, np, th) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def resolve_sb3_device(th, requested_device: str) -> str:
    requested = str(requested_device).strip() or "auto"
    lowered = requested.lower()
    if lowered == "auto":
        return "cuda" if th.cuda.is_available() else "cpu"
    if lowered.startswith("cuda") and not th.cuda.is_available():
        raise RuntimeError(
            f"REQUESTED_DEVICE={requested_device!r}, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build or set REQUESTED_DEVICE='cpu'."
        )
    return requested


def print_torch_runtime_summary(th, *, requested_device: str, resolved_device: str) -> None:
    print("[startup] Torch runtime summary:")
    print(f"  torch_version = {th.__version__}")
    print(f"  requested_device = {requested_device}")
    print(f"  resolved_device = {resolved_device}")
    print(f"  cuda_available = {th.cuda.is_available()}")
    print(f"  cuda_version = {th.version.cuda}")
    if th.cuda.is_available():
        print(f"  cuda_device_count = {th.cuda.device_count()}")
        print(f"  cuda_device_name = {th.cuda.get_device_name(0)}")
    else:
        print("  note = CUDA unavailable, so PPO will run on CPU.")


def _space_shape(space: Any) -> tuple[int, ...] | None:
    shape = getattr(space, "shape", None)
    if shape is None:
        return None
    return tuple(int(item) for item in shape)


def validate_model_spaces(model: Any, env: Any) -> None:
    model_obs_shape = _space_shape(getattr(model, "observation_space", None))
    env_obs_shape = _space_shape(env.observation_space)
    model_action_shape = _space_shape(getattr(model, "action_space", None))
    env_action_shape = _space_shape(env.action_space)

    mismatches: list[str] = []
    if model_obs_shape != env_obs_shape:
        mismatches.append(f"observation shape: model expects {model_obs_shape}, current env provides {env_obs_shape}")
    if model_action_shape != env_action_shape:
        mismatches.append(f"action shape: model expects {model_action_shape}, current env provides {env_action_shape}")

    if mismatches:
        details = "\n  - ".join(mismatches)
        raise RuntimeError(
            "Loaded PPO checkpoint is incompatible with the current ParkIESEnv.\n"
            f"  - {details}\n"
            "Retrain PPO with the updated train_PPO_sb3_direct.py before running this test."
        )


def rollout_one_day(model: Any, env: Any) -> tuple[list[dict[str, Any]], float]:
    obs, _ = env.reset()
    done = False
    infos: list[dict[str, Any]] = []
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        infos.append(step_info)
        total_reward += float(reward)
        done = bool(terminated or truncated)

    return infos, total_reward


def build_config_row(
    cfg: Any,
    *,
    run_id: str,
    timestamp_start: str,
    n_train_cases: int,
    n_val_cases: int,
    n_test_cases: int,
    model: Any,
) -> dict[str, Any]:
    cfg_values = asdict(cfg)
    return {
        "run_id": run_id,
        "timestamp_start": timestamp_start,
        "yearly_csv_path": str(YEARLY_CSV_PATH),
        "yearly_ev_path": str(YEARLY_EV_PATH),
        "model_path": str(MODEL_PATH),
        "summary_csv_path": str(SUMMARY_CSV),
        "timeseries_csv_path": str(TIMESERIES_CSV),
        "test_export_xlsx_path": str(TEST_EXPORT_XLSX),
        "n_train_cases": int(n_train_cases),
        "n_val_cases": int(n_val_cases),
        "n_test_cases": int(n_test_cases),
        "SEED": SEED,
        "VAL_DAYS_PER_MONTH": VAL_DAYS_PER_MONTH,
        "REWARD_SCALE": REWARD_SCALE,
        "algorithm": type(model).__name__,
        "policy_class": type(model.policy).__name__,
        "device": str(model.device),
        "episode_length": cfg_values["episode_length"],
        "dt": cfg_values["dt"],
        "future_horizon": cfg_values["future_horizon"],
        "exogenous_future_horizon": cfg_values["exogenous_future_horizon"],
        "grid_import_max": cfg_values["grid_import_max"],
        "grid_export_max": cfg_values["grid_export_max"],
        "gt_p_max": cfg_values["gt_p_max"],
        "gt_eta_e": cfg_values["gt_eta_e"],
        "gt_ramp": cfg_values["gt_ramp"],
        "gb_h_max": cfg_values["gb_h_max"],
        "gb_ramp": cfg_values["gb_ramp"],
        "ac_c_max": cfg_values["ac_c_max"],
        "ec_c_max": cfg_values["ec_c_max"],
        "ees_p_max": cfg_values["ees_p_max"],
        "ees_e_cap": cfg_values["ees_e_cap"],
        "ees_soc_init": cfg_values["ees_soc_init"],
        "ees_soc_min": cfg_values["ees_soc_min"],
        "ees_soc_max": cfg_values["ees_soc_max"],
        "ees_terminal_soc_tolerance": cfg_values["ees_terminal_soc_tolerance"],
        "penalty_unserved_e": cfg_values["penalty_unserved_e"],
        "penalty_unserved_h": cfg_values["penalty_unserved_h"],
        "penalty_unserved_c": cfg_values["penalty_unserved_c"],
        "penalty_depart_soc": cfg_values["penalty_depart_soc"],
        "penalty_depart_energy_soft": cfg_values["penalty_depart_energy_soft"],
        "penalty_depart_energy_mid": cfg_values["penalty_depart_energy_mid"],
        "penalty_ev_depart_risk": cfg_values["penalty_ev_depart_risk"],
        "penalty_surplus_e": cfg_values["penalty_surplus_e"],
        "penalty_surplus_h": cfg_values["penalty_surplus_h"],
        "penalty_surplus_c": cfg_values["penalty_surplus_c"],
        "penalty_export_e": cfg_values["penalty_export_e"],
        "penalty_ev_export_guard": cfg_values["penalty_ev_export_guard"],
        "penalty_ees_terminal_soc": cfg_values["penalty_ees_terminal_soc"],
        "ev_discharge_price_threshold": cfg_values["ev_discharge_price_threshold"],
        "ev_charge_price_threshold": cfg_values["ev_charge_price_threshold"],
        "ev_peak_export_tolerance_kw": cfg_values["ev_peak_export_tolerance_kw"],
        "reward_storage_discharge_base": cfg_values["reward_storage_discharge_base"],
        "reward_storage_charge_base": cfg_values["reward_storage_charge_base"],
        "reward_ev_target_timing_base": cfg_values["reward_ev_target_timing_base"],
        "ees_reward_discharge_soc_floor": cfg_values["ees_reward_discharge_soc_floor"],
        "ees_reward_charge_soc_target": cfg_values["ees_reward_charge_soc_target"],
        "reward_scale": cfg_values["reward_scale"],
    }


def _sum(infos: list[dict[str, Any]], key: str) -> float:
    return float(sum(item.get(key, 0.0) for item in infos))


def _avg(infos: list[dict[str, Any]], key: str) -> float:
    return float(_sum(infos, key) / max(len(infos), 1))


def build_daily_summary(case_idx: int, case: Any, infos: list[dict[str, Any]], total_reward: float, cfg: Any) -> dict[str, Any]:
    total_gt_export_clip_steps = int(sum(1 for item in infos if float(item.get("p_gt_export_clip", 0.0)) > ALERT_TOL))
    total_gt_safe_infeasible_steps = int(sum(1 for item in infos if not bool(item.get("gt_safe_feasible", True))))
    final_info = infos[-1] if infos else {}
    missing_terminal_info = [
        key
        for key in (
            "final_ees_soc",
            "ees_soc_init",
            "terminal_ees_required_soc",
            "episode_terminal_ees_shortage_kwh",
            "episode_penalty_terminal_ees_soc",
            "ees_terminal_soc_feasible",
        )
        if key not in final_info
    ]
    if missing_terminal_info:
        print(
            "WARNING: EES terminal SOC info missing: "
            f"case_index={case_idx}, missing={missing_terminal_info}; fallback values used.",
            file=sys.stderr,
        )
    final_ees_soc = float(final_info.get("final_ees_soc", final_info.get("ees_soc", 0.0)))
    ees_soc_init = float(final_info.get("ees_soc_init", final_info.get("ees_soc_episode_init", 0.0)))
    terminal_ees_required_soc = float(final_info.get("terminal_ees_required_soc", 0.0))
    terminal_ees_shortage_kwh = float(final_info.get("episode_terminal_ees_shortage_kwh", 0.0))
    ees_terminal_soc_feasible = bool(final_info.get("ees_terminal_soc_feasible", True))
    row = {
        "case_index": int(case_idx),
        "month": case.month,
        "day_of_year": case.day_of_year,
        "season": case.season,
        "split": case.set_name,
        "total_reward": float(total_reward),
        "total_reward_raw": _sum(infos, "reward_raw"),
        "total_system_cost": _sum(infos, "system_cost"),
        "total_penalties": _sum(infos, "penalty_cost"),
        "total_guide_reward": _sum(infos, "guide_reward"),
        "total_cost_grid": _sum(infos, "cost_grid"),
        "total_cost_gas": _sum(infos, "cost_gas"),
        "total_cost_deg": _sum(infos, "cost_deg"),
        "total_cost_om": _sum(infos, "cost_om"),
        "total_unmet_e": _sum(infos, "unmet_e"),
        "total_unmet_h": _sum(infos, "unmet_h"),
        "total_unmet_c": _sum(infos, "unmet_c"),
        "total_surplus_e": _sum(infos, "surplus_e"),
        "total_surplus_h": _sum(infos, "surplus_h"),
        "total_surplus_c": _sum(infos, "surplus_c"),
        "total_grid_overflow": _sum(infos, "grid_overflow"),
        "total_penalty_unserved_e": _sum(infos, "penalty_unserved_e"),
        "total_penalty_unserved_h": _sum(infos, "penalty_unserved_h"),
        "total_penalty_unserved_c": _sum(infos, "penalty_unserved_c"),
        "total_penalty_depart_energy": _sum(infos, "penalty_depart_energy"),
        "total_penalty_depart_energy_soft": _sum(infos, "penalty_depart_energy_soft"),
        "total_penalty_depart_energy_mid": _sum(infos, "penalty_depart_energy_mid"),
        "total_penalty_depart_energy_hard": _sum(infos, "penalty_depart_energy_hard"),
        "total_penalty_depart_risk": _sum(infos, "penalty_depart_risk"),
        "total_penalty_surplus_e": _sum(infos, "penalty_surplus_e"),
        "total_penalty_surplus_h": _sum(infos, "penalty_surplus_h"),
        "total_penalty_surplus_c": _sum(infos, "penalty_surplus_c"),
        "total_penalty_export_e": _sum(infos, "penalty_export_e"),
        "total_penalty_ev_export_guard": _sum(infos, "penalty_ev_export_guard"),
        "total_penalty_terminal_ees_soc": float(final_info.get("episode_penalty_terminal_ees_soc", 0.0)),
        "total_depart_energy_shortage_kwh": _sum(infos, "depart_energy_shortage_kwh"),
        "total_depart_shortage_soft_kwh": _sum(infos, "depart_shortage_soft_kwh"),
        "total_depart_shortage_mid_kwh": _sum(infos, "depart_shortage_mid_kwh"),
        "total_depart_shortage_hard_kwh": _sum(infos, "depart_shortage_hard_kwh"),
        "total_depart_risk_energy_kwh": _sum(infos, "depart_risk_energy_kwh"),
        "total_reward_storage_discharge_bonus": _sum(infos, "reward_storage_discharge_bonus"),
        "total_reward_storage_charge_bonus": _sum(infos, "reward_storage_charge_bonus"),
        "total_reward_ev_target_timing_bonus": _sum(infos, "reward_ev_target_timing_bonus"),
        "total_storage_peak_shaved_kwh": _sum(infos, "storage_peak_shaved_kwh"),
        "total_storage_charge_rewarded_kwh": _sum(infos, "storage_charge_rewarded_kwh"),
        "total_ees_charge_rewarded_kwh": _sum(infos, "ees_charge_rewarded_kwh"),
        "total_ev_flex_target_charge_kwh": _sum(infos, "ev_flex_target_charge_kwh"),
        "total_ev_buffer_charge_kwh": _sum(infos, "ev_buffer_charge_kwh"),
        "total_low_value_charge_kwh": _sum(infos, "low_value_charge_kwh"),
        "total_gt_export_clip": _sum(infos, "p_gt_export_clip"),
        "total_gt_export_clip_steps": total_gt_export_clip_steps,
        "total_gt_safe_infeasible_steps": total_gt_safe_infeasible_steps,
        "total_grid_buy_kwh": _sum(infos, "p_grid_buy") * float(cfg.dt),
        "total_grid_sell_kwh": _sum(infos, "p_grid_sell") * float(cfg.dt),
        "avg_p_gt_kw": _avg(infos, "p_gt"),
        "avg_p_ev_ch_kw": _avg(infos, "p_ev_ch"),
        "avg_p_ev_dis_kw": _avg(infos, "p_ev_dis"),
        "avg_p_ees_ch_kw": _avg(infos, "p_ees_ch"),
        "avg_p_ees_dis_kw": _avg(infos, "p_ees_dis"),
        "ees_soc_episode_init": float(infos[0].get("ees_soc_episode_init", 0.0)) if infos else 0.0,
        "final_ees_soc": final_ees_soc,
        "ees_soc_init": ees_soc_init,
        "terminal_ees_required_soc": terminal_ees_required_soc,
        "terminal_ees_shortage_kwh": terminal_ees_shortage_kwh,
        "ees_terminal_soc_feasible": ees_terminal_soc_feasible,
        "final_gt_power": float(infos[-1].get("p_gt", 0.0)) if infos else 0.0,
    }
    return {column: row.get(column) for column in DAILY_SUMMARY_COLUMNS}


def print_grand_summary(summary_rows: list[dict[str, Any]]) -> None:
    if not summary_rows:
        print("No PPO test results were generated.")
        return

    def total(key: str) -> float:
        return float(sum(item.get(key, 0.0) for item in summary_rows))

    print("\n========== PPO Test Set Summary ==========")
    print(f"Test days: {len(summary_rows)}")
    print(f"Total system cost: {total('total_system_cost'):.4f}")
    print(f"Total penalties: {total('total_penalties'):.4f}")
    print(f"Total cost + penalties: {total('total_system_cost') + total('total_penalties'):.4f}")
    print(f"Total unmet electricity: {total('total_unmet_e'):.4f}")
    print(f"Total unmet heat: {total('total_unmet_h'):.4f}")
    print(f"Total unmet cooling: {total('total_unmet_c'):.4f}")
    print(f"Total EV departure shortage (kWh): {total('total_depart_energy_shortage_kwh'):.4f}")
    print(f"Penalty EES terminal SOC: {total('total_penalty_terminal_ees_soc'):.4f}")
    print(f"EES terminal shortage (kWh): {total('terminal_ees_shortage_kwh'):.4f}")
    print(f"Total grid buy (kWh): {total('total_grid_buy_kwh'):.4f}")
    print(f"Total grid sell (kWh): {total('total_grid_sell_kwh'):.4f}")
    print(f"Storage peak shaved energy (kWh): {total('total_storage_peak_shaved_kwh'):.4f}")
    print(f"GT safety infeasible steps: {int(total('total_gt_safe_infeasible_steps'))}")
    print(f"Total reward: {total('total_reward'):.4f}")


def write_runtime_summary(
    path: Path,
    *,
    method: str,
    n_days: int,
    n_steps: int,
    test_duration_seconds: float,
) -> dict[str, Any]:
    time_per_day = test_duration_seconds / n_days if n_days > 0 else 0.0
    time_per_step = test_duration_seconds / n_steps if n_steps > 0 else 0.0
    runtime_summary = {
        "method": method,
        "n_days": int(n_days),
        "n_steps": int(n_steps),
        "test_duration_seconds": round(float(test_duration_seconds), 6),
        "time_per_day_seconds": round(float(time_per_day), 6),
        "time_per_step_seconds": round(float(time_per_step), 6),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(runtime_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return runtime_summary


def main() -> None:
    configure_stdio()
    try:
        excel_engine = preflight_check()
        modules = import_runtime_modules()
    except Exception as exc:
        print(f"[startup] PPO test startup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    np = modules["np"]
    pd = modules["pd"]
    th = modules["th"]
    PPO = modules["PPO"]
    IESConfig = modules["IESConfig"]
    ParkIESEnv = modules["ParkIESEnv"]
    YearlyCSVDataLoader = modules["YearlyCSVDataLoader"]
    YearlyEVProvider = modules["YearlyEVProvider"]

    set_random_seed(SEED, np, th)
    resolved_device = resolve_sb3_device(th, REQUESTED_DEVICE)
    print_torch_runtime_summary(th, requested_device=REQUESTED_DEVICE, resolved_device=resolved_device)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("[startup] Loading yearly case split for PPO testing...")
    loader = YearlyCSVDataLoader(YEARLY_CSV_PATH, val_days_per_month=VAL_DAYS_PER_MONTH)
    train_cases, val_cases, test_cases = loader.load()
    loader.summary()
    print(f"[startup] Test case count: {len(test_cases)}")
    if not test_cases:
        raise RuntimeError("Test set is empty. Please check the split configuration.")

    print("[startup] Loading yearly EV data...")
    ev_provider = YearlyEVProvider(YEARLY_EV_PATH)

    print(f"[startup] Loading PPO model from: {MODEL_PATH}")
    model = PPO.load(str(MODEL_PATH), device=resolved_device)

    cfg = IESConfig(episode_length=24, dt=1.0, reward_scale=REWARD_SCALE)
    probe_case = test_cases[0]
    probe_ev_data = ev_provider(probe_case)
    probe_env = ParkIESEnv(cfg=cfg, ts_data=probe_case.ts_data, ev_data=probe_ev_data)
    try:
        validate_model_spaces(model, probe_env)
    except RuntimeError as exc:
        print(f"[startup] {exc}", file=sys.stderr)
        probe_env.close()
        raise SystemExit(1) from exc

    print("[startup] PPO test environment probe:")
    print(f"  reward_scale = {cfg.reward_scale}")
    print(f"  obs_dim = {probe_env.obs_dim}")
    print(f"  action_dim = {probe_env.action_space.shape[0]}")
    print(f"  future_horizon = {cfg.future_horizon}")
    print(f"  exogenous_future_horizon = {cfg.exogenous_future_horizon}")
    print(f"  device = {model.device}")
    probe_env.close()

    run_started_at = datetime.now().astimezone()
    run_id = run_started_at.strftime("%Y%m%d_%H%M%S")
    timestamp_start = run_started_at.isoformat(timespec="seconds")

    summary_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    test_started_perf = perf_counter()
    for case_idx, case in enumerate(test_cases):
        ev_data = ev_provider(case)
        env = ParkIESEnv(cfg=cfg, ts_data=case.ts_data, ev_data=ev_data)
        infos, total_reward = rollout_one_day(model, env)

        summary_row = build_daily_summary(case_idx, case, infos, total_reward, cfg)
        summary_rows.append(summary_row)
        if not bool(summary_row.get("ees_terminal_soc_feasible", True)):
            print(
                "WARNING: EES terminal SOC violation: "
                f"case_index={case_idx}, final_ees_soc={summary_row['final_ees_soc']:.6f}, "
                f"required={summary_row['terminal_ees_required_soc']:.6f}, "
                f"shortage_kwh={summary_row['terminal_ees_shortage_kwh']:.6f}",
                file=sys.stderr,
            )
        for step_info in infos:
            step_rows.append(build_step_row(case_idx, case, step_info))
        env.close()

        print(
            f"[{case_idx + 1:02d}/{len(test_cases)}] "
            f"Month={case.month}, DayOfYear={case.day_of_year}, "
            f"total_system_cost={summary_row['total_system_cost']:.4f}, "
            f"total_penalties={summary_row['total_penalties']:.4f}, "
            f"depart_shortage={summary_row['total_depart_energy_shortage_kwh']:.4f}, "
            f"total_reward={summary_row['total_reward']:.4f}"
        )

    alert_rows = build_alert_rows(step_rows)
    test_duration_seconds = max(perf_counter() - test_started_perf, 0.0)
    save_csv(summary_rows, SUMMARY_CSV, DAILY_SUMMARY_COLUMNS)
    save_csv(step_rows, TIMESERIES_CSV, STEP_DETAIL_COLUMNS)
    runtime_summary = write_runtime_summary(
        RUNTIME_SUMMARY_JSON,
        method="PPO",
        n_days=len(summary_rows),
        n_steps=len(step_rows),
        test_duration_seconds=test_duration_seconds,
    )

    config_row = build_config_row(
        cfg,
        run_id=run_id,
        timestamp_start=timestamp_start,
        n_train_cases=len(train_cases),
        n_val_cases=len(val_cases),
        n_test_cases=len(test_cases),
        model=model,
    )
    save_csv([config_row], CONFIG_CSV, CONFIG_COLUMNS)
    export_test_workbook(
        pd,
        output_path=TEST_EXPORT_XLSX,
        excel_engine=excel_engine,
        step_rows=step_rows,
        summary_rows=summary_rows,
        config_row=config_row,
        alert_rows=alert_rows,
    )

    print_grand_summary(summary_rows)
    print(f"PPO daily summary CSV saved to: {SUMMARY_CSV}")
    print(f"PPO timeseries detail CSV saved to: {TIMESERIES_CSV}")
    print(f"PPO runtime summary JSON saved to: {RUNTIME_SUMMARY_JSON}")
    print(
        "PPO test runtime: "
        f"{runtime_summary['test_duration_seconds']:.6f} s, "
        f"{runtime_summary['time_per_day_seconds']:.6f} s/day, "
        f"{runtime_summary['time_per_step_seconds']:.6f} s/step"
    )
    print(f"PPO test config CSV saved to: {CONFIG_CSV}")
    print(f"PPO test export Excel saved to: {TEST_EXPORT_XLSX}")
    print(f"Alert rows exported: {len(alert_rows)}")


if __name__ == "__main__":
    main()
