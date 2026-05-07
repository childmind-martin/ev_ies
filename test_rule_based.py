from __future__ import annotations

import argparse
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
    ALERT_TOL,
    CONFIG_COLUMNS,
    DAILY_SUMMARY_COLUMNS,
    REWARD_SCALE,
    STEP_DETAIL_COLUMNS,
    VAL_DAYS_PER_MONTH,
    YEARLY_CSV_PATH as DEFAULT_YEARLY_CSV_PATH,
    YEARLY_EV_PATH as DEFAULT_YEARLY_EV_PATH,
    build_alert_rows,
    build_step_row,
    configure_stdio,
    export_test_workbook,
    resolve_excel_engine,
    save_csv,
)


DEFAULT_OUTPUT_DIR = Path("./results/rule_based_v2g_test")
DEFAULT_TD3_SUMMARY_CSV = Path("./results/td3_yearly_test/daily_summary.csv")
SUMMARY_FILENAME = "daily_summary.csv"
TIMESERIES_FILENAME = "timeseries_detail.csv"
RUNTIME_SUMMARY_FILENAME = "runtime_summary.json"
DIAGNOSTICS_FILENAME = "rule_based_diagnostics.md"
EXCEL_FILENAME = "rule_based_test_export.xlsx"
COMPARISON_FILENAME = "rule_based_vs_td3_summary.csv"
DIAGNOSTIC_TOL = 1e-2
COMPARISON_HEADING_SUFFIX = "\u521d\u6b65\u5bf9\u6bd4"
np = None


COMPARISON_COLUMNS = (
    "method",
    "n_days",
    "mean_total_system_cost",
    "mean_total_penalties",
    "mean_total_cost_plus_penalty",
    "mean_total_grid_buy_kwh",
    "mean_total_grid_sell_kwh",
    "mean_total_depart_energy_shortage_kwh",
    "mean_total_unmet_e",
    "mean_total_unmet_h",
    "mean_total_unmet_c",
    "mean_total_storage_peak_shaved_kwh",
    "mean_total_ev_flex_target_charge_kwh",
    "mean_total_ev_buffer_charge_kwh",
    "mean_final_ees_soc",
    "terminal_ees_feasible_days",
    "terminal_ees_feasible_ratio",
    "min_final_ees_soc",
    "sum_terminal_ees_shortage_kwh",
    "sum_penalty_terminal_ees_soc",
    "total_depart_energy_shortage_kwh",
    "total_unmet_e",
    "total_unmet_h",
    "total_unmet_c",
    "total_grid_sell_kwh",
    "total_penalty_ev_export_guard",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a TOU + SOC rule-based baseline on the same test set as TD3."
    )
    parser.add_argument("--yearly-csv", type=Path, default=DEFAULT_YEARLY_CSV_PATH)
    parser.add_argument("--yearly-ev", type=Path, default=DEFAULT_YEARLY_EV_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ev-mode", choices=("v2g", "charge_only"), default="v2g")
    return parser


def method_label(ev_mode: str) -> str:
    if ev_mode == "v2g":
        return "Rule-based-V2G"
    if ev_mode == "charge_only":
        return "Rule-based-ChargeOnly"
    raise ValueError("ev_mode must be one of: v2g, charge_only")


def comparison_heading(ev_mode: str) -> str:
    return f"## {method_label(ev_mode)} vs TD3 {COMPARISON_HEADING_SUFFIX}"


def preflight_check(yearly_csv: Path, yearly_ev: Path) -> None:
    if sys.version_info[:2] >= (3, 14):
        raise RuntimeError(
            "Current interpreter is Python 3.14+. This project should be run with "
            "a Python 3.9-3.11 environment that has numpy and pandas installed. "
            "Run the same command with that environment, for example: "
            "<python-3.9-to-3.11> test_rule_based.py --ev-mode v2g."
        )

    required_modules = ["numpy", "pandas"]
    missing_modules = [
        name for name in required_modules if importlib.util.find_spec(name) is None
    ]
    if missing_modules:
        raise ModuleNotFoundError(
            "Missing required packages: "
            f"{', '.join(missing_modules)}. "
            "Please install the dependencies before running the rule-based test."
        )

    if importlib.util.find_spec("gymnasium") is None:
        print(
            "[startup] gymnasium is not installed; using a minimal local compatibility "
            "shim for Env and spaces.Box."
        )

    missing_paths = [str(path) for path in (yearly_csv, yearly_ev) if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Required input data not found: "
            f"{', '.join(missing_paths)}. "
            "Use --yearly-csv and --yearly-ev to specify the data paths."
        )


def ensure_gymnasium_compat() -> None:
    if importlib.util.find_spec("gymnasium") is not None:
        return

    import types

    class Env:
        metadata = {}

        def reset(self, *, seed=None, options=None):
            self.np_random = np.random.default_rng(seed)
            return None

        def close(self):
            return None

    class Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.dtype = np.dtype(dtype)
            if shape is None:
                low_arr = np.asarray(low, dtype=self.dtype)
                high_arr = np.asarray(high, dtype=self.dtype)
            else:
                low_arr = (
                    np.full(shape, low, dtype=self.dtype)
                    if np.isscalar(low)
                    else np.asarray(low, dtype=self.dtype)
                )
                high_arr = (
                    np.full(shape, high, dtype=self.dtype)
                    if np.isscalar(high)
                    else np.asarray(high, dtype=self.dtype)
                )
                low_arr = np.broadcast_to(low_arr, shape).astype(self.dtype, copy=True)
                high_arr = np.broadcast_to(high_arr, shape).astype(self.dtype, copy=True)
            self.low = low_arr
            self.high = high_arr
            self.shape = self.low.shape

        def __repr__(self) -> str:
            return f"Box({self.low}, {self.high}, {self.shape}, {self.dtype})"

    gymnasium_module = types.ModuleType("gymnasium")
    spaces_module = types.ModuleType("gymnasium.spaces")
    spaces_module.Box = Box
    gymnasium_module.Env = Env
    gymnasium_module.spaces = spaces_module
    sys.modules["gymnasium"] = gymnasium_module
    sys.modules["gymnasium.spaces"] = spaces_module


def import_runtime_modules():
    global np
    print("[startup] Importing rule-based test dependencies...")
    import numpy as numpy_module
    import pandas as pd

    np = numpy_module
    ensure_gymnasium_compat()

    from park_ies_env import IESConfig, ParkIESEnv
    from yearly_case_env import YearlyEVProvider
    from yearly_csv_loader import YearlyCSVDataLoader

    print("[startup] Rule-based test dependencies imported.")
    return {
        "pd": pd,
        "IESConfig": IESConfig,
        "ParkIESEnv": ParkIESEnv,
        "YearlyCSVDataLoader": YearlyCSVDataLoader,
        "YearlyEVProvider": YearlyEVProvider,
    }


def resolve_optional_excel_engine() -> str | None:
    try:
        return resolve_excel_engine()
    except ModuleNotFoundError as exc:
        print(f"[export] Excel export skipped: {exc}")
        return None


def rule_based_action(inner_env, ev_mode: str = "v2g") -> np.ndarray:
    mode = ev_mode.strip().lower()
    if mode not in {"v2g", "charge_only"}:
        raise ValueError("ev_mode must be one of: v2g, charge_only")

    cfg = inner_env.cfg
    t = int(inner_env.t)
    T = int(inner_env.T)
    t_idx = min(max(t, 0), T - 1)

    try:
        exogenous = inner_env._get_exogenous(t_idx)
    except Exception:
        exogenous = {
            "elec_load": 0.0,
            "heat_load": 0.0,
            "pv": 0.0,
            "wt": 0.0,
            "grid_buy_price": float(
                np.asarray(cfg.grid_buy_price_24, dtype=np.float32)[t_idx % 24]
            ),
        }

    price = float(exogenous["grid_buy_price"])
    elec_load = float(exogenous.get("elec_load", 0.0))
    heat_load = float(exogenous.get("heat_load", 0.0))
    pv = float(exogenous.get("pv", 0.0))
    wt = float(exogenous.get("wt", 0.0))
    net_electric_pressure = elec_load - pv - wt
    ees_soc = float(inner_env.ees_soc)

    required_soc = max(
        float(cfg.ees_soc_min),
        float(inner_env.ees_soc_episode_init) - float(cfg.ees_terminal_soc_tolerance),
    )
    remaining_after_this = max(0, T - t - 1)
    max_recoverable_soc_after = (
        remaining_after_this
        * float(cfg.ees_p_max)
        * float(cfg.ees_eta_ch)
        * float(cfg.dt)
        / max(float(cfg.ees_e_cap), 1e-6)
    )
    dynamic_min_soc = max(
        float(cfg.ees_soc_min),
        required_soc - max_recoverable_soc_after,
    )

    try:
        ev_now = inner_env._compute_ev_boundaries(t_idx, inner_env.ev_soc)
    except Exception:
        ev_now = {}

    ev_departure_risk = False
    target_charge_boundary_kw = float(ev_now.get("p_flex_target_ch", 0.0) or 0.0)
    discharge_boundary_kw = float(ev_now.get("p_flex_dis", 0.0) or 0.0)
    rigid_charge_boundary_kw = float(ev_now.get("p_rigid", 0.0) or 0.0)
    max_ev_charge_kw = 1.0

    try:
        ev_data = inner_env.ev
        ev_soc = np.asarray(inner_env.ev_soc, dtype=np.float32)
        arr = np.asarray(ev_data["arr_step"], dtype=np.int32)
        dep = np.asarray(ev_data["dep_step"], dtype=np.int32)
        target_soc = np.asarray(ev_data["target_soc"], dtype=np.float32)
        cap_kwh = np.asarray(ev_data["cap_kwh"], dtype=np.float32)
        p_ch_max = np.asarray(ev_data["p_ch_max_kw"], dtype=np.float32)
        max_ev_charge_kw = max(float(np.sum(p_ch_max, dtype=np.float64)), 1.0)
        active = (arr <= t_idx) & (t_idx < dep)
        target_gap_kwh = np.maximum(target_soc - ev_soc, 0.0) * cap_kwh
        time_to_depart_steps = dep - t_idx
        soon_departure_risk = (
            active
            & (target_gap_kwh > 1e-6)
            & (time_to_depart_steps <= 2)
        )
        laxity = np.asarray(
            ev_now.get("laxity_target", np.full(ev_soc.shape, np.inf)),
            dtype=np.float32,
        )
        low_laxity_risk = (
            active
            & (target_gap_kwh > 1e-6)
            & (laxity <= 2.0 * float(cfg.dt) + 1e-6)
        )
        ev_departure_risk = bool(
            rigid_charge_boundary_kw > 1e-6
            or np.any(soon_departure_risk)
            or np.any(low_laxity_risk)
        )
    except Exception:
        ev_departure_risk = rigid_charge_boundary_kw > 1e-6

    a_ees = 0.0
    a_gt = 0.0
    a_ev_ch = 0.0
    a_ev_dis = 0.0

    terminal_recovery_active = bool(
        t >= T - 4 or ees_soc < dynamic_min_soc + 0.02
    )
    if terminal_recovery_active:
        recovery_target_soc = min(float(cfg.ees_soc_max), required_soc + 1e-4)
        shortage_kwh = max(0.0, recovery_target_soc - ees_soc) * float(cfg.ees_e_cap)
        if shortage_kwh > 1e-9:
            remaining_charge_steps = max(1, T - t)
            needed_charge_kw = shortage_kwh / (
                max(float(cfg.ees_eta_ch) * float(cfg.dt) * remaining_charge_steps, 1e-6)
            )
            a_ees = min(1.0, needed_charge_kw / max(float(cfg.ees_p_max), 1e-6))
    elif (
        price >= float(cfg.ev_discharge_price_threshold)
        or net_electric_pressure > float(cfg.gt_low_price_pressure_threshold_kw)
    ):
        allowed_discharge_kw = max(
            0.0,
            (ees_soc - dynamic_min_soc)
            * float(cfg.ees_e_cap)
            * float(cfg.ees_eta_dis)
            / max(float(cfg.dt), 1e-6),
        )
        if allowed_discharge_kw > 1e-6:
            a_ees = -min(1.0, allowed_discharge_kw / max(float(cfg.ees_p_max), 1e-6))
    elif (
        price <= float(cfg.ev_charge_price_threshold)
        and ees_soc < float(cfg.ees_reward_charge_soc_target)
    ):
        a_ees = 0.0 if ees_soc >= float(cfg.ees_soc_max) - 1e-6 else 1.0

    if ev_departure_risk:
        a_ev_ch = 1.0
        a_ev_dis = 0.0
    elif price <= float(cfg.ev_charge_price_threshold):
        a_ev_ch = 1.0
        a_ev_dis = 0.0
    elif target_charge_boundary_kw > max(100.0, 0.02 * max_ev_charge_kw):
        a_ev_ch = 0.5
        a_ev_dis = 0.0

    if (
        mode == "v2g"
        and not ev_departure_risk
        and price >= float(cfg.ev_discharge_price_threshold)
        and net_electric_pressure > float(cfg.ev_discharge_pressure_threshold_kw)
        and discharge_boundary_kw > 1e-6
    ):
        a_ev_dis = 1.0
        a_ev_ch = 0.0

    if mode == "charge_only":
        a_ev_dis = 0.0

    if (
        price >= float(cfg.ev_discharge_price_threshold)
        and net_electric_pressure > float(cfg.gt_low_price_pressure_threshold_kw)
    ):
        a_gt = 1.0
    elif (
        price <= float(cfg.ev_charge_price_threshold)
        and net_electric_pressure < float(cfg.gt_low_price_pressure_threshold_kw)
    ):
        a_gt = -1.0
    elif heat_load > 0.75 * float(cfg.heat_load_max):
        a_gt = 0.5

    action = np.array([a_ees, a_gt, a_ev_ch, a_ev_dis], dtype=np.float32)
    action = np.clip(action, inner_env.action_space.low, inner_env.action_space.high)
    return action.astype(np.float32)


def rollout_rule_based_one_day(env, ev_mode: str = "v2g"):
    obs, info = env.reset()
    done = False
    infos = []
    total_reward = 0.0

    while not done:
        inner_env = (
            env.inner_env
            if hasattr(env, "inner_env") and env.inner_env is not None
            else env
        )
        action = rule_based_action(inner_env, ev_mode=ev_mode)
        obs, reward, terminated, truncated, step_info = env.step(action)
        infos.append(step_info)
        total_reward += float(reward)
        done = bool(terminated or truncated)

    return infos, total_reward


def sum_info(infos: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(item.get(key, 0.0) or 0.0) for item in infos))


def avg_info(infos: list[dict[str, Any]], key: str) -> float:
    return sum_info(infos, key) / max(len(infos), 1)


def count_info(infos: list[dict[str, Any]], key: str, threshold: float = ALERT_TOL) -> int:
    return int(sum(1 for item in infos if float(item.get(key, 0.0) or 0.0) > threshold))


def build_daily_summary_row(
    case_idx: int,
    case: Any,
    infos: list[dict[str, Any]],
    total_reward: float,
    dt: float,
    cfg: Any,
) -> dict[str, Any]:
    final_info = infos[-1] if infos else {}
    if "final_ees_soc" not in final_info:
        print(
            "WARNING: final_ees_soc missing in terminal info: "
            f"case_index={case_idx}; falling back to terminal ees_soc.",
            file=sys.stderr,
        )
    if "ees_soc_init" not in final_info and "ees_soc_episode_init" not in final_info:
        print(
            "WARNING: EES initial SOC info missing in terminal info: "
            f"case_index={case_idx}; falling back to cfg.ees_soc_init.",
            file=sys.stderr,
        )

    final_ees_soc = float(final_info.get("final_ees_soc", final_info.get("ees_soc", 0.0)))
    ees_soc_init = float(
        final_info.get(
            "ees_soc_init",
            final_info.get("ees_soc_episode_init", float(cfg.ees_soc_init)),
        )
    )
    terminal_ees_required_soc = max(
        float(cfg.ees_soc_min),
        ees_soc_init - float(cfg.ees_terminal_soc_tolerance),
    )
    terminal_ees_shortage_kwh = (
        max(0.0, terminal_ees_required_soc - final_ees_soc) * float(cfg.ees_e_cap)
    )
    total_penalty_terminal_ees_soc = (
        terminal_ees_shortage_kwh * float(cfg.penalty_ees_terminal_soc)
    )
    ees_terminal_soc_feasible = terminal_ees_shortage_kwh <= 1e-6

    reported_terminal_shortage = final_info.get("episode_terminal_ees_shortage_kwh")
    if reported_terminal_shortage is not None:
        reported_terminal_shortage = float(reported_terminal_shortage)
        if abs(reported_terminal_shortage - terminal_ees_shortage_kwh) > 1e-4:
            print(
                "WARNING: Recomputed EES terminal shortage differs from env report: "
                f"case_index={case_idx}, recomputed={terminal_ees_shortage_kwh:.6f}, "
                f"reported={reported_terminal_shortage:.6f}",
                file=sys.stderr,
            )

    row = {
        "case_index": int(case_idx),
        "month": case.month,
        "day_of_year": case.day_of_year,
        "season": case.season,
        "split": case.set_name,
        "total_reward": float(total_reward),
        "total_reward_raw": sum_info(infos, "reward_raw"),
        "total_system_cost": sum_info(infos, "system_cost"),
        "total_penalties": sum_info(infos, "penalty_cost"),
        "total_guide_reward": sum_info(infos, "guide_reward"),
        "total_cost_grid": sum_info(infos, "cost_grid"),
        "total_cost_gas": sum_info(infos, "cost_gas"),
        "total_cost_deg": sum_info(infos, "cost_deg"),
        "total_cost_om": sum_info(infos, "cost_om"),
        "total_unmet_e": sum_info(infos, "unmet_e"),
        "total_unmet_h": sum_info(infos, "unmet_h"),
        "total_unmet_c": sum_info(infos, "unmet_c"),
        "total_surplus_e": sum_info(infos, "surplus_e"),
        "total_surplus_h": sum_info(infos, "surplus_h"),
        "total_surplus_c": sum_info(infos, "surplus_c"),
        "total_grid_overflow": sum_info(infos, "grid_overflow"),
        "total_penalty_unserved_e": sum_info(infos, "penalty_unserved_e"),
        "total_penalty_unserved_h": sum_info(infos, "penalty_unserved_h"),
        "total_penalty_unserved_c": sum_info(infos, "penalty_unserved_c"),
        "total_penalty_depart_energy": sum_info(infos, "penalty_depart_energy"),
        "total_penalty_depart_energy_soft": sum_info(infos, "penalty_depart_energy_soft"),
        "total_penalty_depart_energy_mid": sum_info(infos, "penalty_depart_energy_mid"),
        "total_penalty_depart_energy_hard": sum_info(infos, "penalty_depart_energy_hard"),
        "total_penalty_depart_risk": sum_info(infos, "penalty_depart_risk"),
        "total_penalty_surplus_e": sum_info(infos, "penalty_surplus_e"),
        "total_penalty_surplus_h": sum_info(infos, "penalty_surplus_h"),
        "total_penalty_surplus_c": sum_info(infos, "penalty_surplus_c"),
        "total_penalty_export_e": sum_info(infos, "penalty_export_e"),
        "total_penalty_ev_export_guard": sum_info(infos, "penalty_ev_export_guard"),
        "total_penalty_terminal_ees_soc": total_penalty_terminal_ees_soc,
        "total_depart_energy_shortage_kwh": sum_info(infos, "depart_energy_shortage_kwh"),
        "total_depart_shortage_soft_kwh": sum_info(infos, "depart_shortage_soft_kwh"),
        "total_depart_shortage_mid_kwh": sum_info(infos, "depart_shortage_mid_kwh"),
        "total_depart_shortage_hard_kwh": sum_info(infos, "depart_shortage_hard_kwh"),
        "total_depart_risk_energy_kwh": sum_info(infos, "depart_risk_energy_kwh"),
        "total_reward_storage_discharge_bonus": sum_info(
            infos, "reward_storage_discharge_bonus"
        ),
        "total_reward_storage_charge_bonus": sum_info(
            infos, "reward_storage_charge_bonus"
        ),
        "total_reward_ev_target_timing_bonus": sum_info(
            infos, "reward_ev_target_timing_bonus"
        ),
        "total_storage_peak_shaved_kwh": sum_info(infos, "storage_peak_shaved_kwh"),
        "total_storage_charge_rewarded_kwh": sum_info(
            infos, "storage_charge_rewarded_kwh"
        ),
        "total_ees_charge_rewarded_kwh": sum_info(infos, "ees_charge_rewarded_kwh"),
        "total_ev_flex_target_charge_kwh": sum_info(
            infos, "ev_flex_target_charge_kwh"
        ),
        "total_ev_buffer_charge_kwh": sum_info(infos, "ev_buffer_charge_kwh"),
        "total_low_value_charge_kwh": sum_info(infos, "low_value_charge_kwh"),
        "total_gt_export_clip": sum_info(infos, "p_gt_export_clip"),
        "total_gt_export_clip_steps": count_info(infos, "p_gt_export_clip"),
        "total_gt_safe_infeasible_steps": int(
            sum(1 for item in infos if not bool(item.get("gt_safe_feasible", True)))
        ),
        "total_grid_buy_kwh": sum_info(infos, "p_grid_buy") * float(dt),
        "total_grid_sell_kwh": sum_info(infos, "p_grid_sell") * float(dt),
        "avg_p_gt_kw": avg_info(infos, "p_gt"),
        "avg_p_ev_ch_kw": avg_info(infos, "p_ev_ch"),
        "avg_p_ev_dis_kw": avg_info(infos, "p_ev_dis"),
        "avg_p_ees_ch_kw": avg_info(infos, "p_ees_ch"),
        "avg_p_ees_dis_kw": avg_info(infos, "p_ees_dis"),
        "ees_soc_episode_init": float(
            infos[0].get("ees_soc_episode_init", 0.0) if infos else 0.0
        ),
        "final_ees_soc": final_ees_soc,
        "ees_soc_init": ees_soc_init,
        "terminal_ees_required_soc": terminal_ees_required_soc,
        "terminal_ees_shortage_kwh": terminal_ees_shortage_kwh,
        "ees_terminal_soc_feasible": ees_terminal_soc_feasible,
        "final_gt_power": float(final_info.get("p_gt", 0.0)),
    }
    return {column: row.get(column, 0.0) for column in DAILY_SUMMARY_COLUMNS}


def build_config_row(
    cfg: Any,
    *,
    run_id: str,
    timestamp_start: str,
    yearly_csv_path: Path,
    yearly_ev_path: Path,
    output_dir: Path,
    n_train_cases: int,
    n_val_cases: int,
    n_test_cases: int,
    ev_mode: str,
) -> dict[str, Any]:
    cfg_values = asdict(cfg)
    row = {
        "run_id": run_id,
        "timestamp_start": timestamp_start,
        "yearly_csv_path": str(yearly_csv_path),
        "yearly_ev_path": str(yearly_ev_path),
        "model_path": "not_applicable",
        "summary_csv_path": str(output_dir / SUMMARY_FILENAME),
        "timeseries_csv_path": str(output_dir / TIMESERIES_FILENAME),
        "test_export_xlsx_path": str(output_dir / EXCEL_FILENAME),
        "n_train_cases": int(n_train_cases),
        "n_val_cases": int(n_val_cases),
        "n_test_cases": int(n_test_cases),
        "SEED": 42,
        "VAL_DAYS_PER_MONTH": VAL_DAYS_PER_MONTH,
        "REWARD_SCALE": REWARD_SCALE,
        "algorithm": method_label(ev_mode),
        "policy_class": "TOU_SOC_Rule",
        "device": "not_applicable",
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
    return {column: row.get(column, "") for column in CONFIG_COLUMNS}


def numeric_series(pd, df, column: str):
    if column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def stats_text(pd, df, column: str) -> str:
    series = numeric_series(pd, df, column).dropna()
    if series.empty:
        return "mean=nan, std=nan, min=nan, max=nan"
    return (
        f"mean={float(series.mean()):.6f}, "
        f"std={float(series.std(ddof=0)):.6f}, "
        f"min={float(series.min()):.6f}, "
        f"max={float(series.max()):.6f}"
    )


def safe_sum(pd, df, column: str) -> float:
    series = numeric_series(pd, df, column)
    if series.empty:
        return 0.0
    return float(series.fillna(0.0).sum())


def safe_mean(pd, df, column: str) -> float:
    series = numeric_series(pd, df, column).dropna()
    if series.empty:
        return float("nan")
    return float(series.mean())


def safe_min(pd, df, column: str) -> float:
    series = numeric_series(pd, df, column).dropna()
    if series.empty:
        return float("nan")
    return float(series.min())


def boolean_series(pd, df, column: str):
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index, dtype=bool)
    series = df[column]
    if str(series.dtype) == "bool":
        return series.fillna(False).astype(bool)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes", "y"})


def terminal_ees_feasible_mask(pd, df):
    if df.empty:
        return pd.Series(dtype=bool)
    feasible = boolean_series(pd, df, "ees_terminal_soc_feasible")
    if "terminal_ees_shortage_kwh" in df.columns:
        shortage_ok = (
            numeric_series(pd, df, "terminal_ees_shortage_kwh")
            .fillna(float("inf"))
            .le(1e-6)
        )
        feasible = feasible & shortage_ok
    return feasible.astype(bool)


def aggregate_method_summary(pd, df, method: str) -> dict[str, Any]:
    system_cost = numeric_series(pd, df, "total_system_cost")
    penalties = numeric_series(pd, df, "total_penalties")
    cost_plus_penalty = (system_cost.fillna(0.0) + penalties.fillna(0.0)).dropna()
    terminal_feasible = terminal_ees_feasible_mask(pd, df)
    terminal_ees_feasible_days = int(terminal_feasible.sum()) if len(df) > 0 else 0

    row = {
        "method": method,
        "n_days": int(len(df)),
        "mean_total_system_cost": safe_mean(pd, df, "total_system_cost"),
        "mean_total_penalties": safe_mean(pd, df, "total_penalties"),
        "mean_total_cost_plus_penalty": float(cost_plus_penalty.mean())
        if not cost_plus_penalty.empty
        else float("nan"),
        "mean_total_grid_buy_kwh": safe_mean(pd, df, "total_grid_buy_kwh"),
        "mean_total_grid_sell_kwh": safe_mean(pd, df, "total_grid_sell_kwh"),
        "mean_total_depart_energy_shortage_kwh": safe_mean(
            pd, df, "total_depart_energy_shortage_kwh"
        ),
        "mean_total_unmet_e": safe_mean(pd, df, "total_unmet_e"),
        "mean_total_unmet_h": safe_mean(pd, df, "total_unmet_h"),
        "mean_total_unmet_c": safe_mean(pd, df, "total_unmet_c"),
        "mean_total_storage_peak_shaved_kwh": safe_mean(
            pd, df, "total_storage_peak_shaved_kwh"
        ),
        "mean_total_ev_flex_target_charge_kwh": safe_mean(
            pd, df, "total_ev_flex_target_charge_kwh"
        ),
        "mean_total_ev_buffer_charge_kwh": safe_mean(
            pd, df, "total_ev_buffer_charge_kwh"
        ),
        "mean_final_ees_soc": safe_mean(pd, df, "final_ees_soc"),
        "terminal_ees_feasible_days": terminal_ees_feasible_days,
        "terminal_ees_feasible_ratio": terminal_ees_feasible_days / max(int(len(df)), 1),
        "min_final_ees_soc": safe_min(pd, df, "final_ees_soc"),
        "sum_terminal_ees_shortage_kwh": safe_sum(
            pd, df, "terminal_ees_shortage_kwh"
        ),
        "sum_penalty_terminal_ees_soc": safe_sum(
            pd, df, "total_penalty_terminal_ees_soc"
        ),
        "total_depart_energy_shortage_kwh": safe_sum(
            pd, df, "total_depart_energy_shortage_kwh"
        ),
        "total_unmet_e": safe_sum(pd, df, "total_unmet_e"),
        "total_unmet_h": safe_sum(pd, df, "total_unmet_h"),
        "total_unmet_c": safe_sum(pd, df, "total_unmet_c"),
        "total_grid_sell_kwh": safe_sum(pd, df, "total_grid_sell_kwh"),
        "total_penalty_ev_export_guard": safe_sum(
            pd, df, "total_penalty_ev_export_guard"
        ),
    }
    return {column: row.get(column, float("nan")) for column in COMPARISON_COLUMNS}


def maybe_write_td3_comparison(pd, rule_summary_df, output_dir: Path, ev_mode: str):
    if not DEFAULT_TD3_SUMMARY_CSV.exists():
        return None, None

    td3_df = pd.read_csv(DEFAULT_TD3_SUMMARY_CSV)
    comparison_df = pd.DataFrame(
        [
            aggregate_method_summary(pd, rule_summary_df, method_label(ev_mode)),
            aggregate_method_summary(pd, td3_df, "TD3"),
        ],
        columns=COMPARISON_COLUMNS,
    )
    comparison_path = output_dir / COMPARISON_FILENAME
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    print(f"[export] Rule-based vs TD3 comparison saved to: {comparison_path}")
    return comparison_df, comparison_path


def dataframe_has_nan_or_inf(summary_df) -> tuple[bool, bool]:
    has_nan = bool(summary_df.isna().any().any())
    numeric_df = summary_df.select_dtypes(include=[np.number])
    has_inf = False
    if not numeric_df.empty:
        values = numeric_df.to_numpy(dtype=np.float64)
        has_inf = bool(np.isinf(values).any())
    return has_nan, has_inf


def write_diagnostics_report(
    pd,
    *,
    path: Path,
    summary_df,
    timeseries_df,
    cfg: Any,
    ev_mode: str,
    comparison_df=None,
    comparison_path: Path | None = None,
) -> None:
    n_days = int(len(summary_df))
    expected_rows = n_days * int(cfg.episode_length)
    actual_rows = int(len(timeseries_df))
    row_count_ok = actual_rows == expected_rows
    has_nan, has_inf = dataframe_has_nan_or_inf(summary_df)

    case_hour_counts = (
        timeseries_df.groupby("case_index").size()
        if "case_index" in timeseries_df.columns and not timeseries_df.empty
        else pd.Series(dtype="int64")
    )
    bad_case_hour_counts = case_hour_counts[
        case_hour_counts != int(cfg.episode_length)
    ]

    total_unmet_e = safe_sum(pd, summary_df, "total_unmet_e")
    total_unmet_h = safe_sum(pd, summary_df, "total_unmet_h")
    total_unmet_c = safe_sum(pd, summary_df, "total_unmet_c")
    total_unmet = total_unmet_e + total_unmet_h + total_unmet_c
    total_depart_shortage = safe_sum(
        pd, summary_df, "total_depart_energy_shortage_kwh"
    )
    total_grid_buy = safe_sum(pd, summary_df, "total_grid_buy_kwh")
    total_grid_sell = safe_sum(pd, summary_df, "total_grid_sell_kwh")
    total_gt_safe_infeasible = (
        safe_sum(pd, summary_df, "total_gt_safe_infeasible_steps")
        if "total_gt_safe_infeasible_steps" in summary_df.columns
        else None
    )

    final_ees_soc = numeric_series(pd, summary_df, "final_ees_soc").dropna()
    final_ees_soc_min = float(final_ees_soc.min()) if not final_ees_soc.empty else float("nan")
    final_ees_soc_mean = float(final_ees_soc.mean()) if not final_ees_soc.empty else float("nan")
    final_ees_soc_max = float(final_ees_soc.max()) if not final_ees_soc.empty else float("nan")
    terminal_feasible_mask = terminal_ees_feasible_mask(pd, summary_df)
    terminal_shortage = numeric_series(
        pd, summary_df, "terminal_ees_shortage_kwh"
    ).reindex(summary_df.index).fillna(float("inf"))
    terminal_ees_violation_mask = (
        (~terminal_feasible_mask.reindex(summary_df.index, fill_value=False))
        | (terminal_shortage > 1e-6)
    )
    terminal_ees_feasible_days = int(
        len(summary_df) - int(terminal_ees_violation_mask.sum())
    )
    terminal_ees_feasible_ratio = terminal_ees_feasible_days / max(n_days, 1)
    sum_terminal_ees_shortage_kwh = safe_sum(
        pd, summary_df, "terminal_ees_shortage_kwh"
    )
    sum_penalty_terminal_ees_soc = safe_sum(
        pd, summary_df, "total_penalty_terminal_ees_soc"
    )
    terminal_ees_ok = bool(
        terminal_ees_violation_mask.sum() == 0
        and sum_terminal_ees_shortage_kwh <= 1e-6
    )

    ees_soc_series = numeric_series(pd, timeseries_df, "ees_soc").dropna()
    ees_soc_out_of_bounds = False
    if not ees_soc_series.empty:
        ees_soc_out_of_bounds = bool(
            (ees_soc_series.min() < float(cfg.ees_soc_min) - DIAGNOSTIC_TOL)
            or (ees_soc_series.max() > float(cfg.ees_soc_max) + DIAGNOSTIC_TOL)
        )

    penalties = numeric_series(pd, summary_df, "total_penalties").fillna(0.0)
    system_cost = numeric_series(pd, summary_df, "total_system_cost").abs().fillna(0.0)
    penalty_ratio = penalties.abs() / system_cost.clip(lower=1.0)
    penalty_too_large = bool(
        (not penalty_ratio.empty and float(penalty_ratio.max()) > 10.0)
        or (not penalties.empty and float(penalties.max()) > 1e8)
    )

    def mark(condition: bool) -> str:
        return "PASS" if condition else "WARNING"

    lines = [
        f"# {method_label(ev_mode)} diagnostics",
        "",
        f"- method: {method_label(ev_mode)}",
        f"- ev_mode: `{ev_mode}`",
        f"- generated_at: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- test_days: {n_days}",
        f"- timeseries_rows: {actual_rows}",
        f"- expected_timeseries_rows: {expected_rows}",
        "",
        "## PASS / WARNING checks",
        f"- row_count: {mark(row_count_ok)} - expected {expected_rows}, got {actual_rows}",
        f"- complete_24h_cases: {mark(bad_case_hour_counts.empty)} - bad cases: {len(bad_case_hour_counts)}",
        f"- finite_daily_summary: {mark(not has_nan and not has_inf)} - has_nan={has_nan}, has_inf={has_inf}",
        f"- unmet_energy: {mark(total_unmet <= DIAGNOSTIC_TOL)} - total_unmet_e/h/c={total_unmet_e:.6f}/{total_unmet_h:.6f}/{total_unmet_c:.6f}",
        f"- ev_departure_shortage: {mark(total_depart_shortage <= DIAGNOSTIC_TOL)} - total_depart_energy_shortage_kwh={total_depart_shortage:.6f}",
        f"- ees_terminal_soc: {mark(terminal_ees_ok)} - feasible_days={terminal_ees_feasible_days}/{n_days}, shortage_kwh={sum_terminal_ees_shortage_kwh:.6f}, penalty={sum_penalty_terminal_ees_soc:.6f}",
        f"- ees_soc_bounds: {mark(not ees_soc_out_of_bounds)} - configured bounds=[{cfg.ees_soc_min:.6f}, {cfg.ees_soc_max:.6f}]",
        f"- penalty_scale: {mark(not penalty_too_large)} - max penalty/system_cost ratio={float(penalty_ratio.max()) if not penalty_ratio.empty else float('nan'):.6f}",
        "",
        "## Summary statistics",
        f"- total_system_cost: {stats_text(pd, summary_df, 'total_system_cost')}",
        f"- total_penalties: {stats_text(pd, summary_df, 'total_penalties')}",
        f"- total_unmet_e_sum: {total_unmet_e:.6f}",
        f"- total_unmet_h_sum: {total_unmet_h:.6f}",
        f"- total_unmet_c_sum: {total_unmet_c:.6f}",
        f"- total_depart_energy_shortage_kwh_sum: {total_depart_shortage:.6f}",
        f"- total_grid_buy_kwh_sum: {total_grid_buy:.6f}",
        f"- total_grid_sell_kwh_sum: {total_grid_sell:.6f}",
        f"- final_ees_soc_min: {final_ees_soc_min:.6f}",
        f"- final_ees_soc_mean: {final_ees_soc_mean:.6f}",
        f"- final_ees_soc_max: {final_ees_soc_max:.6f}",
        f"- terminal_ees_feasible_days: {terminal_ees_feasible_days}",
        f"- terminal_ees_feasible_ratio: {terminal_ees_feasible_ratio:.6f}",
        f"- terminal_ees_shortage_kwh_sum: {sum_terminal_ees_shortage_kwh:.6f}",
        f"- penalty_terminal_ees_soc_sum: {sum_penalty_terminal_ees_soc:.6f}",
    ]
    if total_gt_safe_infeasible is not None:
        lines.append(
            f"- total_gt_safe_infeasible_steps_sum: {total_gt_safe_infeasible:.0f}"
        )

    lines.extend(
        [
            "",
            "## Obvious anomaly notes",
            f"- missing_24h_data: {'yes' if not bad_case_hour_counts.empty else 'no'}",
            f"- large_unserved_energy: {'yes' if total_unmet > DIAGNOSTIC_TOL else 'no'}",
            f"- large_ev_departure_shortage: {'yes' if total_depart_shortage > DIAGNOSTIC_TOL else 'no'}",
            f"- large_penalties: {'yes' if penalty_too_large else 'no'}",
            f"- ees_soc_out_of_bounds: {'yes' if ees_soc_out_of_bounds else 'no'}",
            f"- ees_terminal_soc_violation: {'yes' if not terminal_ees_ok else 'no'}",
        ]
    )

    violation_cases = summary_df.loc[
        terminal_ees_violation_mask,
        [
            column
            for column in (
                "case_index",
                "final_ees_soc",
                "terminal_ees_required_soc",
                "terminal_ees_shortage_kwh",
                "total_penalty_terminal_ees_soc",
            )
            if column in summary_df.columns
        ],
    ]
    lines.extend(["", "## EES terminal SOC violations"])
    if violation_cases.empty:
        lines.append("- none")
    else:
        lines.extend(
            [
                "",
                "| case_index | final_ees_soc | terminal_ees_required_soc | terminal_ees_shortage_kwh | total_penalty_terminal_ees_soc |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for _, item in violation_cases.iterrows():
            lines.append(
                "| "
                f"{int(item.get('case_index', -1))} | "
                f"{float(item.get('final_ees_soc', float('nan'))):.6f} | "
                f"{float(item.get('terminal_ees_required_soc', float('nan'))):.6f} | "
                f"{float(item.get('terminal_ees_shortage_kwh', float('nan'))):.6f} | "
                f"{float(item.get('total_penalty_terminal_ees_soc', float('nan'))):.6f} |"
            )

    if comparison_df is not None and not comparison_df.empty:
        lines.extend(
            [
                "",
                comparison_heading(ev_mode),
                "",
                "Guide reward is not treated as real economic revenue here. The comparison focuses on cost, penalties, energy service, grid exchange, EV shortage, and storage support.",
            ]
        )
        if comparison_path is not None:
            lines.append(f"- comparison_csv: {comparison_path}")

        metric_cols = [column for column in COMPARISON_COLUMNS if column not in {"method", "n_days"}]
        method_values = comparison_df.set_index("method")
        lines.extend(["", "| metric | Rule-based | TD3 | delta_rule_minus_td3 |", "|---|---:|---:|---:|"])
        rule_method = method_label(ev_mode)
        for column in metric_cols:
            rule_value = (
                float(method_values.loc[rule_method, column])
                if rule_method in method_values.index
                else float("nan")
            )
            td3_value = (
                float(method_values.loc["TD3", column])
                if "TD3" in method_values.index
                else float("nan")
            )
            delta = rule_value - td3_value
            lines.append(f"| {column} | {rule_value:.6f} | {td3_value:.6f} | {delta:.6f} |")
    else:
        lines.extend(
            [
                "",
                comparison_heading(ev_mode),
                "",
                f"TD3 summary not found at `{DEFAULT_TD3_SUMMARY_CSV}`; comparison skipped.",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[export] Diagnostics report saved to: {path}")


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
    args = build_arg_parser().parse_args()

    try:
        preflight_check(args.yearly_csv, args.yearly_ev)
        modules = import_runtime_modules()
    except Exception as exc:
        print(f"[startup] Rule-based test startup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    random.seed(42)
    np.random.seed(42)

    pd = modules["pd"]
    IESConfig = modules["IESConfig"]
    ParkIESEnv = modules["ParkIESEnv"]
    YearlyCSVDataLoader = modules["YearlyCSVDataLoader"]
    YearlyEVProvider = modules["YearlyEVProvider"]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / SUMMARY_FILENAME
    timeseries_csv = output_dir / TIMESERIES_FILENAME
    runtime_summary_json = output_dir / RUNTIME_SUMMARY_FILENAME
    diagnostics_md = output_dir / DIAGNOSTICS_FILENAME
    excel_path = output_dir / EXCEL_FILENAME

    print("[startup] Loading yearly IES data...")
    loader = YearlyCSVDataLoader(args.yearly_csv, val_days_per_month=VAL_DAYS_PER_MONTH)
    train_cases, val_cases, test_cases = loader.load()
    loader.summary()
    if not test_cases:
        raise RuntimeError("Test set is empty. Please check the split configuration.")

    print("[startup] Loading yearly EV data...")
    ev_provider = YearlyEVProvider(args.yearly_ev)

    cfg = IESConfig(episode_length=24, dt=1.0, reward_scale=REWARD_SCALE)
    probe_case = test_cases[0]
    probe_env = ParkIESEnv(
        cfg=cfg,
        ts_data=probe_case.ts_data,
        ev_data=ev_provider(probe_case),
    )
    print("[startup] Rule-based test configuration:")
    print(f"  method = {method_label(args.ev_mode)}")
    print(f"  ev_mode = {args.ev_mode}")
    print(f"  reward_scale = {cfg.reward_scale}")
    print(f"  obs_dim = {probe_env.obs_dim}")
    print(f"  action_space = {probe_env.action_space}")
    probe_env.close()

    run_started_at = datetime.now().astimezone()
    run_id = run_started_at.strftime("%Y%m%d_%H%M%S")
    timestamp_start = run_started_at.isoformat(timespec="seconds")

    summary_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    test_started_perf = perf_counter()
    for case_idx, case in enumerate(test_cases):
        env = ParkIESEnv(
            cfg=cfg,
            ts_data=case.ts_data,
            ev_data=ev_provider(case),
        )
        try:
            infos, total_reward = rollout_rule_based_one_day(env, ev_mode=args.ev_mode)
            summary_row = build_daily_summary_row(
                case_idx=case_idx,
                case=case,
                infos=infos,
                total_reward=total_reward,
                dt=cfg.dt,
                cfg=cfg,
            )
            summary_rows.append(summary_row)
            for step_info in infos:
                step_rows.append(build_step_row(case_idx, case, step_info))
        finally:
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

    save_csv(summary_rows, summary_csv, DAILY_SUMMARY_COLUMNS)
    save_csv(step_rows, timeseries_csv, STEP_DETAIL_COLUMNS)
    runtime_summary = write_runtime_summary(
        runtime_summary_json,
        method=method_label(args.ev_mode),
        n_days=len(summary_rows),
        n_steps=len(step_rows),
        test_duration_seconds=test_duration_seconds,
    )
    print(f"[export] Daily summary CSV saved to: {summary_csv}")
    print(f"[export] Timeseries detail CSV saved to: {timeseries_csv}")
    print(f"[export] Runtime summary JSON saved to: {runtime_summary_json}")

    summary_df = pd.DataFrame(summary_rows, columns=DAILY_SUMMARY_COLUMNS)
    timeseries_df = pd.DataFrame(step_rows, columns=STEP_DETAIL_COLUMNS)
    comparison_df, comparison_path = maybe_write_td3_comparison(
        pd, summary_df, output_dir, args.ev_mode
    )

    config_row = build_config_row(
        cfg,
        run_id=run_id,
        timestamp_start=timestamp_start,
        yearly_csv_path=args.yearly_csv,
        yearly_ev_path=args.yearly_ev,
        output_dir=output_dir,
        n_train_cases=len(train_cases),
        n_val_cases=len(val_cases),
        n_test_cases=len(test_cases),
        ev_mode=args.ev_mode,
    )

    excel_engine = resolve_optional_excel_engine()
    if excel_engine is not None:
        try:
            export_test_workbook(
                pd,
                output_path=excel_path,
                excel_engine=excel_engine,
                step_rows=step_rows,
                summary_rows=summary_rows,
                config_row=config_row,
                alert_rows=alert_rows,
            )
        except Exception as exc:
            print(f"[export] Excel export skipped after failure: {exc}")

    write_diagnostics_report(
        pd,
        path=diagnostics_md,
        summary_df=summary_df,
        timeseries_df=timeseries_df,
        cfg=cfg,
        ev_mode=args.ev_mode,
        comparison_df=comparison_df,
        comparison_path=comparison_path,
    )

    print("\n========== Rule-based Test Set Summary ==========")
    print(f"Method: {method_label(args.ev_mode)}")
    print(f"Test days: {len(summary_rows)}")
    print(f"Timeseries rows: {len(step_rows)}")
    print(
        "Test runtime: "
        f"{runtime_summary['test_duration_seconds']:.6f} s, "
        f"{runtime_summary['time_per_day_seconds']:.6f} s/day, "
        f"{runtime_summary['time_per_step_seconds']:.6f} s/step"
    )
    print(f"Total system cost: {safe_sum(pd, summary_df, 'total_system_cost'):.4f}")
    print(f"Total penalties: {safe_sum(pd, summary_df, 'total_penalties'):.4f}")
    print(
        "Total depart energy shortage (kWh): "
        f"{safe_sum(pd, summary_df, 'total_depart_energy_shortage_kwh'):.4f}"
    )
    key_metrics = aggregate_method_summary(pd, summary_df, method_label(args.ev_mode))
    print("\n========== Rule-based-V2G SCI Baseline Metrics ==========")
    for metric in (
        "method",
        "n_days",
        "mean_total_system_cost",
        "mean_total_penalties",
        "mean_total_cost_plus_penalty",
        "terminal_ees_feasible_days",
        "terminal_ees_feasible_ratio",
        "min_final_ees_soc",
        "sum_terminal_ees_shortage_kwh",
        "sum_penalty_terminal_ees_soc",
        "total_depart_energy_shortage_kwh",
        "total_unmet_e",
        "total_unmet_h",
        "total_unmet_c",
        "total_grid_sell_kwh",
        "total_penalty_ev_export_guard",
    ):
        value = key_metrics.get(metric)
        if isinstance(value, float):
            print(f"{metric}: {value:.6f}")
        else:
            print(f"{metric}: {value}")
    print(f"Alert rows exported: {len(alert_rows)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
