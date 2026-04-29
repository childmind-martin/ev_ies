from __future__ import annotations

import csv
import importlib.util
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any


YEARLY_CSV_PATH = Path(r"D:\wangye\chengxu\3.30\yearly_data_sci.csv")
YEARLY_EV_PATH = Path(r"D:\wangye\chengxu\3.30\ev_data_s123_bilevel.npy")
MODEL_PATH = Path("./models/td3_yearly_single/best/best_model.zip")

RESULT_DIR = Path("./results/td3_yearly_test")
SUMMARY_CSV = RESULT_DIR / "daily_summary.csv"
TIMESERIES_CSV = RESULT_DIR / "timeseries_detail.csv"
TEST_EXPORT_XLSX = RESULT_DIR / "td3_test_export.xlsx"
RUNTIME_SUMMARY_JSON = RESULT_DIR / "runtime_summary.json"

SEED = 42
VAL_DAYS_PER_MONTH = 2
REWARD_SCALE = 1e-5
ALERT_TOL = 1e-8
REQUESTED_DEVICE = "auto"

STEP_DETAIL_COLUMNS = (
    "case_index", "month", "day_of_year", "season", "split",
    "time_step", "terminated", "truncated",
    "a_ees", "a_gt", "a_ev_ch", "a_ev_dis",
    "elec_load", "heat_load", "cool_load", "pv", "wt", "grid_buy_price", "grid_sell_price", "gas_price",
    "p_grid", "p_grid_buy", "p_grid_sell", "p_gt",
    "p_whb_heat", "p_gb_heat", "p_ac_cool", "p_ec_cool", "p_ec_elec_in",
    "p_ev_ch", "p_ev_rigid_ch", "p_ev_flex_target_ch", "p_ev_buffer_ch", "p_ev_dis",
    "p_ees_ch", "p_ees_dis",
    "ees_soc", "ees_soc_episode_init", "ev_soc_mean",
    "cost_grid", "cost_gas", "cost_deg", "cost_om", "system_cost",
    "penalty_cost", "penalty_unserved_e", "penalty_unserved_h", "penalty_unserved_c",
    "penalty_depart_energy", "penalty_depart_energy_soft", "penalty_depart_energy_mid", "penalty_depart_energy_hard", "penalty_depart_risk",
    "penalty_surplus_e", "penalty_surplus_h", "penalty_surplus_c", "penalty_export_e", "penalty_ev_export_guard",
    "guide_reward", "reward_raw", "reward_scaled", "reward_storage_discharge_bonus", "reward_storage_charge_bonus", "reward_ev_target_timing_bonus",
    "unmet_e", "unmet_h", "unmet_c", "surplus_e", "surplus_h", "surplus_c", "grid_overflow",
    "depart_energy_shortage_kwh", "depart_shortage_soft_kwh", "depart_shortage_mid_kwh", "depart_shortage_hard_kwh",
    "depart_risk_energy_kwh", "depart_risk_vehicle_count", "ev_export_overlap_kwh",
    "storage_peak_shaved_kwh", "storage_charge_rewarded_kwh", "ees_charge_rewarded_kwh",
    "ev_flex_target_charge_kwh", "ev_buffer_charge_kwh", "low_value_charge_kwh",
    "ev_peak_pressure_without_storage_kw", "low_value_energy_before_storage_prepare_kwh",
    "ees_discharge_reward_weight", "ees_charge_reward_weight",
    "p_gt_safe_min", "p_gt_safe_max", "p_gt_export_clip", "gt_safe_feasible",
)

STEP_METADATA_COLUMNS = ("case_index", "month", "day_of_year", "season", "split")
STEP_INFO_COLUMNS = tuple(column for column in STEP_DETAIL_COLUMNS if column not in STEP_METADATA_COLUMNS)

DAILY_SUMMARY_COLUMNS = (
    "case_index", "month", "day_of_year", "season", "split",
    "total_reward", "total_reward_raw", "total_system_cost", "total_penalties", "total_guide_reward",
    "total_cost_grid", "total_cost_gas", "total_cost_deg", "total_cost_om",
    "total_unmet_e", "total_unmet_h", "total_unmet_c",
    "total_surplus_e", "total_surplus_h", "total_surplus_c",
    "total_grid_overflow",
    "total_penalty_unserved_e", "total_penalty_unserved_h", "total_penalty_unserved_c",
    "total_penalty_depart_energy", "total_penalty_depart_energy_soft", "total_penalty_depart_energy_mid", "total_penalty_depart_energy_hard",
    "total_penalty_depart_risk", "total_penalty_surplus_e", "total_penalty_surplus_h", "total_penalty_surplus_c",
    "total_penalty_export_e", "total_penalty_ev_export_guard", "total_penalty_terminal_ees_soc",
    "total_depart_energy_shortage_kwh", "total_depart_shortage_soft_kwh", "total_depart_shortage_mid_kwh", "total_depart_shortage_hard_kwh",
    "total_depart_risk_energy_kwh",
    "total_reward_storage_discharge_bonus", "total_reward_storage_charge_bonus", "total_reward_ev_target_timing_bonus",
    "total_storage_peak_shaved_kwh", "total_storage_charge_rewarded_kwh", "total_ees_charge_rewarded_kwh",
    "total_ev_flex_target_charge_kwh", "total_ev_buffer_charge_kwh", "total_low_value_charge_kwh",
    "total_gt_export_clip", "total_gt_export_clip_steps", "total_gt_safe_infeasible_steps",
    "total_grid_buy_kwh", "total_grid_sell_kwh",
    "avg_p_gt_kw", "avg_p_ev_ch_kw", "avg_p_ev_dis_kw", "avg_p_ees_ch_kw", "avg_p_ees_dis_kw",
    "ees_soc_episode_init", "final_ees_soc", "ees_soc_init", "terminal_ees_required_soc",
    "terminal_ees_shortage_kwh", "ees_terminal_soc_feasible", "final_gt_power",
)

CONFIG_COLUMNS = (
    "run_id", "timestamp_start",
    "yearly_csv_path", "yearly_ev_path", "model_path",
    "summary_csv_path", "timeseries_csv_path", "test_export_xlsx_path",
    "n_train_cases", "n_val_cases", "n_test_cases",
    "SEED", "VAL_DAYS_PER_MONTH", "REWARD_SCALE",
    "algorithm", "policy_class", "device",
    "episode_length", "dt", "future_horizon", "exogenous_future_horizon",
    "grid_import_max", "grid_export_max",
    "gt_p_max", "gt_eta_e", "gt_ramp",
    "gb_h_max", "gb_ramp", "ac_c_max", "ec_c_max",
    "ees_p_max", "ees_e_cap", "ees_soc_init", "ees_soc_min", "ees_soc_max",
    "penalty_unserved_e", "penalty_unserved_h", "penalty_unserved_c", "penalty_depart_soc",
    "penalty_depart_energy_soft", "penalty_depart_energy_mid", "penalty_ev_depart_risk",
    "penalty_surplus_e", "penalty_surplus_h", "penalty_surplus_c",
    "penalty_export_e", "penalty_ev_export_guard", "penalty_ees_terminal_soc",
    "ees_terminal_soc_tolerance",
    "ev_discharge_price_threshold", "ev_charge_price_threshold", "ev_peak_export_tolerance_kw",
    "reward_storage_discharge_base", "reward_storage_charge_base", "reward_ev_target_timing_base",
    "ees_reward_discharge_soc_floor", "ees_reward_charge_soc_target", "reward_scale",
)

ALERT_COLUMNS = (
    "case_index", "month", "day_of_year", "season", "split", "time_step",
    "alert_count", "alert_tags",
    "has_unmet", "has_surplus", "has_grid_overflow", "has_gt_export_clip",
    "has_gt_safe_infeasible", "has_depart_energy_shortage",
    "unmet_e", "unmet_h", "unmet_c", "surplus_e", "surplus_h", "surplus_c",
    "grid_overflow", "depart_energy_shortage_kwh", "p_gt_export_clip",
    "gt_safe_feasible",
    "penalty_cost", "reward_raw", "reward_scaled",
    "p_grid", "p_gt", "p_ev_ch", "p_ev_dis", "p_ees_ch", "p_ees_dis",
    "ees_soc", "ev_soc_mean",
)


def configure_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)


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
            "This project should be run with Python 3.9-3.11."
        )

    required_paths = [YEARLY_CSV_PATH, YEARLY_EV_PATH, MODEL_PATH]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"缂哄皯杈撳叆鏂囦欢: {', '.join(missing_paths)}")

    excel_engine = resolve_excel_engine()
    required_modules = [
        "numpy",
        "torch",
        "stable_baselines3",
        "gymnasium",
        "pandas",
    ]
    missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing_modules:
        raise ModuleNotFoundError(
            "Missing required packages: "
            f"{', '.join(missing_modules)}. "
            "Please install the dependencies before testing."
        )
    return excel_engine


def import_runtime_modules():
    print("[startup] Importing test dependencies...")

    import numpy as np
    import pandas as pd
    import torch as th
    from stable_baselines3 import TD3

    from park_ies_env import IESConfig, ParkIESEnv
    from yearly_case_env import YearlyEVProvider
    from yearly_csv_loader import YearlyCSVDataLoader

    print("[startup] Test dependencies imported.")
    return {
        "np": np,
        "pd": pd,
        "th": th,
        "TD3": TD3,
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
            "Please install a CUDA-enabled PyTorch build and confirm the NVIDIA driver is working."
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
        print("  note = CUDA unavailable, so TD3 will run on CPU.")


def rollout_one_day(model, env):
    obs, _ = env.reset()
    done = False
    infos = []
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        infos.append(step_info)
        total_reward += float(reward)
        done = bool(terminated or truncated)

    return infos, total_reward


def save_csv(rows: list[dict[str, Any]], path: Path, columns: tuple[str, ...] | None = None) -> None:
    if not rows:
        return
    if columns is None:
        fieldnames: list[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key in seen:
                    continue
                seen.add(key)
                fieldnames.append(key)
    else:
        fieldnames = list(columns)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_unit(column_name: str) -> str:
    if column_name in {"timestamp_start"}:
        return "datetime"
    if column_name.endswith("_path"):
        return "path"
    if column_name.startswith("has_") or column_name in {
        "terminated", "truncated", "gt_low_price_active", "gt_thermal_feasible",
        "gt_electric_feasible", "gt_safe_feasible", "ees_terminal_soc_feasible"
    }:
        return "bool"
    if column_name in {
        "case_index", "month", "day_of_year", "time_step", "SEED", "VAL_DAYS_PER_MONTH",
        "n_train_cases", "n_val_cases", "n_test_cases", "alert_count"
    } or column_name.endswith("_count") or column_name.endswith("_steps"):
        return "count"
    if column_name in {"season", "split", "run_id", "algorithm", "policy_class", "device", "alert_tags"}:
        return "label"
    if "price" in column_name:
        return "price/kWh"
    if column_name.endswith("_kw") or column_name.startswith("p_") or column_name.endswith("_load") or column_name in {
        "pv", "wt", "unmet_e", "unmet_h", "unmet_c", "surplus_e", "surplus_h", "surplus_c",
        "grid_overflow",
        "final_gt_power", "ev_peak_pressure_without_storage_kw"
    }:
        return "kW"
    if column_name.endswith("_kwh") or column_name in {
        "depart_energy_shortage_kwh", "depart_shortage_soft_kwh", "depart_shortage_mid_kwh",
        "depart_shortage_hard_kwh", "depart_risk_energy_kwh", "ev_export_overlap_kwh",
        "ev_flex_target_charge_kwh", "ev_buffer_charge_kwh",
        "low_value_charge_kwh", "total_grid_buy_kwh", "total_grid_sell_kwh", "ees_e_cap"
    }:
        return "kWh"
    if "soc" in column_name:
        return "p.u."
    if column_name.startswith("cost_") or column_name.startswith("total_cost_") or column_name in {
        "system_cost", "penalty_cost", "total_system_cost", "total_penalties"
    } or column_name.startswith("penalty_") or column_name.startswith("total_penalty_"):
        return "cost"
    if column_name.startswith("reward_") or column_name.startswith("total_reward_") or column_name in {
        "guide_reward", "reward_raw", "reward_scaled", "total_reward", "total_reward_raw", "total_guide_reward"
    }:
        return "reward"
    return ""


def describe_step_column(column_name: str) -> tuple[str, str]:
    if column_name in STEP_METADATA_COLUMNS:
        return f"Test sample metadata field `{column_name}` for the current step.", "test_TD3.py"
    if column_name == "time_step":
        return "Zero-based hour index inside the tested day.", "ParkIESEnv.step()"
    if column_name in {"terminated", "truncated"}:
        return f"Episode termination flag `{column_name}` for the current step.", "ParkIESEnv.step()"
    if column_name.startswith("a_"):
        return f"Executed TD3 action component `{column_name}` after clipping.", "ParkIESEnv.step()"
    if column_name.startswith("cost_") or column_name in {"system_cost", "penalty_cost"}:
        return f"Per-step cost metric `{column_name}`.", "ParkIESEnv.step()"
    if column_name.startswith("penalty_"):
        return f"Per-step penalty component `{column_name}`.", "ParkIESEnv.step()"
    if column_name.startswith("reward_") or column_name in {"guide_reward", "reward_raw", "reward_scaled"}:
        return f"Per-step reward-related metric `{column_name}`.", "ParkIESEnv.step()"
    if column_name.startswith("p_"):
        return f"Per-step dispatch, command, or capacity metric `{column_name}`.", "ParkIESEnv.step()"
    if "soc" in column_name:
        return f"SOC-related metric `{column_name}` recorded at the current step.", "ParkIESEnv.step()"
    if column_name.endswith("_count"):
        return f"Per-step count metric `{column_name}`.", "ParkIESEnv.step()"
    return f"Per-step exported metric `{column_name}`.", "ParkIESEnv.step()"


def describe_summary_column(column_name: str) -> tuple[str, str]:
    if column_name in {"case_index", "month", "day_of_year", "season", "split"}:
        return f"Test sample metadata field `{column_name}` for the whole day.", "test_TD3.py"
    if column_name.startswith("total_"):
        return f"Day-level aggregated metric `{column_name}` summed or integrated across the tested day.", "test_TD3.py"
    if column_name.startswith("avg_"):
        return f"Day-level average power metric `{column_name}` over the tested day.", "test_TD3.py"
    if column_name.startswith("final_") or column_name == "ees_soc_episode_init":
        return f"Day-level initial or terminal metric `{column_name}`.", "test_TD3.py"
    return f"Daily summary metric `{column_name}`.", "test_TD3.py"


def describe_config_column(column_name: str) -> tuple[str, str]:
    if column_name.isupper():
        return f"Test constant `{column_name}` recorded for reproducibility.", "test_TD3.py"
    if column_name.endswith("_path"):
        return f"Filesystem path `{column_name}` used by this test export.", "test_TD3.py"
    if column_name in {"run_id", "timestamp_start", "algorithm", "policy_class", "device"}:
        return f"Runtime metadata field `{column_name}` for this test run.", "test_TD3.py"
    if column_name.endswith("_cases"):
        return f"Dataset split count `{column_name}` observed at startup.", "YearlyCSVDataLoader"
    return f"Environment or reward configuration field `{column_name}` recorded for reproducibility.", "IESConfig/test_TD3.py"


def describe_alert_column(column_name: str) -> tuple[str, str]:
    if column_name in {"case_index", "month", "day_of_year", "season", "split", "time_step"}:
        return f"Metadata field `{column_name}` locating the abnormal step.", "alerts builder"
    if column_name in {"alert_count", "alert_tags"}:
        return f"Summary field `{column_name}` listing the triggered alert conditions for the step.", "alerts builder"
    if column_name.startswith("has_"):
        return f"Boolean alert flag `{column_name}` for the abnormal step.", "alerts builder"
    return f"Diagnostic metric `{column_name}` copied into the alerts sheet for quick troubleshooting.", "alerts builder"


def build_column_description_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for column_name in STEP_DETAIL_COLUMNS:
        meaning, source = describe_step_column(column_name)
        rows.append({"sheet": "step_detail", "column_name": column_name, "unit": infer_unit(column_name), "meaning": meaning, "source": source})
    for column_name in DAILY_SUMMARY_COLUMNS:
        meaning, source = describe_summary_column(column_name)
        rows.append({"sheet": "daily_summary", "column_name": column_name, "unit": infer_unit(column_name), "meaning": meaning, "source": source})
    for column_name in CONFIG_COLUMNS:
        meaning, source = describe_config_column(column_name)
        rows.append({"sheet": "config", "column_name": column_name, "unit": infer_unit(column_name), "meaning": meaning, "source": source})
    for column_name in ALERT_COLUMNS:
        meaning, source = describe_alert_column(column_name)
        rows.append({"sheet": "alerts", "column_name": column_name, "unit": infer_unit(column_name), "meaning": meaning, "source": source})
    return rows


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


def build_step_row(case_idx: int, case: Any, step_info: dict[str, Any]) -> dict[str, Any]:
    missing = [column for column in STEP_INFO_COLUMNS if column not in step_info]
    if missing:
        raise KeyError(f"Missing required step_detail columns in env info: {missing}")
    row = {
        "case_index": int(case_idx),
        "month": case.month,
        "day_of_year": case.day_of_year,
        "season": case.season,
        "split": case.set_name,
    }
    for column in STEP_INFO_COLUMNS:
        row[column] = step_info.get(column)
    return row


def build_alert_row(step_row: dict[str, Any]) -> dict[str, Any] | None:
    tags: list[str] = []
    has_unmet = any(float(step_row.get(key, 0.0)) > ALERT_TOL for key in ("unmet_e", "unmet_h", "unmet_c"))
    has_surplus = any(float(step_row.get(key, 0.0)) > ALERT_TOL for key in ("surplus_e", "surplus_h", "surplus_c"))
    has_grid_overflow = float(step_row.get("grid_overflow", 0.0)) > ALERT_TOL
    has_gt_export_clip = float(step_row.get("p_gt_export_clip", 0.0)) > ALERT_TOL
    has_gt_safe_infeasible = not bool(step_row.get("gt_safe_feasible", True))
    has_depart_energy_shortage = float(step_row.get("depart_energy_shortage_kwh", 0.0)) > ALERT_TOL

    for key in ("unmet_e", "unmet_h", "unmet_c"):
        if float(step_row.get(key, 0.0)) > ALERT_TOL:
            tags.append(key)
    for key in ("surplus_e", "surplus_h", "surplus_c"):
        if float(step_row.get(key, 0.0)) > ALERT_TOL:
            tags.append(key)
    if has_grid_overflow:
        tags.append("grid_overflow")
    if has_gt_export_clip:
        tags.append("p_gt_export_clip")
    if has_gt_safe_infeasible:
        tags.append("gt_safe_infeasible")
    if has_depart_energy_shortage:
        tags.append("depart_energy_shortage_kwh")

    if not tags:
        return None

    return {
        "case_index": step_row["case_index"],
        "month": step_row["month"],
        "day_of_year": step_row["day_of_year"],
        "season": step_row["season"],
        "split": step_row["split"],
        "time_step": step_row["time_step"],
        "alert_count": len(tags),
        "alert_tags": "; ".join(tags),
        "has_unmet": has_unmet,
        "has_surplus": has_surplus,
        "has_grid_overflow": has_grid_overflow,
        "has_gt_export_clip": has_gt_export_clip,
        "has_gt_safe_infeasible": has_gt_safe_infeasible,
        "has_depart_energy_shortage": has_depart_energy_shortage,
        "unmet_e": float(step_row.get("unmet_e", 0.0)),
        "unmet_h": float(step_row.get("unmet_h", 0.0)),
        "unmet_c": float(step_row.get("unmet_c", 0.0)),
        "surplus_e": float(step_row.get("surplus_e", 0.0)),
        "surplus_h": float(step_row.get("surplus_h", 0.0)),
        "surplus_c": float(step_row.get("surplus_c", 0.0)),
        "grid_overflow": float(step_row.get("grid_overflow", 0.0)),
        "depart_energy_shortage_kwh": float(step_row.get("depart_energy_shortage_kwh", 0.0)),
        "p_gt_export_clip": float(step_row.get("p_gt_export_clip", 0.0)),
        "gt_safe_feasible": bool(step_row.get("gt_safe_feasible", True)),
        "penalty_cost": float(step_row.get("penalty_cost", 0.0)),
        "reward_raw": float(step_row.get("reward_raw", 0.0)),
        "reward_scaled": float(step_row.get("reward_scaled", 0.0)),
        "p_grid": float(step_row.get("p_grid", 0.0)),
        "p_gt": float(step_row.get("p_gt", 0.0)),
        "p_ev_ch": float(step_row.get("p_ev_ch", 0.0)),
        "p_ev_dis": float(step_row.get("p_ev_dis", 0.0)),
        "p_ees_ch": float(step_row.get("p_ees_ch", 0.0)),
        "p_ees_dis": float(step_row.get("p_ees_dis", 0.0)),
        "ees_soc": float(step_row.get("ees_soc", 0.0)),
        "ev_soc_mean": float(step_row.get("ev_soc_mean", 0.0)),
    }


def build_alert_rows(step_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alert_rows: list[dict[str, Any]] = []
    for step_row in step_rows:
        alert_row = build_alert_row(step_row)
        if alert_row is not None:
            alert_rows.append(alert_row)
    return alert_rows


def export_test_workbook(
    pd,
    *,
    output_path: Path,
    excel_engine: str,
    step_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    config_row: dict[str, Any],
    alert_rows: list[dict[str, Any]],
) -> None:
    column_description_rows = build_column_description_rows()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    step_df = pd.DataFrame(step_rows, columns=STEP_DETAIL_COLUMNS)
    summary_df = pd.DataFrame(summary_rows, columns=DAILY_SUMMARY_COLUMNS)
    config_df = pd.DataFrame([{column: config_row.get(column) for column in CONFIG_COLUMNS}], columns=CONFIG_COLUMNS)
    alert_df = pd.DataFrame(alert_rows, columns=ALERT_COLUMNS)
    description_df = pd.DataFrame(column_description_rows, columns=["sheet", "column_name", "unit", "meaning", "source"])

    with pd.ExcelWriter(output_path, engine=excel_engine) as writer:
        step_df.to_excel(writer, sheet_name="step_detail", index=False)
        summary_df.to_excel(writer, sheet_name="daily_summary", index=False)
        config_df.to_excel(writer, sheet_name="config", index=False)
        description_df.to_excel(writer, sheet_name="column_description", index=False)
        alert_df.to_excel(writer, sheet_name="alerts", index=False)

    print(f"[export] Test diagnostics workbook saved to: {output_path.resolve()}")


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
        print(f"[startup] 鍚姩澶辫触: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    np = modules["np"]
    pd = modules["pd"]
    th = modules["th"]
    TD3 = modules["TD3"]
    IESConfig = modules["IESConfig"]
    ParkIESEnv = modules["ParkIESEnv"]
    YearlyCSVDataLoader = modules["YearlyCSVDataLoader"]
    YearlyEVProvider = modules["YearlyEVProvider"]

    set_random_seed(SEED, np, th)
    resolved_device = resolve_sb3_device(th, REQUESTED_DEVICE)
    print_torch_runtime_summary(th, requested_device=REQUESTED_DEVICE, resolved_device=resolved_device)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("[startup] 姝ｅ湪鍔犺浇娴嬭瘯鏁版嵁...")
    loader = YearlyCSVDataLoader(YEARLY_CSV_PATH, val_days_per_month=VAL_DAYS_PER_MONTH)
    train_cases, val_cases, test_cases = loader.load()
    loader.summary()
    print(f"娴嬭瘯鏃ユ暟閲? {len(test_cases)}")
    if not test_cases:
        raise RuntimeError("Test set is empty. Please check the split configuration.")

    print("[startup] 姝ｅ湪鍔犺浇 EV 骞村害鏁版嵁...")
    ev_provider = YearlyEVProvider(YEARLY_EV_PATH)

    print(f"[startup] 姝ｅ湪鍔犺浇妯″瀷: {MODEL_PATH}")
    model = TD3.load(str(MODEL_PATH), device=resolved_device)

    cfg = IESConfig(episode_length=24, dt=1.0, reward_scale=REWARD_SCALE)
    probe_case = test_cases[0]
    probe_ev_data = ev_provider(probe_case)
    probe_env = ParkIESEnv(cfg=cfg, ts_data=probe_case.ts_data, ev_data=probe_ev_data)
    print("[startup] 鏈娴嬭瘯鐜閰嶇疆:")
    print(f"  reward_scale = {cfg.reward_scale}")
    print(f"  obs_dim = {probe_env.obs_dim}")
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

        total_system_cost = float(sum(item.get("system_cost", 0.0) for item in infos))
        total_cost_grid = float(sum(item.get("cost_grid", 0.0) for item in infos))
        total_cost_gas = float(sum(item.get("cost_gas", 0.0) for item in infos))
        total_cost_deg = float(sum(item.get("cost_deg", 0.0) for item in infos))
        total_cost_om = float(sum(item.get("cost_om", 0.0) for item in infos))
        total_penalties = float(sum(item.get("penalty_cost", 0.0) for item in infos))
        total_guide_reward = float(sum(item.get("guide_reward", 0.0) for item in infos))
        total_reward_raw = float(sum(item.get("reward_raw", 0.0) for item in infos))
        total_unmet_e = float(sum(item.get("unmet_e", 0.0) for item in infos))
        total_unmet_h = float(sum(item.get("unmet_h", 0.0) for item in infos))
        total_unmet_c = float(sum(item.get("unmet_c", 0.0) for item in infos))
        total_surplus_e = float(sum(item.get("surplus_e", 0.0) for item in infos))
        total_surplus_h = float(sum(item.get("surplus_h", 0.0) for item in infos))
        total_surplus_c = float(sum(item.get("surplus_c", 0.0) for item in infos))
        total_depart_energy_shortage_kwh = float(sum(item.get("depart_energy_shortage_kwh", 0.0) for item in infos))
        total_depart_shortage_soft_kwh = float(sum(item.get("depart_shortage_soft_kwh", 0.0) for item in infos))
        total_depart_shortage_mid_kwh = float(sum(item.get("depart_shortage_mid_kwh", 0.0) for item in infos))
        total_depart_shortage_hard_kwh = float(sum(item.get("depart_shortage_hard_kwh", 0.0) for item in infos))
        total_depart_risk_energy_kwh = float(sum(item.get("depart_risk_energy_kwh", 0.0) for item in infos))
        total_grid_overflow = float(sum(item.get("grid_overflow", 0.0) for item in infos))
        total_penalty_unserved_e = float(sum(item.get("penalty_unserved_e", 0.0) for item in infos))
        total_penalty_unserved_h = float(sum(item.get("penalty_unserved_h", 0.0) for item in infos))
        total_penalty_unserved_c = float(sum(item.get("penalty_unserved_c", 0.0) for item in infos))
        total_penalty_depart_energy = float(sum(item.get("penalty_depart_energy", 0.0) for item in infos))
        total_penalty_depart_energy_soft = float(sum(item.get("penalty_depart_energy_soft", 0.0) for item in infos))
        total_penalty_depart_energy_mid = float(sum(item.get("penalty_depart_energy_mid", 0.0) for item in infos))
        total_penalty_depart_energy_hard = float(sum(item.get("penalty_depart_energy_hard", 0.0) for item in infos))
        total_penalty_depart_risk = float(sum(item.get("penalty_depart_risk", 0.0) for item in infos))
        total_penalty_surplus_e = float(sum(item.get("penalty_surplus_e", 0.0) for item in infos))
        total_penalty_export_e = float(sum(item.get("penalty_export_e", 0.0) for item in infos))
        total_penalty_ev_export_guard = float(sum(item.get("penalty_ev_export_guard", 0.0) for item in infos))
        total_penalty_terminal_ees_soc = 0.0
        total_penalty_surplus_h = float(sum(item.get("penalty_surplus_h", 0.0) for item in infos))
        total_penalty_surplus_c = float(sum(item.get("penalty_surplus_c", 0.0) for item in infos))
        total_reward_storage_discharge_bonus = float(sum(item.get("reward_storage_discharge_bonus", 0.0) for item in infos))
        total_reward_storage_charge_bonus = float(sum(item.get("reward_storage_charge_bonus", 0.0) for item in infos))
        total_reward_ev_target_timing_bonus = float(sum(item.get("reward_ev_target_timing_bonus", 0.0) for item in infos))
        total_storage_peak_shaved_kwh = float(sum(item.get("storage_peak_shaved_kwh", 0.0) for item in infos))
        total_storage_charge_rewarded_kwh = float(sum(item.get("storage_charge_rewarded_kwh", 0.0) for item in infos))
        total_ees_charge_rewarded_kwh = float(sum(item.get("ees_charge_rewarded_kwh", 0.0) for item in infos))
        total_ev_flex_target_charge_kwh = float(sum(item.get("ev_flex_target_charge_kwh", 0.0) for item in infos))
        total_ev_buffer_charge_kwh = float(sum(item.get("ev_buffer_charge_kwh", 0.0) for item in infos))
        total_low_value_charge_kwh = float(sum(item.get("low_value_charge_kwh", 0.0) for item in infos))
        total_gt_export_clip = float(sum(item.get("p_gt_export_clip", 0.0) for item in infos))
        total_gt_export_clip_steps = int(sum(1 for item in infos if float(item.get("p_gt_export_clip", 0.0)) > ALERT_TOL))
        total_gt_safe_infeasible_steps = int(sum(1 for item in infos if not bool(item.get("gt_safe_feasible", True))))

        total_grid_buy = float(sum(item.get("p_grid_buy", 0.0) for item in infos) * cfg.dt)
        total_grid_sell = float(sum(item.get("p_grid_sell", 0.0) for item in infos) * cfg.dt)

        avg_p_gt = float(sum(item.get("p_gt", 0.0) for item in infos) / max(len(infos), 1))
        avg_p_ev_ch = float(sum(item.get("p_ev_ch", 0.0) for item in infos) / max(len(infos), 1))
        avg_p_ev_dis = float(sum(item.get("p_ev_dis", 0.0) for item in infos) / max(len(infos), 1))
        avg_p_ees_ch = float(sum(item.get("p_ees_ch", 0.0) for item in infos) / max(len(infos), 1))
        avg_p_ees_dis = float(sum(item.get("p_ees_dis", 0.0) for item in infos) / max(len(infos), 1))

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
        total_penalty_terminal_ees_soc = float(final_info.get("episode_penalty_terminal_ees_soc", 0.0))
        ees_terminal_soc_feasible = bool(final_info.get("ees_terminal_soc_feasible", True))
        final_gt_power = float(infos[-1].get("p_gt", 0.0)) if infos else 0.0
        if not ees_terminal_soc_feasible:
            print(
                "WARNING: EES terminal SOC violation: "
                f"case_index={case_idx}, final_ees_soc={final_ees_soc:.6f}, "
                f"required={terminal_ees_required_soc:.6f}, "
                f"shortage_kwh={terminal_ees_shortage_kwh:.6f}",
                file=sys.stderr,
            )

        summary_rows.append(
            {
                "case_index": case_idx,
                "month": case.month,
                "day_of_year": case.day_of_year,
                "season": case.season,
                "split": case.set_name,
                "total_reward": total_reward,
                "total_reward_raw": total_reward_raw,
                "total_system_cost": total_system_cost,
                "total_penalties": total_penalties,
                "total_guide_reward": total_guide_reward,
                "total_cost_grid": total_cost_grid,
                "total_cost_gas": total_cost_gas,
                "total_cost_deg": total_cost_deg,
                "total_cost_om": total_cost_om,
                "total_unmet_e": total_unmet_e,
                "total_unmet_h": total_unmet_h,
                "total_unmet_c": total_unmet_c,
                "total_surplus_e": total_surplus_e,
                "total_surplus_h": total_surplus_h,
                "total_surplus_c": total_surplus_c,
                "total_grid_overflow": total_grid_overflow,
                "total_penalty_unserved_e": total_penalty_unserved_e,
                "total_penalty_unserved_h": total_penalty_unserved_h,
                "total_penalty_unserved_c": total_penalty_unserved_c,
                "total_penalty_depart_energy": total_penalty_depart_energy,
                "total_penalty_depart_energy_soft": total_penalty_depart_energy_soft,
                "total_penalty_depart_energy_mid": total_penalty_depart_energy_mid,
                "total_penalty_depart_energy_hard": total_penalty_depart_energy_hard,
                "total_penalty_depart_risk": total_penalty_depart_risk,
                "total_penalty_surplus_e": total_penalty_surplus_e,
                "total_penalty_surplus_h": total_penalty_surplus_h,
                "total_penalty_surplus_c": total_penalty_surplus_c,
                "total_penalty_export_e": total_penalty_export_e,
                "total_penalty_ev_export_guard": total_penalty_ev_export_guard,
                "total_penalty_terminal_ees_soc": total_penalty_terminal_ees_soc,
                "total_depart_energy_shortage_kwh": total_depart_energy_shortage_kwh,
                "total_depart_shortage_soft_kwh": total_depart_shortage_soft_kwh,
                "total_depart_shortage_mid_kwh": total_depart_shortage_mid_kwh,
                "total_depart_shortage_hard_kwh": total_depart_shortage_hard_kwh,
                "total_depart_risk_energy_kwh": total_depart_risk_energy_kwh,
                "total_reward_storage_discharge_bonus": total_reward_storage_discharge_bonus,
                "total_reward_storage_charge_bonus": total_reward_storage_charge_bonus,
                "total_reward_ev_target_timing_bonus": total_reward_ev_target_timing_bonus,
                "total_storage_peak_shaved_kwh": total_storage_peak_shaved_kwh,
                "total_storage_charge_rewarded_kwh": total_storage_charge_rewarded_kwh,
                "total_ees_charge_rewarded_kwh": total_ees_charge_rewarded_kwh,
                "total_ev_flex_target_charge_kwh": total_ev_flex_target_charge_kwh,
                "total_ev_buffer_charge_kwh": total_ev_buffer_charge_kwh,
                "total_low_value_charge_kwh": total_low_value_charge_kwh,
                "total_gt_export_clip": total_gt_export_clip,
                "total_gt_export_clip_steps": total_gt_export_clip_steps,
                "total_gt_safe_infeasible_steps": total_gt_safe_infeasible_steps,
                "total_grid_buy_kwh": total_grid_buy,
                "total_grid_sell_kwh": total_grid_sell,
                "avg_p_gt_kw": avg_p_gt,
                "avg_p_ev_ch_kw": avg_p_ev_ch,
                "avg_p_ev_dis_kw": avg_p_ev_dis,
                "avg_p_ees_ch_kw": avg_p_ees_ch,
                "avg_p_ees_dis_kw": avg_p_ees_dis,
                "ees_soc_episode_init": float(infos[0].get("ees_soc_episode_init", 0.0)) if infos else 0.0,
                "final_ees_soc": final_ees_soc,
                "ees_soc_init": ees_soc_init,
                "terminal_ees_required_soc": terminal_ees_required_soc,
                "terminal_ees_shortage_kwh": terminal_ees_shortage_kwh,
                "ees_terminal_soc_feasible": ees_terminal_soc_feasible,
                "final_gt_power": final_gt_power,
            }
        )

        for step_info in infos:
            step_rows.append(build_step_row(case_idx, case, step_info))

        env.close()

        print(
            f"[{case_idx + 1:02d}/{len(test_cases)}] "
            f"Month={case.month}, DayOfYear={case.day_of_year}, "
            f"total_system_cost={total_system_cost:.4f}, "
            f"total_penalties={total_penalties:.4f}, "
            f"total_cost_om={total_cost_om:.4f}, "
            f"gt_export_clip={total_gt_export_clip:.4f}, "
            f"surplus_e={total_surplus_e:.4f}, "
            f"surplus_h={total_surplus_h:.4f}, "
            f"surplus_c={total_surplus_c:.4f}, "
            f"unmet_c={total_unmet_c:.4f}, "
            f"total_reward={total_reward:.4f}"
        )

    alert_rows = build_alert_rows(step_rows)
    test_duration_seconds = max(perf_counter() - test_started_perf, 0.0)

    save_csv(summary_rows, SUMMARY_CSV, DAILY_SUMMARY_COLUMNS)
    save_csv(step_rows, TIMESERIES_CSV, STEP_DETAIL_COLUMNS)
    runtime_summary = write_runtime_summary(
        RUNTIME_SUMMARY_JSON,
        method="TD3",
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
    export_test_workbook(
        pd,
        output_path=TEST_EXPORT_XLSX,
        excel_engine=excel_engine,
        step_rows=step_rows,
        summary_rows=summary_rows,
        config_row=config_row,
        alert_rows=alert_rows,
    )

    if summary_rows:
        grand_total_cost = float(sum(item["total_system_cost"] for item in summary_rows))
        grand_total_reward = float(sum(item["total_reward"] for item in summary_rows))
        grand_total_reward_raw = float(sum(item["total_reward_raw"] for item in summary_rows))
        grand_total_cost_om = float(sum(item.get("total_cost_om", 0.0) for item in summary_rows))
        grand_total_penalties = float(sum(item["total_penalties"] for item in summary_rows))
        grand_total_guide_reward = float(sum(item["total_guide_reward"] for item in summary_rows))
        grand_total_unmet_e = float(sum(item["total_unmet_e"] for item in summary_rows))
        grand_total_unmet_h = float(sum(item["total_unmet_h"] for item in summary_rows))
        grand_total_unmet_c = float(sum(item["total_unmet_c"] for item in summary_rows))
        grand_total_surplus_e = float(sum(item["total_surplus_e"] for item in summary_rows))
        grand_total_surplus_h = float(sum(item["total_surplus_h"] for item in summary_rows))
        grand_total_surplus_c = float(sum(item["total_surplus_c"] for item in summary_rows))
        grand_total_penalty_unserved_e = float(sum(item["total_penalty_unserved_e"] for item in summary_rows))
        grand_total_penalty_unserved_h = float(sum(item["total_penalty_unserved_h"] for item in summary_rows))
        grand_total_penalty_unserved_c = float(sum(item["total_penalty_unserved_c"] for item in summary_rows))
        grand_total_penalty_depart_energy = float(sum(item["total_penalty_depart_energy"] for item in summary_rows))
        grand_total_penalty_depart_energy_soft = float(sum(item["total_penalty_depart_energy_soft"] for item in summary_rows))
        grand_total_penalty_depart_energy_mid = float(sum(item["total_penalty_depart_energy_mid"] for item in summary_rows))
        grand_total_penalty_depart_energy_hard = float(sum(item["total_penalty_depart_energy_hard"] for item in summary_rows))
        grand_total_penalty_depart_risk = float(sum(item["total_penalty_depart_risk"] for item in summary_rows))
        grand_total_penalty_surplus_e = float(sum(item["total_penalty_surplus_e"] for item in summary_rows))
        grand_total_penalty_export_e = float(sum(item["total_penalty_export_e"] for item in summary_rows))
        grand_total_penalty_ev_export_guard = float(sum(item["total_penalty_ev_export_guard"] for item in summary_rows))
        grand_total_penalty_terminal_ees_soc = float(sum(item["total_penalty_terminal_ees_soc"] for item in summary_rows))
        grand_total_penalty_surplus_h = float(sum(item["total_penalty_surplus_h"] for item in summary_rows))
        grand_total_penalty_surplus_c = float(sum(item["total_penalty_surplus_c"] for item in summary_rows))
        grand_total_terminal_ees_shortage_kwh = float(sum(item["terminal_ees_shortage_kwh"] for item in summary_rows))
        grand_total_depart_risk_energy_kwh = float(sum(item["total_depart_risk_energy_kwh"] for item in summary_rows))
        grand_total_reward_storage_discharge_bonus = float(sum(item["total_reward_storage_discharge_bonus"] for item in summary_rows))
        grand_total_reward_storage_charge_bonus = float(sum(item["total_reward_storage_charge_bonus"] for item in summary_rows))
        grand_total_reward_ev_target_timing_bonus = float(sum(item["total_reward_ev_target_timing_bonus"] for item in summary_rows))
        grand_total_storage_peak_shaved_kwh = float(sum(item["total_storage_peak_shaved_kwh"] for item in summary_rows))
        grand_total_storage_charge_rewarded_kwh = float(sum(item["total_storage_charge_rewarded_kwh"] for item in summary_rows))
        grand_total_ees_charge_rewarded_kwh = float(sum(item["total_ees_charge_rewarded_kwh"] for item in summary_rows))
        grand_total_ev_flex_target_charge_kwh = float(sum(item["total_ev_flex_target_charge_kwh"] for item in summary_rows))
        grand_total_ev_buffer_charge_kwh = float(sum(item["total_ev_buffer_charge_kwh"] for item in summary_rows))
        grand_total_low_value_charge_kwh = float(sum(item["total_low_value_charge_kwh"] for item in summary_rows))
        grand_total_gt_export_clip = float(sum(item["total_gt_export_clip"] for item in summary_rows))
        grand_total_gt_export_clip_steps = int(sum(item["total_gt_export_clip_steps"] for item in summary_rows))
        grand_total_gt_safe_infeasible_steps = int(sum(item["total_gt_safe_infeasible_steps"] for item in summary_rows))

        print("\n========== TD3 Test Set Summary ==========")
        print(f"Test days: {len(summary_rows)}")
        print(f"Total system cost: {grand_total_cost:.4f}")
        print(f"Total penalties: {grand_total_penalties:.4f}")
        print(f"Total guide reward: {grand_total_guide_reward:.4f}")
        print(f"Total raw reward: {grand_total_reward_raw:.4f}")
        print(f"Total O&M cost: {grand_total_cost_om:.4f}")
        print(f"Total unmet electricity: {grand_total_unmet_e:.4f}")
        print(f"Total unmet heat: {grand_total_unmet_h:.4f}")
        print(f"Total unmet cooling: {grand_total_unmet_c:.4f}")
        print(f"Total surplus electricity: {grand_total_surplus_e:.4f}")
        print(f"Total surplus heat: {grand_total_surplus_h:.4f}")
        print(f"Total surplus cooling: {grand_total_surplus_c:.4f}")
        print(f"Penalty unmet electricity: {grand_total_penalty_unserved_e:.4f}")
        print(f"Penalty unmet heat: {grand_total_penalty_unserved_h:.4f}")
        print(f"Penalty unmet cooling: {grand_total_penalty_unserved_c:.4f}")
        print(f"Penalty depart energy: {grand_total_penalty_depart_energy:.4f}")
        print(f"Penalty depart energy (soft band): {grand_total_penalty_depart_energy_soft:.4f}")
        print(f"Penalty depart energy (mid band): {grand_total_penalty_depart_energy_mid:.4f}")
        print(f"Penalty depart energy (hard band): {grand_total_penalty_depart_energy_hard:.4f}")
        print(f"Penalty depart risk: {grand_total_penalty_depart_risk:.4f}")
        print(f"Penalty surplus electricity: {grand_total_penalty_surplus_e:.4f}")
        print(f"Penalty export electricity: {grand_total_penalty_export_e:.4f}")
        print(f"Penalty EV export guard: {grand_total_penalty_ev_export_guard:.4f}")
        print(f"Penalty EES terminal SOC: {grand_total_penalty_terminal_ees_soc:.4f}")
        print(f"Penalty surplus heat: {grand_total_penalty_surplus_h:.4f}")
        print(f"Penalty surplus cooling: {grand_total_penalty_surplus_c:.4f}")
        print(f"EES terminal shortage (kWh): {grand_total_terminal_ees_shortage_kwh:.4f}")
        print(f"EV departure risk energy (kWh): {grand_total_depart_risk_energy_kwh:.4f}")
        print(f"Storage discharge reward bonus: {grand_total_reward_storage_discharge_bonus:.4f}")
        print(f"Storage charge reward bonus: {grand_total_reward_storage_charge_bonus:.4f}")
        print(f"EV target timing reward bonus: {grand_total_reward_ev_target_timing_bonus:.4f}")
        print(f"Storage peak shaved energy (kWh): {grand_total_storage_peak_shaved_kwh:.4f}")
        print(f"Storage charge rewarded energy (kWh): {grand_total_storage_charge_rewarded_kwh:.4f}")
        print(f"EES charge rewarded energy (kWh): {grand_total_ees_charge_rewarded_kwh:.4f}")
        print(f"EV flex target charge energy (kWh): {grand_total_ev_flex_target_charge_kwh:.4f}")
        print(f"EV buffer charge energy (kWh): {grand_total_ev_buffer_charge_kwh:.4f}")
        print(f"Low value charge energy (kWh): {grand_total_low_value_charge_kwh:.4f}")
        print(f"GT export clip total: {grand_total_gt_export_clip:.4f}")
        print(f"GT export clip steps: {grand_total_gt_export_clip_steps}")
        print(f"GT safety infeasible steps: {grand_total_gt_safe_infeasible_steps}")
        print(f"Total reward: {grand_total_reward:.4f}")
    else:
        print("No test results were generated.")

    print(f"Test summary CSV saved to: {SUMMARY_CSV}")
    print(f"Timeseries detail CSV saved to: {TIMESERIES_CSV}")
    print(f"Runtime summary JSON saved to: {RUNTIME_SUMMARY_JSON}")
    print(
        "Test runtime: "
        f"{runtime_summary['test_duration_seconds']:.6f} s, "
        f"{runtime_summary['time_per_day_seconds']:.6f} s/day, "
        f"{runtime_summary['time_per_step_seconds']:.6f} s/step"
    )
    print(f"Test export Excel saved to: {TEST_EXPORT_XLSX}")
    print(f"Alert rows exported: {len(alert_rows)}")


if __name__ == "__main__":
    main()
