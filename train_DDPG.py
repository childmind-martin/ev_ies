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


YEARLY_CSV_PATH = Path(r"D:\wangye\chengxu\3.30\yearly_data_sci.csv")
YEARLY_EV_PATH = Path(r"D:\wangye\chengxu\3.30\ev_data_s123_bilevel.npy")

MODEL_DIR = Path("./models/ddpg_yearly_single")
BEST_MODEL_DIR = MODEL_DIR / "best"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
LOG_DIR = Path("./logs/ddpg_yearly_single")
TB_DIR = Path("./tb/ddpg_yearly_single")
RESULT_DIR = Path("./results/ddpg_yearly_training")
TRAINING_EXPORT_XLSX = RESULT_DIR / "ddpg_training_export.xlsx"
TRAINING_STEP_DETAIL_CSV = RESULT_DIR / "ddpg_training_step_detail.csv"
TRAINING_EPISODE_SUMMARY_CSV = RESULT_DIR / "episode_summary.csv"
TRAINING_RUNTIME_SUMMARY_JSON = RESULT_DIR / "training_runtime_summary.json"

SEED = 42
TOTAL_EPISODES = 4000
TOTAL_TIMESTEPS = TOTAL_EPISODES * 24
BUFFER_SIZE = 100000
LEARNING_STARTS = 1_000
BATCH_SIZE = 256
LEARNING_RATE = 1.5e-4
TAU = 0.005
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
EVAL_FREQ = 5000
SAVE_FREQ = 5000
REWARD_SCALE = 1e-5
VAL_DAYS_PER_MONTH = 2

POLICY_NAME = "MlpPolicy"
POLICY_NET_ARCH_PI = [256, 256]
POLICY_NET_ARCH_QF = [256, 256]
ACTIVATION_FN_NAME = "ReLU"
REQUESTED_DEVICE = "auto"

EPISODE_INFO_KEYS = (
    "episode_system_cost",
    "episode_cost_grid",
    "episode_cost_gas",
    "episode_cost_deg",
    "episode_cost_om",
    "episode_penalty_cost",
    "episode_guide_reward",
    "episode_reward_raw",
    "episode_unserved_e",
    "episode_unserved_h",
    "episode_unserved_c",
    "episode_surplus_e",
    "episode_export_e",
    "episode_surplus_h",
    "episode_surplus_c",
    "episode_penalty_unserved_e",
    "episode_penalty_unserved_h",
    "episode_penalty_unserved_c",
    "episode_penalty_surplus_e",
    "episode_penalty_surplus_h",
    "episode_penalty_surplus_c",
    "episode_penalty_export_e",
    "episode_penalty_depart_energy",
    "episode_penalty_depart_risk",
    "episode_penalty_ev_export_guard",
    "episode_terminal_ees_shortage_kwh",
    "episode_penalty_terminal_ees_soc",
    "episode_reward_storage_discharge",
    "episode_reward_storage_charge",
    "episode_reward_ev_target_timing",
    "episode_storage_peak_shaved_kwh",
    "episode_storage_charge_rewarded_kwh",
    "episode_ees_charge_rewarded_kwh",
    "episode_ev_flex_target_charge_kwh",
    "episode_ev_buffer_charge_kwh",
    "episode_low_value_charge_kwh",
    "episode_gt_export_clip",
    "episode_gt_export_clip_steps",
    "episode_gt_safe_infeasible_steps",
)

EPISODE_SCALAR_TAGS = {
    "episode_system_cost": "custom/episode_system_cost",
    "episode_cost_grid": "custom/episode_cost_grid",
    "episode_cost_gas": "custom/episode_cost_gas",
    "episode_cost_deg": "custom/episode_cost_deg",
    "episode_cost_om": "custom/episode_cost_om",
    "episode_penalty_cost": "custom/episode_penalty_cost",
    "episode_guide_reward": "custom/episode_guide_reward",
    "episode_reward_raw": "custom/episode_reward_raw",
    "episode_unserved_e": "custom/episode_unserved_e",
    "episode_unserved_h": "custom/episode_unserved_h",
    "episode_unserved_c": "custom/episode_unserved_c",
    "episode_surplus_e": "custom/episode_surplus_e",
    "episode_export_e": "custom/episode_export_e",
    "episode_surplus_h": "custom/episode_surplus_h",
    "episode_surplus_c": "custom/episode_surplus_c",
    "episode_penalty_unserved_e": "custom/episode_penalty_unserved_e",
    "episode_penalty_unserved_h": "custom/episode_penalty_unserved_h",
    "episode_penalty_unserved_c": "custom/episode_penalty_unserved_c",
    "episode_penalty_surplus_e": "custom/episode_penalty_surplus_e",
    "episode_penalty_surplus_h": "custom/episode_penalty_surplus_h",
    "episode_penalty_surplus_c": "custom/episode_penalty_surplus_c",
    "episode_penalty_export_e": "custom/episode_penalty_export_e",
    "episode_penalty_depart_energy": "custom/episode_penalty_depart_energy",
    "episode_penalty_depart_risk": "custom/episode_penalty_depart_risk",
    "episode_penalty_ev_export_guard": "custom/episode_penalty_ev_export_guard",
    "episode_terminal_ees_shortage_kwh": "custom/episode_terminal_ees_shortage_kwh",
    "episode_penalty_terminal_ees_soc": "custom/episode_penalty_terminal_ees_soc",
    "episode_reward_storage_discharge": "custom/episode_reward_storage_discharge",
    "episode_reward_storage_charge": "custom/episode_reward_storage_charge",
    "episode_reward_ev_target_timing": "custom/episode_reward_ev_target_timing",
    "episode_storage_peak_shaved_kwh": "custom/episode_storage_peak_shaved_kwh",
    "episode_storage_charge_rewarded_kwh": "custom/episode_storage_charge_rewarded_kwh",
    "episode_ees_charge_rewarded_kwh": "custom/episode_ees_charge_rewarded_kwh",
    "episode_ev_flex_target_charge_kwh": "custom/episode_ev_flex_target_charge_kwh",
    "episode_ev_buffer_charge_kwh": "custom/episode_ev_buffer_charge_kwh",
    "episode_low_value_charge_kwh": "custom/episode_low_value_charge_kwh",
    "episode_gt_export_clip": "custom/episode_gt_export_clip",
    "episode_gt_export_clip_steps": "custom/episode_gt_export_clip_steps",
    "episode_gt_safe_infeasible_steps": "custom/episode_gt_safe_infeasible_steps",
}

STEP_DETAIL_COLUMNS = (
    "global_step", "episode_idx", "time_step", "month", "day_of_year", "season", "split", "terminated",
    "a_ees", "a_gt", "a_ev_ch", "a_ev_dis",
    "elec_load", "heat_load", "cool_load", "pv", "wt", "grid_buy_price", "grid_sell_price", "gas_price",
    "p_grid", "p_grid_buy", "p_grid_sell", "p_gt", "p_ev_ch", "p_ev_dis", "p_ees_ch", "p_ees_dis", "p_ec_elec_in",
    "p_ev_rigid_ch", "p_ev_flex_target_ch", "p_ev_buffer_ch",
    "p_whb_heat", "p_gb_heat", "p_ac_cool", "p_ec_cool", "unmet_h", "surplus_h", "unmet_c", "surplus_c",
    "ees_soc", "ees_soc_episode_init", "ev_soc_mean",
    "cost_grid", "cost_gas", "cost_deg", "cost_om", "system_cost",
    "penalty_cost", "penalty_unserved_e", "penalty_unserved_h", "penalty_unserved_c", "penalty_depart_energy",
    "penalty_depart_energy_soft", "penalty_depart_energy_mid", "penalty_depart_energy_hard", "penalty_depart_risk",
    "penalty_surplus_e", "penalty_surplus_h", "penalty_surplus_c", "penalty_export_e", "penalty_ev_export_guard",
    "unmet_e", "surplus_e", "depart_energy_shortage_kwh", "depart_shortage_soft_kwh", "depart_shortage_mid_kwh", "depart_shortage_hard_kwh",
    "depart_risk_energy_kwh", "depart_risk_vehicle_count", "ev_export_overlap_kwh", "grid_overflow",
    "guide_reward", "reward_storage_discharge_bonus", "reward_storage_charge_bonus", "reward_ev_target_timing_bonus",
    "storage_peak_shaved_kwh", "storage_charge_rewarded_kwh", "ees_charge_rewarded_kwh",
    "ev_flex_target_charge_kwh", "ev_buffer_charge_kwh", "low_value_charge_kwh",
    "ev_peak_pressure_without_storage_kw", "low_value_energy_before_storage_prepare_kwh",
    "ees_discharge_reward_weight", "ees_charge_reward_weight",
    "p_gt_safe_min", "p_gt_safe_max", "p_gt_export_clip", "gt_safe_feasible", "reward_raw", "reward_scaled",
)

EPISODE_SUMMARY_COLUMNS = (
    "episode_idx", "global_step_start", "global_step_end", "month", "day_of_year", "season", "split",
    "episode_reward_scaled", "episode_reward_raw", "episode_system_cost", "episode_cost_grid", "episode_cost_gas",
    "episode_cost_deg", "episode_cost_om", "episode_penalty_cost", "episode_guide_reward",
    "episode_unserved_e", "episode_unserved_h", "episode_unserved_c", "episode_surplus_e", "episode_export_e",
    "episode_surplus_h", "episode_surplus_c", "episode_penalty_unserved_e", "episode_penalty_unserved_h", "episode_penalty_unserved_c",
    "episode_penalty_surplus_e", "episode_penalty_surplus_h", "episode_penalty_surplus_c", "episode_penalty_export_e",
    "episode_penalty_depart_energy", "episode_penalty_depart_risk", "episode_penalty_ev_export_guard",
    "episode_terminal_ees_shortage_kwh", "episode_penalty_terminal_ees_soc",
    "terminal_ees_required_soc", "ees_terminal_soc_feasible",
    "episode_reward_storage_discharge", "episode_reward_storage_charge", "episode_reward_ev_target_timing",
    "episode_storage_peak_shaved_kwh", "episode_storage_charge_rewarded_kwh", "episode_ees_charge_rewarded_kwh",
    "episode_ev_flex_target_charge_kwh", "episode_ev_buffer_charge_kwh", "episode_low_value_charge_kwh",
    "episode_gt_export_clip", "episode_gt_export_clip_steps", "episode_gt_safe_infeasible_steps",
    "ees_soc_episode_init", "final_ees_soc", "final_gt_power",
)

CONFIG_COLUMNS = (
    "run_id", "timestamp_start", "yearly_csv_path", "yearly_ev_path", "n_train_cases", "n_val_cases", "n_test_cases",
    "training_duration_seconds", "training_duration_hms",
    "TOTAL_EPISODES", "TOTAL_TIMESTEPS", "SEED", "BUFFER_SIZE", "LEARNING_STARTS", "BATCH_SIZE", "LEARNING_RATE", "TAU",
    "TRAIN_FREQ", "GRADIENT_STEPS", "EVAL_FREQ", "SAVE_FREQ", "VAL_DAYS_PER_MONTH", "REWARD_SCALE",
    "policy", "net_arch_pi", "net_arch_qf", "activation_fn", "device",
    "episode_length", "dt", "future_horizon", "exogenous_future_horizon", "grid_import_max", "grid_export_max",
    "gt_p_max", "gt_eta_e", "gt_ramp", "gb_h_max", "gb_ramp", "ac_c_max", "ec_c_max", "ees_p_max", "ees_e_cap",
    "ees_soc_init", "ees_soc_min", "ees_soc_max",
    "penalty_unserved_e", "penalty_unserved_h", "penalty_unserved_c", "penalty_depart_soc", "penalty_depart_energy_soft",
    "penalty_depart_energy_mid", "penalty_ev_depart_risk", "penalty_surplus_e", "penalty_surplus_h", "penalty_surplus_c",
    "penalty_export_e", "penalty_ev_export_guard", "penalty_ees_terminal_soc",
    "ees_terminal_soc_tolerance",
    "ev_discharge_price_threshold", "ev_charge_price_threshold", "ev_peak_export_tolerance_kw",
    "reward_storage_discharge_base", "reward_storage_charge_base", "reward_ev_target_timing_base",
    "ees_reward_discharge_soc_floor", "ees_reward_charge_soc_target", "reward_scale",
)

STEP_INFO_COLUMNS = tuple(column for column in STEP_DETAIL_COLUMNS if column not in {"global_step", "episode_idx"})
EPISODE_DERIVED_COLUMNS = {
    "episode_idx",
    "global_step_start",
    "global_step_end",
    "episode_reward_scaled",
    "final_ees_soc",
    "terminal_ees_required_soc",
    "ees_terminal_soc_feasible",
    "final_gt_power",
}


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
        raise RuntimeError("Current interpreter is Python 3.14+ or a non-final release. This project should be run with Python 3.9-3.11.")
    missing_paths = [str(path) for path in (YEARLY_CSV_PATH, YEARLY_EV_PATH) if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"缺少输入数据文件: {', '.join(missing_paths)}")
    excel_engine = resolve_excel_engine()
    required_modules = ["numpy", "pandas", "gymnasium", "torch", "stable_baselines3"]
    missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing_modules:
        raise ModuleNotFoundError("Missing required packages: " f"{', '.join(missing_modules)}. " "Please install: pip install numpy pandas gymnasium torch stable-baselines3 tensorboard openpyxl")
    return excel_engine


def infer_unit(column_name: str) -> str:
    if column_name in {"global_step", "global_step_start", "global_step_end", "TOTAL_TIMESTEPS", "EVAL_FREQ", "SAVE_FREQ", "LEARNING_STARTS"}:
        return "step"
    if column_name == "TOTAL_EPISODES":
        return "episode"
    if column_name in {"episode_idx", "time_step", "SEED", "BUFFER_SIZE", "BATCH_SIZE", "TRAIN_FREQ", "GRADIENT_STEPS", "VAL_DAYS_PER_MONTH", "episode_length", "n_train_cases", "n_val_cases", "n_test_cases"} or column_name.endswith("_count") or column_name.endswith("_steps"):
        return "count"
    if column_name in {"month", "day_of_year"}:
        return "index"
    if column_name in {"season", "split", "policy", "activation_fn", "device", "run_id"}:
        return "label"
    if column_name == "timestamp_start":
        return "datetime"
    if column_name == "training_duration_seconds":
        return "s"
    if column_name == "training_duration_hms":
        return "HH:MM:SS"
    if column_name.endswith("_path"):
        return "path"
    if "price" in column_name:
        return "price/kWh"
    if column_name.startswith("p_") or column_name.endswith("_load") or column_name in {"pv", "wt", "unmet_e", "unmet_h", "unmet_c", "surplus_e", "surplus_h", "surplus_c", "grid_overflow", "gt_import_pressure_at_thermal", "gt_safe_band_width", "final_gt_power"}:
        return "kW"
    if column_name.endswith("_kwh") or column_name == "ees_e_cap":
        return "kWh"
    if "soc" in column_name:
        return "p.u."
    if column_name.startswith("cost_") or column_name.startswith("penalty_") or column_name == "system_cost":
        return "cost"
    if column_name.startswith("reward_") or column_name in {"guide_reward", "episode_reward_scaled", "episode_reward_raw"}:
        return "reward"
    if column_name in {"terminated", "gt_low_price_active", "gt_thermal_feasible", "gt_electric_feasible", "gt_safe_feasible"}:
        return "bool"
    return ""


def describe_step_column(column_name: str) -> tuple[str, str]:
    if column_name == "global_step":
        return "Global DDPG environment step counted during training.", "callback"
    if column_name == "episode_idx":
        return "1-based index of the current training episode.", "callback"
    if column_name == "time_step":
        return "Zero-based hour index inside the current episode.", "env info"
    if column_name in {"month", "day_of_year", "season", "split"}:
        return f"Sample metadata field `{column_name}` attached to the current training step.", "env info"
    if column_name == "terminated":
        return "Whether the environment ended naturally after this step.", "env info"
    if column_name.startswith("a_"):
        return f"Executed action value `{column_name}` after action-space clipping.", "env info"
    if column_name.startswith("penalty_"):
        return f"Per-step penalty component `{column_name}`.", "env info"
    if column_name.startswith("cost_") or column_name == "system_cost":
        return f"Per-step cost metric `{column_name}`.", "env info"
    if column_name.startswith("reward_") or column_name == "guide_reward":
        return f"Per-step reward-related metric `{column_name}`.", "env info"
    if column_name.startswith("p_"):
        return f"Per-step dispatch or capacity metric `{column_name}`.", "env info"
    if "soc" in column_name:
        return f"SOC-related metric `{column_name}` recorded for the current step.", "env info"
    if column_name.endswith("_count"):
        return f"Per-step count metric `{column_name}`.", "env info"
    return f"Per-step exported metric `{column_name}`.", "env info"


def describe_episode_column(column_name: str) -> tuple[str, str]:
    if column_name == "episode_idx":
        return "1-based index of the completed training episode.", "callback"
    if column_name in {"global_step_start", "global_step_end"}:
        return f"Global DDPG step marker `{column_name}` for the completed episode.", "callback"
    if column_name in {"month", "day_of_year", "season", "split"}:
        return f"Sample metadata field `{column_name}` attached to the completed episode.", "terminal info"
    if column_name == "episode_reward_scaled":
        return "Sum of reward_scaled across the completed episode.", "callback"
    if column_name in {"final_ees_soc", "final_gt_power"}:
        return f"Terminal-state metric `{column_name}` for the completed episode.", "callback"
    if column_name.startswith("episode_"):
        return f"Episode aggregate metric `{column_name}` exported at episode termination.", "terminal info"
    return f"Episode-level exported metric `{column_name}`.", "terminal info"


def describe_config_column(column_name: str) -> tuple[str, str]:
    if column_name.isupper():
        return f"Training constant `{column_name}` recorded for experiment reproducibility.", "train_DDPG.py"
    if column_name in {"run_id", "timestamp_start", "training_duration_seconds", "training_duration_hms", "policy", "net_arch_pi", "net_arch_qf", "activation_fn", "device"}:
        return f"Run setup field `{column_name}` recorded for experiment reproducibility.", "train_DDPG.py"
    if column_name.endswith("_path"):
        return f"Input data path `{column_name}` used for this run.", "train_DDPG.py"
    if column_name.endswith("_cases"):
        return f"Dataset case count `{column_name}` observed at startup.", "YearlyCSVDataLoader"
    return f"Configuration value `{column_name}` captured for experiment reproducibility.", "IESConfig/train_DDPG.py"


def build_column_description_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for column_name in STEP_DETAIL_COLUMNS:
        meaning, source = describe_step_column(column_name)
        rows.append({"sheet": "step_detail", "column_name": column_name, "unit": infer_unit(column_name), "meaning": meaning, "source": source})
    for column_name in EPISODE_SUMMARY_COLUMNS:
        meaning, source = describe_episode_column(column_name)
        rows.append({"sheet": "episode_summary", "column_name": column_name, "unit": infer_unit(column_name), "meaning": meaning, "source": source})
    for column_name in CONFIG_COLUMNS:
        meaning, source = describe_config_column(column_name)
        rows.append({"sheet": "config", "column_name": column_name, "unit": infer_unit(column_name), "meaning": meaning, "source": source})
    return rows


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def write_training_runtime_summary(
    path: Path,
    *,
    method: str,
    total_episodes: int | None = None,
    total_timesteps: int,
    training_duration_seconds: float | None,
    device: str,
    seed: int,
) -> dict[str, Any]:
    duration = float(training_duration_seconds or 0.0)
    row = {
        "method": method,
        "total_episodes": int(total_episodes) if total_episodes is not None else None,
        "total_timesteps": int(total_timesteps),
        "training_duration_seconds": round(duration, 6),
        "training_duration_hms": format_duration(duration),
        "device": str(device),
        "seed": int(seed),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return row


def build_config_row(cfg: Any, *, run_id: str, timestamp_start: str, n_train_cases: int, n_val_cases: int, n_test_cases: int, device: str) -> dict[str, Any]:
    cfg_values = asdict(cfg)
    return {
        "run_id": run_id, "timestamp_start": timestamp_start, "yearly_csv_path": str(YEARLY_CSV_PATH), "yearly_ev_path": str(YEARLY_EV_PATH),
        "n_train_cases": int(n_train_cases), "n_val_cases": int(n_val_cases), "n_test_cases": int(n_test_cases),
        "training_duration_seconds": None, "training_duration_hms": "",
        "TOTAL_EPISODES": TOTAL_EPISODES, "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS, "SEED": SEED, "BUFFER_SIZE": BUFFER_SIZE, "LEARNING_STARTS": LEARNING_STARTS, "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE, "TAU": TAU, "TRAIN_FREQ": TRAIN_FREQ, "GRADIENT_STEPS": GRADIENT_STEPS,
        "EVAL_FREQ": EVAL_FREQ, "SAVE_FREQ": SAVE_FREQ, "VAL_DAYS_PER_MONTH": VAL_DAYS_PER_MONTH, "REWARD_SCALE": REWARD_SCALE,
        "policy": POLICY_NAME, "net_arch_pi": str(POLICY_NET_ARCH_PI), "net_arch_qf": str(POLICY_NET_ARCH_QF), "activation_fn": ACTIVATION_FN_NAME, "device": device,
        "episode_length": cfg_values["episode_length"], "dt": cfg_values["dt"], "future_horizon": cfg_values["future_horizon"], "exogenous_future_horizon": cfg_values["exogenous_future_horizon"],
        "grid_import_max": cfg_values["grid_import_max"], "grid_export_max": cfg_values["grid_export_max"], "gt_p_max": cfg_values["gt_p_max"], "gt_eta_e": cfg_values["gt_eta_e"], "gt_ramp": cfg_values["gt_ramp"],
        "gb_h_max": cfg_values["gb_h_max"], "gb_ramp": cfg_values["gb_ramp"], "ac_c_max": cfg_values["ac_c_max"], "ec_c_max": cfg_values["ec_c_max"],
        "ees_p_max": cfg_values["ees_p_max"], "ees_e_cap": cfg_values["ees_e_cap"], "ees_soc_init": cfg_values["ees_soc_init"], "ees_soc_min": cfg_values["ees_soc_min"], "ees_soc_max": cfg_values["ees_soc_max"],
        "ees_terminal_soc_tolerance": cfg_values["ees_terminal_soc_tolerance"],
        "penalty_unserved_e": cfg_values["penalty_unserved_e"], "penalty_unserved_h": cfg_values["penalty_unserved_h"], "penalty_unserved_c": cfg_values["penalty_unserved_c"],
        "penalty_depart_soc": cfg_values["penalty_depart_soc"], "penalty_depart_energy_soft": cfg_values["penalty_depart_energy_soft"], "penalty_depart_energy_mid": cfg_values["penalty_depart_energy_mid"], "penalty_ev_depart_risk": cfg_values["penalty_ev_depart_risk"],
        "penalty_surplus_e": cfg_values["penalty_surplus_e"], "penalty_surplus_h": cfg_values["penalty_surplus_h"], "penalty_surplus_c": cfg_values["penalty_surplus_c"], "penalty_export_e": cfg_values["penalty_export_e"],
        "penalty_ev_export_guard": cfg_values["penalty_ev_export_guard"], "penalty_ees_terminal_soc": cfg_values["penalty_ees_terminal_soc"], "ev_discharge_price_threshold": cfg_values["ev_discharge_price_threshold"], "ev_charge_price_threshold": cfg_values["ev_charge_price_threshold"],
        "ev_peak_export_tolerance_kw": cfg_values["ev_peak_export_tolerance_kw"], "reward_storage_discharge_base": cfg_values["reward_storage_discharge_base"], "reward_storage_charge_base": cfg_values["reward_storage_charge_base"],
        "reward_ev_target_timing_base": cfg_values["reward_ev_target_timing_base"], "ees_reward_discharge_soc_floor": cfg_values["ees_reward_discharge_soc_floor"], "ees_reward_charge_soc_target": cfg_values["ees_reward_charge_soc_target"],
        "reward_scale": cfg_values["reward_scale"],
    }


def build_training_export_callback(
    base_callback_cls,
    *,
    pd,
    output_path: Path,
    step_csv_path: Path,
    episode_csv_path: Path,
    excel_engine: str,
    config_row: dict[str, Any],
):
    column_description_rows = build_column_description_rows()

    class TrainingExportCallback(base_callback_cls):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.step_rows: list[dict[str, Any]] = []
            self.episode_rows: list[dict[str, Any]] = []
            self.current_episode_idx = 1
            self.current_episode_reward_scaled = 0.0
            self.current_episode_start_step: int | None = None
            self.training_started_perf: float | None = None
            self.training_duration_seconds: float | None = None
            self.training_duration_hms = ""

        def _on_training_start(self) -> None:
            if int(getattr(self.training_env, "num_envs", 1)) != 1:
                raise RuntimeError("TrainingExportCallback only supports a single training environment.")
            self.training_started_perf = perf_counter()

        def _record_episode_scalars(self, info: dict[str, Any]) -> None:
            if "episode_system_cost" not in info:
                return
            for key, tag in EPISODE_SCALAR_TAGS.items():
                if key in info:
                    self.logger.record(tag, float(info[key]))

        def _build_step_row(self, info: dict[str, Any]) -> dict[str, Any]:
            missing = [column for column in STEP_INFO_COLUMNS if column not in info]
            if missing:
                raise KeyError(f"Missing required step_detail columns in env info: {missing}")
            global_step = int(self.num_timesteps)
            if self.current_episode_start_step is None:
                self.current_episode_start_step = global_step
            row = {"global_step": global_step, "episode_idx": int(self.current_episode_idx)}
            for column in STEP_INFO_COLUMNS:
                row[column] = info.get(column)
            return row

        def _build_episode_row(self, info: dict[str, Any], global_step_end: int) -> dict[str, Any]:
            missing = [column for column in EPISODE_SUMMARY_COLUMNS if column not in EPISODE_DERIVED_COLUMNS and column not in info]
            if missing:
                raise KeyError(f"Missing required episode_summary columns in terminal info: {missing}")
            row = {
                "episode_idx": int(self.current_episode_idx),
                "global_step_start": int(self.current_episode_start_step or global_step_end),
                "global_step_end": int(global_step_end),
                "episode_reward_scaled": float(self.current_episode_reward_scaled),
                "final_ees_soc": float(info.get("final_ees_soc", info.get("ees_soc", 0.0))),
                "terminal_ees_required_soc": float(info.get("terminal_ees_required_soc", 0.0)),
                "ees_terminal_soc_feasible": bool(info.get("ees_terminal_soc_feasible", True)),
                "final_gt_power": float(info.get("p_gt", 0.0)),
            }
            for column in EPISODE_SUMMARY_COLUMNS:
                if column in row:
                    continue
                row[column] = info.get(column)
            return row

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            if not infos:
                return True
            if len(infos) != 1:
                raise RuntimeError("TrainingExportCallback only supports a single training environment.")
            info = dict(infos[0])
            self._record_episode_scalars(info)
            step_row = self._build_step_row(info)
            self.step_rows.append(step_row)
            self.current_episode_reward_scaled += float(info.get("reward_scaled", 0.0))
            if bool(step_row["terminated"]):
                self.logger.record("custom/episode_reward_scaled", float(self.current_episode_reward_scaled))
                self.episode_rows.append(self._build_episode_row(info, int(step_row["global_step"])))
                self.current_episode_idx += 1
                self.current_episode_reward_scaled = 0.0
                self.current_episode_start_step = None
            return True

        def _on_training_end(self) -> None:
            if self.training_started_perf is not None:
                self.training_duration_seconds = max(perf_counter() - self.training_started_perf, 0.0)
            else:
                self.training_duration_seconds = None
            self.training_duration_hms = format_duration(self.training_duration_seconds)
            config_row["training_duration_seconds"] = (
                round(self.training_duration_seconds, 3)
                if self.training_duration_seconds is not None
                else None
            )
            config_row["training_duration_hms"] = self.training_duration_hms
            output_path.parent.mkdir(parents=True, exist_ok=True)
            step_csv_path.parent.mkdir(parents=True, exist_ok=True)
            episode_csv_path.parent.mkdir(parents=True, exist_ok=True)
            step_df = pd.DataFrame(self.step_rows, columns=STEP_DETAIL_COLUMNS)
            episode_df = pd.DataFrame(self.episode_rows, columns=EPISODE_SUMMARY_COLUMNS)
            config_df = pd.DataFrame([{column: config_row.get(column) for column in CONFIG_COLUMNS}], columns=CONFIG_COLUMNS)
            description_df = pd.DataFrame(column_description_rows, columns=["sheet", "column_name", "unit", "meaning", "source"])
            step_df.to_csv(step_csv_path, index=False, encoding="utf-8-sig")
            episode_df.to_csv(episode_csv_path, index=False, encoding="utf-8-sig")
            with pd.ExcelWriter(output_path, engine=excel_engine) as writer:
                episode_df.to_excel(writer, sheet_name="episode_summary", index=False)
                config_df.to_excel(writer, sheet_name="config", index=False)
                description_df.to_excel(writer, sheet_name="column_description", index=False)
            print(f"[export] Training step_detail CSV saved to: {step_csv_path.resolve()}")
            print(f"[export] Training episode_summary CSV saved to: {episode_csv_path.resolve()}")
            print(f"[export] Training diagnostics workbook saved to: {output_path.resolve()}")

    return TrainingExportCallback()


def read_validation_reward_summary(np, eval_npz_path: Path) -> dict[str, float | None]:
    if not eval_npz_path.exists():
        return {"best_validation_reward": None, "final_validation_reward": None}
    data = np.load(eval_npz_path)
    if "results" not in data:
        return {"best_validation_reward": None, "final_validation_reward": None}
    results = np.asarray(data["results"], dtype=np.float64)
    if results.size == 0:
        return {"best_validation_reward": None, "final_validation_reward": None}
    mean_rewards = np.mean(results, axis=1)
    return {
        "best_validation_reward": float(np.max(mean_rewards)),
        "final_validation_reward": float(mean_rewards[-1]),
    }


def import_runtime_modules():
    print("[startup] Importing training dependencies...")
    import pandas as pd
    import numpy as np
    import torch as th
    from stable_baselines3 import DDPG
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise
    from park_ies_env import IESConfig, ParkIESEnv
    from yearly_case_env import YearlyEVProvider, YearlyCaseEnv
    from yearly_csv_loader import YearlyCSVDataLoader
    print("[startup] Training dependencies imported.")
    return {
        "pd": pd, "np": np, "th": th, "DDPG": DDPG, "BaseCallback": BaseCallback,
        "CheckpointCallback": CheckpointCallback, "EvalCallback": EvalCallback, "Monitor": Monitor,
        "NormalActionNoise": NormalActionNoise,
        "IESConfig": IESConfig, "ParkIESEnv": ParkIESEnv, "YearlyCSVDataLoader": YearlyCSVDataLoader,
        "YearlyEVProvider": YearlyEVProvider, "YearlyCaseEnv": YearlyCaseEnv,
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
        print("  note = CUDA unavailable, so DDPG will run on CPU.")


def main() -> None:
    configure_stdio()
    try:
        excel_engine = preflight_check()
        modules = import_runtime_modules()
    except Exception as exc:
        print(f"[startup] 启动失败: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    pd = modules["pd"]
    np = modules["np"]
    th = modules["th"]
    DDPG = modules["DDPG"]
    BaseCallback = modules["BaseCallback"]
    CheckpointCallback = modules["CheckpointCallback"]
    EvalCallback = modules["EvalCallback"]
    Monitor = modules["Monitor"]
    NormalActionNoise = modules["NormalActionNoise"]
    IESConfig = modules["IESConfig"]
    ParkIESEnv = modules["ParkIESEnv"]
    YearlyCSVDataLoader = modules["YearlyCSVDataLoader"]
    YearlyEVProvider = modules["YearlyEVProvider"]
    YearlyCaseEnv = modules["YearlyCaseEnv"]

    set_random_seed(SEED, np, th)
    resolved_device = resolve_sb3_device(th, REQUESTED_DEVICE)
    print_torch_runtime_summary(th, requested_device=REQUESTED_DEVICE, resolved_device=resolved_device)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("[startup] 正在加载全年训练数据...")
    loader = YearlyCSVDataLoader(YEARLY_CSV_PATH, val_days_per_month=VAL_DAYS_PER_MONTH)
    train_cases, val_cases, test_cases = loader.load()
    loader.summary()
    print(f"训练日数量: {len(train_cases)}")
    print(f"验证日数量: {len(val_cases)}")
    print(f"测试日数量: {len(test_cases)}")
    if not val_cases:
        raise RuntimeError("Fixed validation set is empty. Please check the split configuration.")

    n_eval_episodes = len(val_cases)
    print("[startup] 正在加载 EV 年度数据...")
    ev_provider = YearlyEVProvider(YEARLY_EV_PATH)

    cfg = IESConfig(episode_length=24, dt=1.0, reward_scale=REWARD_SCALE)
    probe_case = val_cases[0] if val_cases else train_cases[0]
    probe_ev_data = ev_provider(probe_case)
    probe_env = ParkIESEnv(cfg=cfg, ts_data=probe_case.ts_data, ev_data=probe_ev_data)
    print("[startup] 本次训练关键配置 DDPG:")
    print(f"  reward_scale = {cfg.reward_scale}")
    print(f"  total_episodes = {TOTAL_EPISODES}")
    print(f"  total_timesteps = {TOTAL_TIMESTEPS}")
    print(f"  buffer_size = {BUFFER_SIZE}")
    print(f"  learning_starts = {LEARNING_STARTS}")
    print(f"  batch_size = {BATCH_SIZE}")
    print(f"  learning_rate = {LEARNING_RATE}")
    print(f"  tau = {TAU}")
    print(f"  val_days_per_month = {VAL_DAYS_PER_MONTH}")
    print(f"  n_eval_episodes = {n_eval_episodes}")
    print(f"  obs_dim = {probe_env.obs_dim}")
    print(f"  future_horizon = {cfg.future_horizon}")
    print(f"  exogenous_future_horizon = {cfg.exogenous_future_horizon}")
    probe_env.close()

    train_env = Monitor(YearlyCaseEnv(loader=loader, split="train", ev_provider=ev_provider, cfg=cfg, shuffle_train=True, seed=SEED), info_keywords=EPISODE_INFO_KEYS)
    eval_env = Monitor(YearlyCaseEnv(loader=loader, split="val", ev_provider=ev_provider, cfg=cfg, shuffle_train=False, seed=SEED + 1000), info_keywords=EPISODE_INFO_KEYS)
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.15 * np.ones(n_actions),
    )

    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=str(CHECKPOINT_DIR), name_prefix="ddpg_yearly_single", save_replay_buffer=False, save_vecnormalize=False)
    eval_callback = EvalCallback(eval_env=eval_env, best_model_save_path=str(BEST_MODEL_DIR), log_path=str(LOG_DIR), eval_freq=EVAL_FREQ, n_eval_episodes=n_eval_episodes, deterministic=True, render=False)

    run_started_at = datetime.now().astimezone()
    run_id = run_started_at.strftime("%Y%m%d_%H%M%S")
    timestamp_start = run_started_at.isoformat(timespec="seconds")

    print("[startup] 正在创建 DDPG 模型...")
    policy_kwargs = dict(
        net_arch=dict(pi=POLICY_NET_ARCH_PI, qf=POLICY_NET_ARCH_QF),
        activation_fn=th.nn.ReLU,
    )
    model = DDPG(
        POLICY_NAME,
        train_env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        tau=TAU,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(TB_DIR),
        seed=SEED,
        device=resolved_device,
        verbose=1,
    )
    print(f"[startup] DDPG model device = {model.device}")

    config_row = build_config_row(
        cfg,
        run_id=run_id,
        timestamp_start=timestamp_start,
        n_train_cases=len(train_cases),
        n_val_cases=len(val_cases),
        n_test_cases=len(test_cases),
        device=str(model.device),
    )
    training_export_callback = build_training_export_callback(
        BaseCallback,
        pd=pd,
        output_path=TRAINING_EXPORT_XLSX,
        step_csv_path=TRAINING_STEP_DETAIL_CSV,
        episode_csv_path=TRAINING_EPISODE_SUMMARY_CSV,
        excel_engine=excel_engine,
        config_row=config_row,
    )

    print("[startup] 开始 DDPG 训练...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback, eval_callback, training_export_callback])

    final_model_path = MODEL_DIR / "ddpg_yearly_single_final"
    model.save(str(final_model_path))
    train_env.close()
    eval_env.close()

    print(f"DDPG 训练完成，最终模型已保存到: {final_model_path}")
    if getattr(training_export_callback, "training_duration_seconds", None) is not None:
        print(
            "DDPG 训练总耗时: "
            f"{training_export_callback.training_duration_hms} "
            f"({training_export_callback.training_duration_seconds:.3f} s)"
        )
    print(f"训练 step_detail CSV 已导出到: {TRAINING_STEP_DETAIL_CSV.resolve()}")
    print(f"训练诊断 Excel 已导出到: {TRAINING_EXPORT_XLSX.resolve()}")
    print(f"DDPG episode_summary CSV: {TRAINING_EPISODE_SUMMARY_CSV.resolve()}")
    runtime_row = write_training_runtime_summary(
        TRAINING_RUNTIME_SUMMARY_JSON,
        method="DDPG",
        total_episodes=TOTAL_EPISODES,
        total_timesteps=TOTAL_TIMESTEPS,
        training_duration_seconds=getattr(training_export_callback, "training_duration_seconds", None),
        device=str(model.device),
        seed=SEED,
    )
    print(f"训练耗时摘要 JSON 已导出到: {TRAINING_RUNTIME_SUMMARY_JSON.resolve()}")
    print(
        "DDPG training_runtime_summary: "
        f"{runtime_row['training_duration_hms']} "
        f"({runtime_row['training_duration_seconds']:.6f} s)"
    )
    best_model_path = BEST_MODEL_DIR / "best_model.zip"
    if best_model_path.exists():
        print(f"DDPG 当前最佳模型路径: {best_model_path}")

    validation_summary = read_validation_reward_summary(np, LOG_DIR / "evaluations.npz")
    best_validation_reward = validation_summary["best_validation_reward"]
    final_validation_reward = validation_summary["final_validation_reward"]
    print("DDPG training completed")
    if best_validation_reward is None:
        print("best validation reward: unavailable")
    else:
        print(f"best validation reward: {best_validation_reward:.6f}")
    if final_validation_reward is None:
        print("final validation reward: unavailable")
    else:
        print(f"final validation reward: {final_validation_reward:.6f}")
    print(f"training duration: {runtime_row['training_duration_hms']} ({runtime_row['training_duration_seconds']:.6f} s)")
    print("result paths:")
    print(f"  best_model: {best_model_path}")
    print(f"  evaluations: {LOG_DIR / 'evaluations.npz'}")
    print(f"  episode_summary: {TRAINING_EPISODE_SUMMARY_CSV}")
    print(f"  training_export: {TRAINING_EXPORT_XLSX}")
    print(f"  runtime_summary: {TRAINING_RUNTIME_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
