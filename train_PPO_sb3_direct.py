from __future__ import annotations

import csv
import importlib.util
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from train_TD3 import (
    EPISODE_DERIVED_COLUMNS,
    EPISODE_INFO_KEYS,
    EPISODE_SCALAR_TAGS,
    EPISODE_SUMMARY_COLUMNS,
    STEP_DETAIL_COLUMNS,
    STEP_INFO_COLUMNS,
    configure_stdio,
    format_duration,
    write_training_runtime_summary,
)


YEARLY_CSV_PATH = Path(r"D:\wangye\chengxu\3.30\yearly_data_sci.csv")
YEARLY_EV_PATH = Path(r"D:\wangye\chengxu\3.30\ev_data_s123_bilevel.npy")

MODEL_DIR = Path("./models/ppo_sb3_direct")
BEST_MODEL_DIR = MODEL_DIR / "best"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
LOG_DIR = Path("./logs/ppo_sb3_direct")
TB_DIR = Path("./tb/ppo_sb3_direct")
RESULT_DIR = Path("./results/ppo_sb3_direct_training")
CONFIG_CSV = RESULT_DIR / "train_config.csv"
TRAINING_EXPORT_XLSX = RESULT_DIR / "ppo_sb3_direct_training_export.xlsx"
TRAINING_STEP_DETAIL_CSV = RESULT_DIR / "ppo_sb3_direct_training_step_detail.csv"
TRAINING_RUNTIME_SUMMARY_JSON = RESULT_DIR / "training_runtime_summary.json"

SEED = 42
TOTAL_EPISODES = 4000
TOTAL_TIMESTEPS = TOTAL_EPISODES * 24
N_STEPS = 768
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.20
ENT_COEF = 0.00
VF_COEF = 0.50
MAX_GRAD_NORM = 0.50
EVAL_FREQ = 5_000
SAVE_FREQ = 5_000
REWARD_SCALE = 1e-5
VAL_DAYS_PER_MONTH = 2

POLICY_NAME = "MlpPolicy"
POLICY_NET_ARCH_PI = [256, 256]
POLICY_NET_ARCH_VF = [256, 256]
ACTIVATION_FN_NAME = "ReLU"
REQUESTED_DEVICE = "auto"

CONFIG_COLUMNS = (
    "run_id", "timestamp_start", "algorithm",
    "yearly_csv_path", "yearly_ev_path", "model_dir", "best_model_path", "final_model_path",
    "n_train_cases", "n_val_cases", "n_test_cases", "obs_dim", "action_dim",
    "training_duration_seconds", "training_duration_hms",
    "TOTAL_EPISODES", "TOTAL_TIMESTEPS", "SEED", "N_STEPS", "BATCH_SIZE", "LEARNING_RATE", "N_EPOCHS",
    "GAMMA", "GAE_LAMBDA", "CLIP_RANGE", "ENT_COEF", "VF_COEF", "MAX_GRAD_NORM",
    "EVAL_FREQ", "SAVE_FREQ", "VAL_DAYS_PER_MONTH", "REWARD_SCALE",
    "policy", "net_arch_pi", "net_arch_vf", "activation_fn", "device",
    "episode_length", "dt", "future_horizon", "exogenous_future_horizon",
    "grid_import_max", "grid_export_max",
    "gt_p_max", "gt_eta_e", "gt_ramp",
    "gb_h_max", "gb_ramp", "ac_c_max", "ec_c_max",
    "ees_p_max", "ees_e_cap", "ees_soc_init", "ees_soc_min", "ees_soc_max",
    "ees_terminal_soc_tolerance",
    "penalty_unserved_e", "penalty_unserved_h", "penalty_unserved_c", "penalty_depart_soc",
    "penalty_depart_energy_soft", "penalty_depart_energy_mid", "penalty_ev_depart_risk",
    "penalty_surplus_e", "penalty_surplus_h", "penalty_surplus_c",
    "penalty_export_e", "penalty_ev_export_guard", "penalty_ees_terminal_soc",
    "ev_discharge_price_threshold", "ev_charge_price_threshold", "ev_peak_export_tolerance_kw",
    "reward_storage_discharge_base", "reward_storage_charge_base", "reward_ev_target_timing_base",
    "ees_reward_discharge_soc_floor", "ees_reward_charge_soc_target", "reward_scale",
)


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

    missing_paths = [str(path) for path in (YEARLY_CSV_PATH, YEARLY_EV_PATH) if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing input files: {', '.join(missing_paths)}")

    excel_engine = resolve_excel_engine()
    required_modules = ["numpy", "pandas", "gymnasium", "torch", "stable_baselines3"]
    missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing_modules:
        raise ModuleNotFoundError(
            "Missing required packages: "
            f"{', '.join(missing_modules)}. "
            "Install numpy pandas gymnasium torch stable-baselines3 tensorboard openpyxl."
        )
    return excel_engine


def import_runtime_modules():
    print("[startup] Importing PPO training dependencies...")
    import numpy as np
    import pandas as pd
    import torch as th
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor

    from park_ies_env import IESConfig, ParkIESEnv
    from yearly_case_env import YearlyEVProvider, YearlyCaseEnv
    from yearly_csv_loader import YearlyCSVDataLoader

    print("[startup] PPO training dependencies imported.")
    return {
        "np": np,
        "pd": pd,
        "th": th,
        "PPO": PPO,
        "BaseCallback": BaseCallback,
        "CheckpointCallback": CheckpointCallback,
        "EvalCallback": EvalCallback,
        "Monitor": Monitor,
        "IESConfig": IESConfig,
        "ParkIESEnv": ParkIESEnv,
        "YearlyCSVDataLoader": YearlyCSVDataLoader,
        "YearlyEVProvider": YearlyEVProvider,
        "YearlyCaseEnv": YearlyCaseEnv,
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


def save_single_row_csv(path: Path, row: dict[str, Any], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def infer_unit(column_name: str) -> str:
    if column_name in {"global_step", "global_step_start", "global_step_end", "TOTAL_TIMESTEPS", "EVAL_FREQ", "SAVE_FREQ"}:
        return "step"
    if column_name == "TOTAL_EPISODES":
        return "episode"
    if column_name in {
        "episode_idx", "time_step", "SEED", "N_STEPS", "BATCH_SIZE", "N_EPOCHS",
        "VAL_DAYS_PER_MONTH", "episode_length", "n_train_cases", "n_val_cases",
        "n_test_cases", "obs_dim", "action_dim",
    } or column_name.endswith("_count") or column_name.endswith("_steps"):
        return "count"
    if column_name in {"month", "day_of_year"}:
        return "index"
    if column_name in {"season", "split", "policy", "activation_fn", "device", "run_id", "algorithm"}:
        return "label"
    if column_name == "timestamp_start":
        return "datetime"
    if column_name == "training_duration_seconds":
        return "s"
    if column_name == "training_duration_hms":
        return "HH:MM:SS"
    if column_name.endswith("_path") or column_name.endswith("_dir"):
        return "path"
    if "price" in column_name:
        return "price/kWh"
    if column_name.startswith("p_") or column_name.endswith("_load") or column_name in {
        "pv", "wt", "unmet_e", "unmet_h", "unmet_c", "surplus_e", "surplus_h",
        "surplus_c", "grid_overflow", "final_gt_power",
    }:
        return "kW"
    if column_name.endswith("_kwh") or column_name == "ees_e_cap":
        return "kWh"
    if "soc" in column_name:
        return "p.u."
    if column_name.startswith("cost_") or column_name.startswith("penalty_") or column_name == "system_cost":
        return "cost"
    if column_name.startswith("reward_") or column_name in {"guide_reward", "episode_reward_scaled", "episode_reward_raw"}:
        return "reward"
    if column_name in {"terminated", "gt_safe_feasible"}:
        return "bool"
    return ""


def describe_column(sheet_name: str, column_name: str) -> tuple[str, str]:
    if sheet_name == "step_detail":
        if column_name in {"global_step", "episode_idx"}:
            return f"PPO training callback field `{column_name}`.", "callback"
        return f"Per-step environment metric `{column_name}` exported with the TD3-compatible schema.", "env info"
    if sheet_name == "episode_summary":
        if column_name in EPISODE_DERIVED_COLUMNS:
            return f"PPO callback-derived episode field `{column_name}`.", "callback"
        return f"Episode aggregate `{column_name}` exported by ParkIESEnv.", "terminal info"
    return f"Training configuration field `{column_name}`.", "train_PPO_sb3_direct.py"


def build_column_description_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sheet_name, columns in (
        ("step_detail", STEP_DETAIL_COLUMNS),
        ("episode_summary", EPISODE_SUMMARY_COLUMNS),
        ("config", CONFIG_COLUMNS),
    ):
        for column_name in columns:
            meaning, source = describe_column(sheet_name, column_name)
            rows.append(
                {
                    "sheet": sheet_name,
                    "column_name": column_name,
                    "unit": infer_unit(column_name),
                    "meaning": meaning,
                    "source": source,
                }
            )
    return rows


def build_config_row(
    cfg: Any,
    *,
    run_id: str,
    timestamp_start: str,
    n_train_cases: int,
    n_val_cases: int,
    n_test_cases: int,
    obs_dim: int,
    action_dim: int,
    device: str,
    final_model_path: Path,
    best_model_path: Path,
) -> dict[str, Any]:
    cfg_values = asdict(cfg)
    return {
        "run_id": run_id,
        "timestamp_start": timestamp_start,
        "algorithm": "PPO",
        "yearly_csv_path": str(YEARLY_CSV_PATH),
        "yearly_ev_path": str(YEARLY_EV_PATH),
        "model_dir": str(MODEL_DIR.resolve()),
        "best_model_path": str(best_model_path.resolve()),
        "final_model_path": str(final_model_path.resolve()),
        "n_train_cases": int(n_train_cases),
        "n_val_cases": int(n_val_cases),
        "n_test_cases": int(n_test_cases),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "training_duration_seconds": None,
        "training_duration_hms": "",
        "TOTAL_EPISODES": TOTAL_EPISODES,
        "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS,
        "SEED": SEED,
        "N_STEPS": N_STEPS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "N_EPOCHS": N_EPOCHS,
        "GAMMA": GAMMA,
        "GAE_LAMBDA": GAE_LAMBDA,
        "CLIP_RANGE": CLIP_RANGE,
        "ENT_COEF": ENT_COEF,
        "VF_COEF": VF_COEF,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "EVAL_FREQ": EVAL_FREQ,
        "SAVE_FREQ": SAVE_FREQ,
        "VAL_DAYS_PER_MONTH": VAL_DAYS_PER_MONTH,
        "REWARD_SCALE": REWARD_SCALE,
        "policy": POLICY_NAME,
        "net_arch_pi": str(POLICY_NET_ARCH_PI),
        "net_arch_vf": str(POLICY_NET_ARCH_VF),
        "activation_fn": ACTIVATION_FN_NAME,
        "device": device,
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


def build_training_export_callback(
    base_callback_cls,
    *,
    pd,
    output_path: Path,
    step_csv_path: Path,
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
                raise RuntimeError("TrainingExportCallback only supports one training environment.")
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
            missing = [
                column
                for column in EPISODE_SUMMARY_COLUMNS
                if column not in EPISODE_DERIVED_COLUMNS and column not in info
            ]
            if missing:
                raise KeyError(f"Missing required episode_summary columns in terminal info: {missing}")
            row = {
                "episode_idx": int(self.current_episode_idx),
                "global_step_start": int(self.current_episode_start_step or global_step_end),
                "global_step_end": int(global_step_end),
                "episode_reward_scaled": float(self.current_episode_reward_scaled),
                "final_ees_soc": float(info.get("final_ees_soc", info.get("ees_soc", 0.0))),
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
                raise RuntimeError("TrainingExportCallback only supports one training environment.")

            info = dict(infos[0])
            self._record_episode_scalars(info)
            step_row = self._build_step_row(info)
            self.step_rows.append(step_row)
            self.current_episode_reward_scaled += float(info.get("reward_scaled", 0.0))

            dones = self.locals.get("dones", [False])
            done_flag = dones[0] if hasattr(dones, "__len__") else dones
            episode_done = bool(step_row.get("terminated", False)) or bool(done_flag)
            if episode_done:
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
            step_df = pd.DataFrame(self.step_rows, columns=STEP_DETAIL_COLUMNS)
            episode_df = pd.DataFrame(self.episode_rows, columns=EPISODE_SUMMARY_COLUMNS)
            config_df = pd.DataFrame([{column: config_row.get(column) for column in CONFIG_COLUMNS}], columns=CONFIG_COLUMNS)
            description_df = pd.DataFrame(column_description_rows, columns=["sheet", "column_name", "unit", "meaning", "source"])
            step_df.to_csv(step_csv_path, index=False, encoding="utf-8-sig")
            with pd.ExcelWriter(output_path, engine=excel_engine) as writer:
                episode_df.to_excel(writer, sheet_name="episode_summary", index=False)
                config_df.to_excel(writer, sheet_name="config", index=False)
                description_df.to_excel(writer, sheet_name="column_description", index=False)
            print(f"[export] PPO training step_detail CSV saved to: {step_csv_path.resolve()}")
            print(f"[export] PPO training workbook saved to: {output_path.resolve()}")

    return TrainingExportCallback()


def main() -> None:
    configure_stdio()
    try:
        excel_engine = preflight_check()
        modules = import_runtime_modules()
    except Exception as exc:
        print(f"[startup] PPO training startup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    pd = modules["pd"]
    np = modules["np"]
    th = modules["th"]
    PPO = modules["PPO"]
    BaseCallback = modules["BaseCallback"]
    CheckpointCallback = modules["CheckpointCallback"]
    EvalCallback = modules["EvalCallback"]
    Monitor = modules["Monitor"]
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

    print("[startup] Loading yearly training data...")
    loader = YearlyCSVDataLoader(YEARLY_CSV_PATH, val_days_per_month=VAL_DAYS_PER_MONTH)
    train_cases, val_cases, test_cases = loader.load()
    loader.summary()
    print(f"train cases: {len(train_cases)}")
    print(f"val cases: {len(val_cases)}")
    print(f"test cases: {len(test_cases)}")
    if not val_cases:
        raise RuntimeError("Validation split is empty. Please check dataset splitting.")

    n_eval_episodes = len(val_cases)
    print("[startup] Loading yearly EV data...")
    ev_provider = YearlyEVProvider(YEARLY_EV_PATH)

    cfg = IESConfig(episode_length=24, dt=1.0, reward_scale=REWARD_SCALE)
    probe_case = val_cases[0] if val_cases else train_cases[0]
    probe_ev_data = ev_provider(probe_case)
    probe_env = ParkIESEnv(cfg=cfg, ts_data=probe_case.ts_data, ev_data=probe_ev_data)
    obs_dim = int(probe_env.obs_dim)
    action_dim = int(probe_env.action_space.shape[0])
    print("[startup] PPO config aligned with current TD3 environment:")
    print(f"  reward_scale = {cfg.reward_scale}")
    print(f"  total_episodes = {TOTAL_EPISODES}")
    print(f"  total_timesteps = {TOTAL_TIMESTEPS}")
    print(f"  learning_rate = {LEARNING_RATE}")
    print(f"  n_steps = {N_STEPS}")
    print(f"  batch_size = {BATCH_SIZE}")
    print(f"  n_epochs = {N_EPOCHS}")
    print(f"  gamma = {GAMMA}")
    print(f"  gae_lambda = {GAE_LAMBDA}")
    print(f"  clip_range = {CLIP_RANGE}")
    print(f"  ent_coef = {ENT_COEF}")
    print(f"  vf_coef = {VF_COEF}")
    print(f"  max_grad_norm = {MAX_GRAD_NORM}")
    print(f"  val_days_per_month = {VAL_DAYS_PER_MONTH}")
    print(f"  n_eval_episodes = {n_eval_episodes}")
    print(f"  obs_dim = {obs_dim}")
    print(f"  action_dim = {action_dim}")
    print(f"  future_horizon = {cfg.future_horizon}")
    print(f"  exogenous_future_horizon = {cfg.exogenous_future_horizon}")
    probe_env.close()

    train_env = Monitor(
        YearlyCaseEnv(
            loader=loader,
            split="train",
            ev_provider=ev_provider,
            cfg=cfg,
            shuffle_train=True,
            seed=SEED,
        ),
        info_keywords=EPISODE_INFO_KEYS,
    )
    eval_env = Monitor(
        YearlyCaseEnv(
            loader=loader,
            split="val",
            ev_provider=ev_provider,
            cfg=cfg,
            shuffle_train=False,
            seed=SEED + 1000,
        ),
        info_keywords=EPISODE_INFO_KEYS,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_sb3_direct",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(BEST_MODEL_DIR),
        log_path=str(LOG_DIR),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    run_started_at = datetime.now().astimezone()
    run_id = run_started_at.strftime("%Y%m%d_%H%M%S")
    timestamp_start = run_started_at.isoformat(timespec="seconds")
    final_model_path = MODEL_DIR / "ppo_sb3_direct_final"
    best_model_path = BEST_MODEL_DIR / "best_model.zip"

    print("[startup] Creating PPO model...")
    model = PPO(
        policy=POLICY_NAME,
        env=train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=dict(net_arch=dict(pi=POLICY_NET_ARCH_PI, vf=POLICY_NET_ARCH_VF), activation_fn=th.nn.ReLU),
        verbose=0,
        seed=SEED,
        device=resolved_device,
        tensorboard_log=str(TB_DIR),
    )
    print(f"[startup] PPO model device = {model.device}")

    config_row = build_config_row(
        cfg,
        run_id=run_id,
        timestamp_start=timestamp_start,
        n_train_cases=len(train_cases),
        n_val_cases=len(val_cases),
        n_test_cases=len(test_cases),
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=str(model.device),
        final_model_path=final_model_path,
        best_model_path=best_model_path,
    )
    training_export_callback = build_training_export_callback(
        BaseCallback,
        pd=pd,
        output_path=TRAINING_EXPORT_XLSX,
        step_csv_path=TRAINING_STEP_DETAIL_CSV,
        excel_engine=excel_engine,
        config_row=config_row,
    )

    print("[startup] Starting PPO training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback, eval_callback, training_export_callback])

    model.save(str(final_model_path))
    save_single_row_csv(CONFIG_CSV, config_row, CONFIG_COLUMNS)
    train_env.close()
    eval_env.close()

    print(f"PPO training finished, final model saved to: {final_model_path}")
    if getattr(training_export_callback, "training_duration_seconds", None) is not None:
        print(
            "PPO training duration: "
            f"{training_export_callback.training_duration_hms} "
            f"({training_export_callback.training_duration_seconds:.3f} s)"
        )
    print(f"PPO training step_detail CSV exported to: {TRAINING_STEP_DETAIL_CSV.resolve()}")
    print(f"PPO training workbook exported to: {TRAINING_EXPORT_XLSX.resolve()}")
    print(f"PPO training config exported to: {CONFIG_CSV.resolve()}")
    runtime_row = write_training_runtime_summary(
        TRAINING_RUNTIME_SUMMARY_JSON,
        method="PPO",
        total_episodes=TOTAL_EPISODES,
        total_timesteps=TOTAL_TIMESTEPS,
        training_duration_seconds=getattr(training_export_callback, "training_duration_seconds", None),
        device=str(model.device),
        seed=SEED,
    )
    print(f"PPO training runtime summary exported to: {TRAINING_RUNTIME_SUMMARY_JSON.resolve()}")
    print(
        "PPO training_runtime_summary: "
        f"{runtime_row['training_duration_hms']} "
        f"({runtime_row['training_duration_seconds']:.6f} s)"
    )
    if best_model_path.exists():
        print(f"PPO best model path: {best_model_path}")


if __name__ == "__main__":
    main()
