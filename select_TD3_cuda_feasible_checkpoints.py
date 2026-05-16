from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import TD3

from ies_config import IESConfig
from park_ies_env import ParkIESEnv
from yearly_case_env import YearlyEVProvider
from yearly_csv_loader import YearlyCSVDataLoader


DEFAULT_SEEDS = [42, 2024, 2025]
OUTPUT_DIR = Path("results/td3_cuda_multiseed_feasible_selected")
YEARLY_CSV_PATH = Path(r"D:\wangye\chengxu\3.30\yearly_data_sci.csv")
YEARLY_EV_PATH = Path(r"D:\wangye\chengxu\3.30\ev_data_s123_bilevel.npy")
VAL_DAYS_PER_MONTH = 2
REWARD_SCALE = 1e-5
ACTION_NOISE_TYPE = None
TARGET_POLICY_NOISE = 0.20
TARGET_NOISE_CLIP = 0.50
CHECKPOINT_PATTERN = re.compile(r"_(\d+)_steps\.zip$")

SEED_SUMMARY_COLUMNS = [
    "method",
    "seed",
    "selected_checkpoint_step",
    "selected_checkpoint_path",
    "selected_reason",
    "n_days",
    "mean_total_system_cost",
    "mean_total_penalties",
    "mean_total_cost_plus_penalty",
    "terminal_ees_feasible_days",
    "terminal_ees_feasible_ratio",
    "min_final_ees_soc",
    "sum_terminal_ees_shortage_kwh",
    "mean_terminal_ees_shortage_kwh_per_day",
    "sum_penalty_terminal_ees_soc",
    "mean_total_grid_buy_kwh",
    "mean_total_grid_sell_kwh",
    "mean_total_storage_peak_shaved_kwh",
    "mean_total_depart_energy_shortage_kwh",
    "total_unmet_e",
    "total_unmet_h",
    "total_unmet_c",
    "total_penalty_ev_export_guard",
    "val_EES_feasible",
    "val_terminal_ees_feasible_days",
    "val_terminal_ees_feasible_ratio",
    "val_terminal_shortage",
    "val_cost_plus_penalty",
    "val_reward",
    "test_EES_feasible",
    "test_terminal_shortage",
    "test_cost_plus_penalty",
    "model_path",
    "result_dir",
]

AGGREGATE_METRICS = [
    "mean_total_system_cost",
    "mean_total_penalties",
    "mean_total_cost_plus_penalty",
    "terminal_ees_feasible_days",
    "terminal_ees_feasible_ratio",
    "min_final_ees_soc",
    "sum_terminal_ees_shortage_kwh",
    "mean_terminal_ees_shortage_kwh_per_day",
    "sum_penalty_terminal_ees_soc",
    "mean_total_grid_buy_kwh",
    "mean_total_grid_sell_kwh",
    "mean_total_storage_peak_shaved_kwh",
    "mean_total_depart_energy_shortage_kwh",
    "total_unmet_e",
    "total_unmet_h",
    "total_unmet_c",
    "total_penalty_ev_export_guard",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select CUDA TD3 checkpoints by validation EES feasibility, then test selected checkpoints."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--skip-existing-tests",
        action="store_true",
        help="Keep existing feasible-selected test directories when all required outputs exist.",
    )
    return parser.parse_args()


def validate_cuda_device(device: str) -> str:
    cleaned = str(device).strip()
    lowered = cleaned.lower()
    if lowered != "cuda" and not lowered.startswith("cuda:"):
        raise SystemExit("Use --device cuda or --device cuda:<index> for this CUDA selection flow.")
    return cleaned


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse checkpoint step from {path}")
    return int(match.group(1))


def checkpoint_dir(seed: int) -> Path:
    return Path(f"models/td3_yearly_single_cuda_seed_{int(seed)}/checkpoints")


def feasible_model_path(seed: int) -> Path:
    return Path(f"models/td3_yearly_single_cuda_feasible_seed_{int(seed)}/best/best_model.zip")


def feasible_result_dir(seed: int) -> Path:
    return Path(f"results/td3_yearly_test_cuda_feasible_seed_{int(seed)}")


def reward_best_result_dir(seed: int) -> Path:
    return Path(f"results/td3_yearly_test_cuda_seed_{int(seed)}")


def original_result_dir(seed: int) -> Path:
    return Path(f"results/td3_yearly_test_seed_{int(seed)}")


def rollout_one_day(model: TD3, env: ParkIESEnv) -> tuple[list[dict[str, Any]], float]:
    obs, _ = env.reset()
    done = False
    infos: list[dict[str, Any]] = []
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        infos.append(dict(info))
        total_reward += float(reward)
        done = bool(terminated or truncated)
    return infos, total_reward


def evaluate_checkpoint(
    *,
    model_path: Path,
    split: str,
    loader: YearlyCSVDataLoader,
    ev_provider: YearlyEVProvider,
    device: str,
) -> dict[str, float | int]:
    model = TD3.load(str(model_path), device=device)
    cfg = IESConfig(episode_length=24, dt=1.0, reward_scale=REWARD_SCALE)
    rows: list[dict[str, float | bool]] = []

    for case in loader.get_cases(split):
        env = ParkIESEnv(cfg=cfg, ts_data=case.ts_data, ev_data=ev_provider(case))
        infos, total_reward = rollout_one_day(model, env)
        env.close()

        final_info = infos[-1] if infos else {}
        total_system_cost = float(sum(item.get("system_cost", 0.0) for item in infos))
        total_penalties = float(sum(item.get("penalty_cost", 0.0) for item in infos))
        rows.append(
            {
                "total_reward": float(total_reward),
                "total_cost_plus_penalty": total_system_cost + total_penalties,
                "terminal_ees_shortage_kwh": float(
                    final_info.get("episode_terminal_ees_shortage_kwh", 0.0)
                ),
                "ees_terminal_soc_feasible": bool(
                    final_info.get("ees_terminal_soc_feasible", True)
                ),
            }
        )

    df = pd.DataFrame(rows)
    feasible = df["ees_terminal_soc_feasible"].astype(bool)
    n_days = int(len(df))
    feasible_days = int(feasible.sum())
    return {
        "n_days": n_days,
        "terminal_ees_feasible_days": feasible_days,
        "terminal_ees_feasible_ratio": float(feasible_days / n_days) if n_days else math.nan,
        "terminal_ees_shortage_kwh": float(df["terminal_ees_shortage_kwh"].sum()),
        "cost_plus_penalty": float(df["total_cost_plus_penalty"].mean()),
        "reward": float(df["total_reward"].mean()),
    }


def load_eval_reward_map(log_dir: Path) -> dict[int, float]:
    eval_path = log_dir / "evaluations.npz"
    if not eval_path.exists():
        return {}
    data = np.load(eval_path)
    timesteps = np.asarray(data["timesteps"], dtype=int)
    results = np.asarray(data["results"], dtype=float)
    rewards = np.nanmean(results, axis=1) if results.ndim > 1 else results
    return {int(step): float(reward) for step, reward in zip(timesteps, rewards)}


def select_checkpoint(validation_df: pd.DataFrame) -> pd.Series:
    n_val = int(validation_df["val_n_days"].max())
    full_feasible = validation_df[
        validation_df["val_terminal_ees_feasible_days"].astype(int) >= n_val
    ].copy()
    candidates = full_feasible if not full_feasible.empty else validation_df.copy()
    ordered = candidates.sort_values(
        by=[
            "val_terminal_ees_feasible_days",
            "val_terminal_ees_shortage_kwh",
            "val_cost_plus_penalty",
            "val_reward",
        ],
        ascending=[False, True, True, False],
        kind="mergesort",
    )
    return ordered.iloc[0]


def selection_reason(selected: pd.Series, validation_df: pd.DataFrame) -> str:
    n_val = int(validation_df["val_n_days"].max())
    full_count = int(
        (validation_df["val_terminal_ees_feasible_days"].astype(int) >= n_val).sum()
    )
    if full_count > 0:
        return (
            f"Selected from {full_count} validation-full-feasible checkpoints "
            f"({n_val}/{n_val}) by minimum terminal shortage, then cost, then reward."
        )
    best_feasible = int(validation_df["val_terminal_ees_feasible_days"].max())
    return (
        f"No validation-full-feasible checkpoint existed; selected among checkpoints with "
        f"maximum validation feasible days ({best_feasible}/{n_val}) by shortage, cost, reward."
    )


def scan_seed_checkpoints(
    *,
    seed: int,
    loader: YearlyCSVDataLoader,
    ev_provider: YearlyEVProvider,
    device: str,
) -> pd.DataFrame:
    ckpt_dir = checkpoint_dir(seed)
    checkpoints = sorted(ckpt_dir.glob("td3_yearly_single_*_steps.zip"), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No CUDA TD3 checkpoints found for seed {seed}: {ckpt_dir}")

    eval_reward_map = load_eval_reward_map(Path(f"logs/td3_yearly_single_cuda_seed_{seed}"))
    rows: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        metrics = evaluate_checkpoint(
            model_path=checkpoint,
            split="val",
            loader=loader,
            ev_provider=ev_provider,
            device=device,
        )
        rows.append(
            {
                "seed": int(seed),
                "checkpoint_step": int(step),
                "checkpoint_path": str(checkpoint),
                "val_n_days": int(metrics["n_days"]),
                "val_terminal_ees_feasible_days": int(
                    metrics["terminal_ees_feasible_days"]
                ),
                "val_terminal_ees_feasible_ratio": float(
                    metrics["terminal_ees_feasible_ratio"]
                ),
                "val_terminal_ees_shortage_kwh": float(
                    metrics["terminal_ees_shortage_kwh"]
                ),
                "val_cost_plus_penalty": float(metrics["cost_plus_penalty"]),
                "val_reward": float(metrics["reward"]),
                "eval_callback_reward": eval_reward_map.get(step, np.nan),
            }
        )
        print(
            f"[val] seed={seed} step={step} "
            f"feasible={metrics['terminal_ees_feasible_days']}/{metrics['n_days']} "
            f"shortage={metrics['terminal_ees_shortage_kwh']:.6f} "
            f"cost={metrics['cost_plus_penalty']:.6f} "
            f"reward={metrics['reward']:.9f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def test_outputs_complete(result_dir: Path) -> bool:
    required = [
        "daily_summary.csv",
        "timeseries_detail.csv",
        "runtime_summary.json",
        "td3_test_export.xlsx",
    ]
    return all((result_dir / name).exists() for name in required)


def copy_selected_model(seed: int, checkpoint_path: Path) -> Path:
    target = feasible_model_path(seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, target)
    return target


def update_runtime_summary(
    *,
    seed: int,
    result_dir: Path,
    model_path: Path,
    selected_row: pd.Series,
    device: str,
) -> None:
    runtime_path = result_dir / "runtime_summary.json"
    data: dict[str, Any] = {}
    if runtime_path.exists():
        data = json.loads(runtime_path.read_text(encoding="utf-8"))
    data.update(
        {
            "method": "TD3-CUDA-feasible-selected",
            "seed": int(seed),
            "device": str(device),
            "run_name": f"cuda_feasible_seed_{seed}",
            "model_path": str(model_path),
            "result_dir": str(result_dir),
            "selected_checkpoint_path": str(selected_row["checkpoint_path"]),
            "selected_checkpoint_step": int(selected_row["checkpoint_step"]),
            "selection_rule": (
                "validation feasibility first: max feasible days, min terminal shortage, "
                "min cost+penalty, max reward; test metrics are not used for selection"
            ),
            "deterministic": True,
            "action_noise_type": ACTION_NOISE_TYPE,
            "target_policy_noise": TARGET_POLICY_NOISE,
            "target_noise_clip": TARGET_NOISE_CLIP,
        }
    )
    runtime_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_selected_test(
    *,
    seed: int,
    selected_row: pd.Series,
    device: str,
    skip_existing: bool,
) -> Path:
    selected_checkpoint = Path(str(selected_row["checkpoint_path"]))
    model_path = copy_selected_model(seed, selected_checkpoint)
    result_dir = feasible_result_dir(seed)
    if skip_existing and test_outputs_complete(result_dir):
        print(f"[skip] seed={seed} feasible-selected test exists: {result_dir}", flush=True)
    else:
        command = [
            sys.executable,
            "test_TD3.py",
            "--seed",
            str(seed),
            "--run-name",
            f"cuda_feasible_seed_{seed}",
            "--device",
            device,
        ]
        print(f"[test] {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)
    update_runtime_summary(
        seed=seed,
        result_dir=result_dir,
        model_path=model_path,
        selected_row=selected_row,
        device=device,
    )
    return result_dir


def numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def bool_column(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    values = df[column]
    if values.dtype == bool:
        return values.fillna(default)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def summarize_result_dir(
    *,
    method: str,
    seed: int,
    result_dir: Path,
    model_path: Path,
    selected_row: pd.Series | None = None,
) -> dict[str, object]:
    daily_path = result_dir / "daily_summary.csv"
    if not daily_path.exists():
        raise FileNotFoundError(f"Missing daily summary: {daily_path}")
    daily = pd.read_csv(daily_path)
    n_days = int(len(daily))
    total_system_cost = numeric_column(daily, "total_system_cost")
    total_penalties = numeric_column(daily, "total_penalties")
    if "total_cost_plus_penalty" in daily.columns:
        cost_plus_penalty = numeric_column(daily, "total_cost_plus_penalty")
    else:
        cost_plus_penalty = total_system_cost + total_penalties

    feasible = bool_column(daily, "ees_terminal_soc_feasible")
    feasible_days = int(feasible.sum())
    terminal_shortage = numeric_column(daily, "terminal_ees_shortage_kwh")

    row: dict[str, object] = {
        "method": method,
        "seed": int(seed),
        "selected_checkpoint_step": "",
        "selected_checkpoint_path": "",
        "selected_reason": "",
        "n_days": n_days,
        "mean_total_system_cost": float(total_system_cost.mean()),
        "mean_total_penalties": float(total_penalties.mean()),
        "mean_total_cost_plus_penalty": float(cost_plus_penalty.mean()),
        "terminal_ees_feasible_days": feasible_days,
        "terminal_ees_feasible_ratio": float(feasible_days / n_days) if n_days else np.nan,
        "min_final_ees_soc": float(numeric_column(daily, "final_ees_soc", np.nan).min()),
        "sum_terminal_ees_shortage_kwh": float(terminal_shortage.sum()),
        "mean_terminal_ees_shortage_kwh_per_day": float(terminal_shortage.mean()),
        "sum_penalty_terminal_ees_soc": float(
            numeric_column(daily, "total_penalty_terminal_ees_soc").sum()
        ),
        "mean_total_grid_buy_kwh": float(numeric_column(daily, "total_grid_buy_kwh").mean()),
        "mean_total_grid_sell_kwh": float(numeric_column(daily, "total_grid_sell_kwh").mean()),
        "mean_total_storage_peak_shaved_kwh": float(
            numeric_column(daily, "total_storage_peak_shaved_kwh").mean()
        ),
        "mean_total_depart_energy_shortage_kwh": float(
            numeric_column(daily, "total_depart_energy_shortage_kwh").mean()
        ),
        "total_unmet_e": float(numeric_column(daily, "total_unmet_e").sum()),
        "total_unmet_h": float(numeric_column(daily, "total_unmet_h").sum()),
        "total_unmet_c": float(numeric_column(daily, "total_unmet_c").sum()),
        "total_penalty_ev_export_guard": float(
            numeric_column(daily, "total_penalty_ev_export_guard").sum()
        ),
        "val_EES_feasible": "",
        "val_terminal_ees_feasible_days": "",
        "val_terminal_ees_feasible_ratio": "",
        "val_terminal_shortage": "",
        "val_cost_plus_penalty": "",
        "val_reward": "",
        "test_EES_feasible": f"{feasible_days}/{n_days}",
        "test_terminal_shortage": float(terminal_shortage.sum()),
        "test_cost_plus_penalty": float(cost_plus_penalty.mean()),
        "model_path": str(model_path),
        "result_dir": str(result_dir),
    }

    if selected_row is not None:
        row.update(
            {
                "selected_checkpoint_step": int(selected_row["checkpoint_step"]),
                "selected_checkpoint_path": str(selected_row["checkpoint_path"]),
                "selected_reason": str(selected_row["selected_reason"]),
                "val_EES_feasible": (
                    f"{int(selected_row['val_terminal_ees_feasible_days'])}/"
                    f"{int(selected_row['val_n_days'])}"
                ),
                "val_terminal_ees_feasible_days": int(
                    selected_row["val_terminal_ees_feasible_days"]
                ),
                "val_terminal_ees_feasible_ratio": float(
                    selected_row["val_terminal_ees_feasible_ratio"]
                ),
                "val_terminal_shortage": float(selected_row["val_terminal_ees_shortage_kwh"]),
                "val_cost_plus_penalty": float(selected_row["val_cost_plus_penalty"]),
                "val_reward": float(selected_row["val_reward"]),
            }
        )
    return row


def mean_std_text(mean_value: float, std_value: float) -> str:
    if pd.isna(mean_value):
        return "NA"
    return f"{float(mean_value):.6g} +/- {float(std_value):.6g}"


def build_algorithm_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, group in seed_df.groupby("method", sort=False):
        row: dict[str, object] = {
            "method": method,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in AGGREGATE_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            mean_value = float(values.mean()) if not values.empty else np.nan
            std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_mean_std"] = mean_std_text(mean_value, std_value)
        rows.append(row)
    return pd.DataFrame(rows)


def build_vs_original_summary(algorithm_df: pd.DataFrame) -> pd.DataFrame:
    by_method = algorithm_df.set_index("method")
    rows: list[dict[str, object]] = []
    for metric in AGGREGATE_METRICS:
        original_mean = by_method.at["TD3-original", f"{metric}_mean"]
        original_std = by_method.at["TD3-original", f"{metric}_std"]
        selected_mean = by_method.at[
            "TD3-CUDA-feasible-selected", f"{metric}_mean"
        ]
        selected_std = by_method.at["TD3-CUDA-feasible-selected", f"{metric}_std"]
        delta = float(selected_mean - original_mean)
        if abs(float(original_mean)) > 1e-12:
            delta_percent = float(delta / abs(float(original_mean)) * 100.0)
        else:
            delta_percent = np.nan
        rows.append(
            {
                "metric": metric,
                "original_mean": original_mean,
                "original_std": original_std,
                "feasible_selected_mean": selected_mean,
                "feasible_selected_std": selected_std,
                "selected_minus_original": delta,
                "selected_minus_original_percent": delta_percent,
                "original_mean_std": mean_std_text(original_mean, original_std),
                "feasible_selected_mean_std": mean_std_text(selected_mean, selected_std),
            }
        )
    return pd.DataFrame(rows)


def candidate_status(seed_df: pd.DataFrame, algorithm_df: pd.DataFrame) -> tuple[bool, list[str], dict[str, bool]]:
    selected = seed_df[seed_df["method"] == "TD3-CUDA-feasible-selected"]
    by_method = algorithm_df.set_index("method")
    old = by_method.loc["TD3-original"]
    new = by_method.loc["TD3-CUDA-feasible-selected"]
    full_feasible_count = int((selected["terminal_ees_feasible_days"].astype(int) >= 60).sum())

    conditions = {
        "cost_not_worse_than_0p5pct": bool(
            float(new["mean_total_cost_plus_penalty_mean"])
            <= float(old["mean_total_cost_plus_penalty_mean"]) * 1.005
        ),
        "ees_terminal_feasibility_ok": bool(
            float(new["terminal_ees_feasible_ratio_mean"]) >= 0.98
            or full_feasible_count >= 2
        ),
        "terminal_shortage_per_day_ok": bool(
            float(new["mean_terminal_ees_shortage_kwh_per_day_mean"]) <= 2.0
        ),
        "ev_departure_shortage_ok": bool(
            abs(float(new["mean_total_depart_energy_shortage_kwh_mean"])) <= 1e-3
        ),
        "unmet_load_ok": bool(
            abs(float(selected["total_unmet_e"].sum())) <= 1e-6
            and abs(float(selected["total_unmet_h"].sum())) <= 1e-6
            and abs(float(selected["total_unmet_c"].sum())) <= 1e-6
        ),
    }
    reasons = [name for name, passed in conditions.items() if not passed]
    return all(conditions.values()), reasons, conditions


def markdown_table(df: pd.DataFrame, columns: list[str], float_digits: int = 6) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: list[str] = []
    for _, row in df.iterrows():
        values: list[str] = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, (float, np.floating)):
                values.append("NA" if pd.isna(value) else f"{float(value):.{float_digits}g}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def write_report(
    *,
    output_dir: Path,
    selected_df: pd.DataFrame,
    seed_summary_df: pd.DataFrame,
    algorithm_df: pd.DataFrame,
    vs_original_df: pd.DataFrame,
    candidate: bool,
    reasons: list[str],
    conditions: dict[str, bool],
) -> None:
    selected_rows = seed_summary_df[
        seed_summary_df["method"] == "TD3-CUDA-feasible-selected"
    ].copy()
    reward_best_rows = seed_summary_df[
        seed_summary_df["method"] == "TD3-CUDA-reward-best"
    ].copy()
    comparison_rows = algorithm_df[
        algorithm_df["method"].isin(
            ["TD3-CUDA-reward-best", "TD3-CUDA-feasible-selected", "TD3-original"]
        )
    ].copy()
    condition_df = pd.DataFrame(
        [{"condition": name, "passed": passed} for name, passed in conditions.items()]
    )
    non_full = selected_rows[selected_rows["terminal_ees_feasible_days"].astype(int) < 60]

    lines = [
        "# TD3 CUDA feasibility-first checkpoint selection report",
        "",
        f"Generated at: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "Selection rule uses validation metrics only: maximize validation terminal EES feasible days, then minimize validation terminal EES shortage, then minimize validation cost+penalty, then maximize validation reward.",
        "",
        "Test-set metrics are used only after the checkpoint is selected.",
        "",
        f"candidate_for_paper_main_result = {candidate}",
        "",
    ]
    if candidate:
        lines.append("The feasible-selected CUDA run satisfies the automatic replacement checks.")
    else:
        lines.append(
            "The feasible-selected CUDA run is useful as a reproducibility and checkpoint-selection record, but it should not replace the original TD3 result as the paper main result."
        )
        if reasons:
            lines.extend(["", "Reasons not passed: " + ", ".join(reasons) + "."])
    if not non_full.empty:
        lines.extend(
            [
                "",
                "Seeds still below 60/60 terminal EES feasibility: "
                + ", ".join(
                    f"seed {int(row.seed)} ({int(row.terminal_ees_feasible_days)}/60)"
                    for _, row in non_full.iterrows()
                )
                + ".",
            ]
        )

    lines.extend(
        [
            "",
            "## Selected Checkpoints",
            "",
            markdown_table(
                selected_df,
                [
                    "seed",
                    "selected_checkpoint_step",
                    "val_EES_feasible",
                    "val_terminal_shortage",
                    "val_cost_plus_penalty",
                    "val_reward",
                    "test_EES_feasible",
                    "test_terminal_shortage",
                    "test_cost_plus_penalty",
                    "selected_reason",
                ],
            ),
            "",
            "## Selected CUDA Versus Reward-Best CUDA",
            "",
            markdown_table(
                comparison_rows,
                [
                    "method",
                    "n_seeds",
                    "mean_total_cost_plus_penalty_mean_std",
                    "terminal_ees_feasible_ratio_mean_std",
                    "mean_terminal_ees_shortage_kwh_per_day_mean_std",
                    "sum_terminal_ees_shortage_kwh_mean_std",
                ],
            ),
            "",
            "## Selected CUDA Versus Original TD3",
            "",
            markdown_table(
                vs_original_df[
                    vs_original_df["metric"].isin(
                        [
                            "mean_total_cost_plus_penalty",
                            "terminal_ees_feasible_ratio",
                            "mean_terminal_ees_shortage_kwh_per_day",
                            "sum_terminal_ees_shortage_kwh",
                            "mean_total_depart_energy_shortage_kwh",
                            "total_unmet_e",
                            "total_unmet_h",
                            "total_unmet_c",
                        ]
                    )
                ],
                [
                    "metric",
                    "original_mean_std",
                    "feasible_selected_mean_std",
                    "selected_minus_original",
                    "selected_minus_original_percent",
                ],
            ),
            "",
            "## Replacement Conditions",
            "",
            markdown_table(condition_df, ["condition", "passed"]),
            "",
            "## Reward-Best Seed Details",
            "",
            markdown_table(
                reward_best_rows,
                [
                    "seed",
                    "terminal_ees_feasible_days",
                    "mean_total_cost_plus_penalty",
                    "sum_terminal_ees_shortage_kwh",
                    "mean_terminal_ees_shortage_kwh_per_day",
                ],
            ),
            "",
        ]
    )
    (output_dir / "selected_cuda_checkpoint_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    seeds = [int(seed) for seed in args.seeds]
    device = validate_cuda_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(42)

    loader = YearlyCSVDataLoader(YEARLY_CSV_PATH, val_days_per_month=VAL_DAYS_PER_MONTH)
    loader.load()
    ev_provider = YearlyEVProvider(YEARLY_EV_PATH)

    selected_rows: list[dict[str, object]] = []
    selected_series_by_seed: dict[int, pd.Series] = {}
    for seed in seeds:
        validation_df = scan_seed_checkpoints(
            seed=seed,
            loader=loader,
            ev_provider=ev_provider,
            device=device,
        )
        selected = select_checkpoint(validation_df).copy()
        selected["selected_reason"] = selection_reason(selected, validation_df)
        selected_series_by_seed[seed] = selected
        selected_rows.append(
            {
                "seed": int(seed),
                "selected_checkpoint_step": int(selected["checkpoint_step"]),
                "selected_checkpoint_path": str(selected["checkpoint_path"]),
                "selected_reason": str(selected["selected_reason"]),
                "val_EES_feasible": (
                    f"{int(selected['val_terminal_ees_feasible_days'])}/"
                    f"{int(selected['val_n_days'])}"
                ),
                "val_terminal_ees_feasible_days": int(
                    selected["val_terminal_ees_feasible_days"]
                ),
                "val_terminal_ees_feasible_ratio": float(
                    selected["val_terminal_ees_feasible_ratio"]
                ),
                "val_terminal_shortage": float(selected["val_terminal_ees_shortage_kwh"]),
                "val_cost_plus_penalty": float(selected["val_cost_plus_penalty"]),
                "val_reward": float(selected["val_reward"]),
                "eval_callback_reward": float(selected["eval_callback_reward"]),
            }
        )
        print(
            f"[select] seed={seed} step={int(selected['checkpoint_step'])} "
            f"val_feasible={selected_rows[-1]['val_EES_feasible']} "
            f"val_cost={float(selected['val_cost_plus_penalty']):.6f}",
            flush=True,
        )

    selected_df = pd.DataFrame(selected_rows)

    selected_test_rows: list[dict[str, object]] = []
    for seed in seeds:
        selected_row = selected_series_by_seed[seed]
        result_dir = run_selected_test(
            seed=seed,
            selected_row=selected_row,
            device=device,
            skip_existing=bool(args.skip_existing_tests),
        )
        selected_summary = summarize_result_dir(
            method="TD3-CUDA-feasible-selected",
            seed=seed,
            result_dir=result_dir,
            model_path=feasible_model_path(seed),
            selected_row=selected_row,
        )
        selected_test_rows.append(selected_summary)

    selected_summary_df = pd.DataFrame(selected_test_rows, columns=SEED_SUMMARY_COLUMNS)
    for index, row in selected_summary_df.iterrows():
        selected_df.loc[selected_df["seed"] == int(row["seed"]), "test_EES_feasible"] = row[
            "test_EES_feasible"
        ]
        selected_df.loc[selected_df["seed"] == int(row["seed"]), "test_terminal_shortage"] = row[
            "test_terminal_shortage"
        ]
        selected_df.loc[selected_df["seed"] == int(row["seed"]), "test_cost_plus_penalty"] = row[
            "test_cost_plus_penalty"
        ]
        selected_df.loc[selected_df["seed"] == int(row["seed"]), "test_result_dir"] = row[
            "result_dir"
        ]

    original_rows = [
        summarize_result_dir(
            method="TD3-original",
            seed=seed,
            result_dir=original_result_dir(seed),
            model_path=Path(f"models/td3_yearly_single_seed_{seed}/best/best_model.zip"),
        )
        for seed in seeds
    ]
    reward_best_rows = [
        summarize_result_dir(
            method="TD3-CUDA-reward-best",
            seed=seed,
            result_dir=reward_best_result_dir(seed),
            model_path=Path(
                f"models/td3_yearly_single_cuda_seed_{seed}/best/best_model.zip"
            ),
        )
        for seed in seeds
    ]
    seed_summary_df = pd.DataFrame(
        [*original_rows, *reward_best_rows, *selected_test_rows],
        columns=SEED_SUMMARY_COLUMNS,
    )
    algorithm_df = build_algorithm_summary(seed_summary_df)
    vs_original_df = build_vs_original_summary(algorithm_df)
    candidate, reasons, conditions = candidate_status(seed_summary_df, algorithm_df)

    selected_path = output_dir / "selected_cuda_checkpoints.csv"
    seed_summary_path = output_dir / "td3_cuda_feasible_selected_seed_summary.csv"
    algorithm_path = output_dir / "td3_cuda_feasible_selected_algorithm_summary.csv"
    vs_original_path = output_dir / "td3_cuda_feasible_selected_vs_original_summary.csv"

    selected_df.to_csv(selected_path, index=False, encoding="utf-8-sig")
    selected_summary_df.to_csv(seed_summary_path, index=False, encoding="utf-8-sig")
    algorithm_df.to_csv(algorithm_path, index=False, encoding="utf-8-sig")
    vs_original_df.to_csv(vs_original_path, index=False, encoding="utf-8-sig")
    write_report(
        output_dir=output_dir,
        selected_df=selected_df,
        seed_summary_df=seed_summary_df,
        algorithm_df=algorithm_df,
        vs_original_df=vs_original_df,
        candidate=candidate,
        reasons=reasons,
        conditions=conditions,
    )

    print(f"[output] selected checkpoints: {selected_path}")
    print(f"[output] selected seed summary: {seed_summary_path}")
    print(f"[output] algorithm summary: {algorithm_path}")
    print(f"[output] vs original summary: {vs_original_path}")
    print(f"[output] report: {output_dir / 'selected_cuda_checkpoint_report.md'}")
    print(selected_df.to_string(index=False))


if __name__ == "__main__":
    main()
