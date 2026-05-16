from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import DDPG, PPO

from ies_config import IESConfig
from park_ies_env import ParkIESEnv
from yearly_case_env import YearlyEVProvider
from yearly_csv_loader import YearlyCSVDataLoader


DEFAULT_METHODS = ["DDPG", "PPO"]
DEFAULT_SEEDS = [42, 2024, 2025]
OUTPUT_DIR = Path("results/ddpg_ppo_feasible_checkpoint_selection")
YEARLY_CSV_PATH = Path(r"D:\wangye\chengxu\3.30\yearly_data_sci.csv")
YEARLY_EV_PATH = Path(r"D:\wangye\chengxu\3.30\ev_data_s123_bilevel.npy")
VAL_DAYS_PER_MONTH = 2
REWARD_SCALE = 1e-5
SELECTION_RULE = "validation feasibility-first"
CHECKPOINT_PATTERN = re.compile(r"_(\d+)_steps\.zip$")

METHOD_SPECS = {
    "DDPG": {
        "class": DDPG,
        "checkpoint_dir": "models/ddpg_yearly_single_seed_{seed}/checkpoints",
        "checkpoint_glob": "ddpg_yearly_single_*_steps.zip",
        "result_dir": "results/ddpg_feasible_selected_test_seed_{seed}",
    },
    "PPO": {
        "class": PPO,
        "checkpoint_dir": "models/ppo_sb3_direct_seed_{seed}/checkpoints",
        "checkpoint_glob": "ppo_sb3_direct_*_steps.zip",
        "result_dir": "results/ppo_feasible_selected_test_seed_{seed}",
    },
}

SELECTED_COLUMNS = [
    "method",
    "seed",
    "checkpoint_found",
    "selected_checkpoint_step",
    "selected_checkpoint_path",
    "selected_reason",
    "val_EES_feasible",
    "val_terminal_ees_feasible_days",
    "val_terminal_ees_feasible_ratio",
    "val_terminal_shortage",
    "val_cost_plus_penalty",
    "val_mean_total_system_cost",
    "val_mean_total_penalties",
    "val_ev_departure_shortage_kwh",
    "val_unmet_e",
    "val_unmet_h",
    "val_unmet_c",
    "val_reward",
    "test_EES_feasible",
    "test_terminal_shortage",
    "test_cost_plus_penalty",
    "test_EV_departure_shortage",
    "test_unmet_e",
    "test_unmet_h",
    "test_unmet_c",
    "test_result_dir",
    "warning",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select DDPG/PPO checkpoints by validation feasibility and test selected checkpoints."
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--skip-existing-tests",
        action="store_true",
        help="Reuse existing selected test directories when all required files exist.",
    )
    return parser.parse_args()


def normalize_methods(methods: list[str]) -> list[str]:
    normalized: list[str] = []
    for method in methods:
        key = method.upper()
        if key not in METHOD_SPECS:
            raise ValueError(f"Unknown method {method!r}. Valid methods: {', '.join(METHOD_SPECS)}")
        normalized.append(key)
    return normalized


def validate_device(device: str) -> str:
    cleaned = str(device).strip() or "auto"
    return cleaned


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse checkpoint step from {path}")
    return int(match.group(1))


def checkpoint_paths(method: str, seed: int) -> list[Path]:
    spec = METHOD_SPECS[method]
    directory = Path(str(spec["checkpoint_dir"]).format(seed=int(seed)))
    if not directory.exists():
        return []
    return sorted(directory.glob(str(spec["checkpoint_glob"])), key=checkpoint_step)


def result_dir_for(method: str, seed: int) -> Path:
    return Path(str(METHOD_SPECS[method]["result_dir"]).format(seed=int(seed)))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def rollout_one_day(model: Any, env: ParkIESEnv) -> tuple[list[dict[str, Any]], float]:
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


def build_daily_row(
    *,
    case_index: int,
    case: Any,
    infos: list[dict[str, Any]],
    total_reward: float,
    cfg: IESConfig,
) -> dict[str, Any]:
    final_info = infos[-1] if infos else {}
    total_system_cost = float(sum(to_float(item.get("system_cost")) for item in infos))
    total_penalties = float(sum(to_float(item.get("penalty_cost")) for item in infos))
    final_ees_soc = to_float(final_info.get("final_ees_soc", final_info.get("ees_soc", np.nan)), np.nan)
    terminal_required_soc = to_float(final_info.get("terminal_ees_required_soc", 0.0))
    terminal_shortage = to_float(final_info.get("episode_terminal_ees_shortage_kwh", 0.0))
    terminal_penalty = to_float(final_info.get("episode_penalty_terminal_ees_soc", 0.0))
    feasible = bool(final_info.get("ees_terminal_soc_feasible", True))
    return {
        "case_index": int(case_index),
        "month": int(case.month),
        "day_of_year": int(case.day_of_year),
        "season": str(case.season),
        "split": str(case.set_name),
        "total_reward": float(total_reward),
        "total_reward_raw": float(sum(to_float(item.get("reward_raw")) for item in infos)),
        "total_system_cost": total_system_cost,
        "total_penalties": total_penalties,
        "total_cost_plus_penalty": total_system_cost + total_penalties,
        "total_guide_reward": float(sum(to_float(item.get("guide_reward")) for item in infos)),
        "total_cost_grid": float(sum(to_float(item.get("cost_grid")) for item in infos)),
        "total_cost_gas": float(sum(to_float(item.get("cost_gas")) for item in infos)),
        "total_cost_deg": float(sum(to_float(item.get("cost_deg")) for item in infos)),
        "total_cost_om": float(sum(to_float(item.get("cost_om")) for item in infos)),
        "total_unmet_e": float(sum(to_float(item.get("unmet_e")) for item in infos)),
        "total_unmet_h": float(sum(to_float(item.get("unmet_h")) for item in infos)),
        "total_unmet_c": float(sum(to_float(item.get("unmet_c")) for item in infos)),
        "total_surplus_e": float(sum(to_float(item.get("surplus_e")) for item in infos)),
        "total_surplus_h": float(sum(to_float(item.get("surplus_h")) for item in infos)),
        "total_surplus_c": float(sum(to_float(item.get("surplus_c")) for item in infos)),
        "total_grid_overflow": float(sum(to_float(item.get("grid_overflow")) for item in infos)),
        "total_penalty_unserved_e": float(sum(to_float(item.get("penalty_unserved_e")) for item in infos)),
        "total_penalty_unserved_h": float(sum(to_float(item.get("penalty_unserved_h")) for item in infos)),
        "total_penalty_unserved_c": float(sum(to_float(item.get("penalty_unserved_c")) for item in infos)),
        "total_penalty_depart_energy": float(sum(to_float(item.get("penalty_depart_energy")) for item in infos)),
        "total_penalty_depart_risk": float(sum(to_float(item.get("penalty_depart_risk")) for item in infos)),
        "total_penalty_export_e": float(sum(to_float(item.get("penalty_export_e")) for item in infos)),
        "total_penalty_ev_export_guard": float(sum(to_float(item.get("penalty_ev_export_guard")) for item in infos)),
        "total_penalty_terminal_ees_soc": terminal_penalty,
        "total_depart_energy_shortage_kwh": float(
            sum(to_float(item.get("depart_energy_shortage_kwh")) for item in infos)
        ),
        "total_depart_risk_energy_kwh": float(
            sum(to_float(item.get("depart_risk_energy_kwh")) for item in infos)
        ),
        "total_storage_peak_shaved_kwh": float(
            sum(to_float(item.get("storage_peak_shaved_kwh")) for item in infos)
        ),
        "total_storage_charge_rewarded_kwh": float(
            sum(to_float(item.get("storage_charge_rewarded_kwh")) for item in infos)
        ),
        "total_ees_charge_rewarded_kwh": float(
            sum(to_float(item.get("ees_charge_rewarded_kwh")) for item in infos)
        ),
        "total_ev_flex_target_charge_kwh": float(
            sum(to_float(item.get("ev_flex_target_charge_kwh")) for item in infos)
        ),
        "total_ev_buffer_charge_kwh": float(
            sum(to_float(item.get("ev_buffer_charge_kwh")) for item in infos)
        ),
        "total_gt_export_clip": float(sum(to_float(item.get("p_gt_export_clip")) for item in infos)),
        "total_gt_export_clip_steps": int(
            sum(1 for item in infos if to_float(item.get("p_gt_export_clip")) > 1e-8)
        ),
        "total_gt_safe_infeasible_steps": int(
            sum(1 for item in infos if not bool(item.get("gt_safe_feasible", True)))
        ),
        "total_grid_buy_kwh": float(sum(to_float(item.get("p_grid_buy")) for item in infos) * cfg.dt),
        "total_grid_sell_kwh": float(sum(to_float(item.get("p_grid_sell")) for item in infos) * cfg.dt),
        "avg_p_gt_kw": float(sum(to_float(item.get("p_gt")) for item in infos) / max(len(infos), 1)),
        "avg_p_ev_ch_kw": float(sum(to_float(item.get("p_ev_ch")) for item in infos) / max(len(infos), 1)),
        "avg_p_ev_dis_kw": float(sum(to_float(item.get("p_ev_dis")) for item in infos) / max(len(infos), 1)),
        "avg_p_ees_ch_kw": float(sum(to_float(item.get("p_ees_ch")) for item in infos) / max(len(infos), 1)),
        "avg_p_ees_dis_kw": float(sum(to_float(item.get("p_ees_dis")) for item in infos) / max(len(infos), 1)),
        "ees_soc_episode_init": to_float(infos[0].get("ees_soc_episode_init", 0.0)) if infos else 0.0,
        "final_ees_soc": final_ees_soc,
        "ees_soc_init": to_float(final_info.get("ees_soc_init", final_info.get("ees_soc_episode_init", 0.0))),
        "terminal_ees_required_soc": terminal_required_soc,
        "terminal_ees_shortage_kwh": terminal_shortage,
        "ees_terminal_soc_feasible": feasible,
        "final_gt_power": to_float(final_info.get("p_gt", 0.0)),
    }


def build_step_rows(case_index: int, case: Any, infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step, info in enumerate(infos):
        row: dict[str, Any] = {
            "case_index": int(case_index),
            "month": int(case.month),
            "day_of_year": int(case.day_of_year),
            "season": str(case.season),
            "split": str(case.set_name),
            "time_step": int(info.get("time_step", step)),
        }
        for key, value in info.items():
            if key in row:
                continue
            if isinstance(value, np.generic):
                row[key] = value.item()
            else:
                row[key] = value
        rows.append(row)
    return rows


def evaluate_model(
    *,
    method: str,
    model_path: Path,
    split: str,
    loader: YearlyCSVDataLoader,
    ev_provider: YearlyEVProvider,
    device: str,
    collect_steps: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_cls = METHOD_SPECS[method]["class"]
    model = model_cls.load(str(model_path), device=device)
    cfg = IESConfig(episode_length=24, dt=1.0, reward_scale=REWARD_SCALE)
    daily_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(loader.get_cases(split)):
        env = ParkIESEnv(cfg=cfg, ts_data=case.ts_data, ev_data=ev_provider(case))
        infos, total_reward = rollout_one_day(model, env)
        env.close()
        daily_rows.append(
            build_daily_row(
                case_index=case_index,
                case=case,
                infos=infos,
                total_reward=total_reward,
                cfg=cfg,
            )
        )
        if collect_steps:
            step_rows.extend(build_step_rows(case_index, case, infos))
    return pd.DataFrame(daily_rows), pd.DataFrame(step_rows)


def metrics_from_daily(daily: pd.DataFrame) -> dict[str, float | int]:
    if daily.empty:
        return {
            "n_days": 0,
            "terminal_ees_feasible_days": 0,
            "terminal_ees_feasible_ratio": math.nan,
            "terminal_ees_shortage_kwh": math.nan,
            "mean_total_cost_plus_penalty": math.nan,
            "mean_total_system_cost": math.nan,
            "mean_total_penalties": math.nan,
            "ev_departure_shortage_kwh": math.nan,
            "unmet_e": math.nan,
            "unmet_h": math.nan,
            "unmet_c": math.nan,
            "reward": math.nan,
        }
    feasible = daily["ees_terminal_soc_feasible"].astype(bool)
    n_days = int(len(daily))
    feasible_days = int(feasible.sum())
    return {
        "n_days": n_days,
        "terminal_ees_feasible_days": feasible_days,
        "terminal_ees_feasible_ratio": float(feasible_days / n_days),
        "terminal_ees_shortage_kwh": float(pd.to_numeric(daily["terminal_ees_shortage_kwh"], errors="coerce").fillna(0.0).sum()),
        "mean_total_cost_plus_penalty": float(pd.to_numeric(daily["total_cost_plus_penalty"], errors="coerce").mean()),
        "mean_total_system_cost": float(pd.to_numeric(daily["total_system_cost"], errors="coerce").mean()),
        "mean_total_penalties": float(pd.to_numeric(daily["total_penalties"], errors="coerce").mean()),
        "ev_departure_shortage_kwh": float(pd.to_numeric(daily["total_depart_energy_shortage_kwh"], errors="coerce").fillna(0.0).sum()),
        "unmet_e": float(pd.to_numeric(daily["total_unmet_e"], errors="coerce").fillna(0.0).sum()),
        "unmet_h": float(pd.to_numeric(daily["total_unmet_h"], errors="coerce").fillna(0.0).sum()),
        "unmet_c": float(pd.to_numeric(daily["total_unmet_c"], errors="coerce").fillna(0.0).sum()),
        "reward": float(pd.to_numeric(daily["total_reward"], errors="coerce").mean()),
    }


def scan_checkpoints(
    *,
    method: str,
    seed: int,
    loader: YearlyCSVDataLoader,
    ev_provider: YearlyEVProvider,
    device: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    paths = checkpoint_paths(method, seed)
    for checkpoint in paths:
        step = checkpoint_step(checkpoint)
        daily, _ = evaluate_model(
            method=method,
            model_path=checkpoint,
            split="val",
            loader=loader,
            ev_provider=ev_provider,
            device=device,
            collect_steps=False,
        )
        metrics = metrics_from_daily(daily)
        rows.append(
            {
                "method": method,
                "seed": int(seed),
                "checkpoint_step": int(step),
                "checkpoint_path": str(checkpoint),
                "val_n_days": int(metrics["n_days"]),
                "val_terminal_ees_feasible_days": int(metrics["terminal_ees_feasible_days"]),
                "val_terminal_ees_feasible_ratio": float(metrics["terminal_ees_feasible_ratio"]),
                "val_terminal_ees_shortage_kwh": float(metrics["terminal_ees_shortage_kwh"]),
                "val_mean_total_cost_plus_penalty": float(metrics["mean_total_cost_plus_penalty"]),
                "val_mean_total_system_cost": float(metrics["mean_total_system_cost"]),
                "val_mean_total_penalties": float(metrics["mean_total_penalties"]),
                "val_ev_departure_shortage_kwh": float(metrics["ev_departure_shortage_kwh"]),
                "val_unmet_e": float(metrics["unmet_e"]),
                "val_unmet_h": float(metrics["unmet_h"]),
                "val_unmet_c": float(metrics["unmet_c"]),
                "val_reward": float(metrics["reward"]),
            }
        )
        print(
            f"[val] {method} seed={seed} step={step} "
            f"unmet=({metrics['unmet_e']:.6g},{metrics['unmet_h']:.6g},{metrics['unmet_c']:.6g}) "
            f"depart={metrics['ev_departure_shortage_kwh']:.6g} "
            f"EES={metrics['terminal_ees_feasible_days']}/{metrics['n_days']} "
            f"shortage={metrics['terminal_ees_shortage_kwh']:.6f} "
            f"cost={metrics['mean_total_cost_plus_penalty']:.6f} "
            f"reward={metrics['reward']:.9f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def select_checkpoint(validation_df: pd.DataFrame) -> pd.Series:
    n_val = int(validation_df["val_n_days"].max())
    full_feasible = validation_df[
        validation_df["val_terminal_ees_feasible_days"].astype(int) >= n_val
    ].copy()
    candidates = full_feasible if not full_feasible.empty else validation_df.copy()
    candidates["val_unmet_total"] = (
        candidates["val_unmet_e"].abs()
        + candidates["val_unmet_h"].abs()
        + candidates["val_unmet_c"].abs()
    )
    ordered = candidates.sort_values(
        by=[
            "val_unmet_e",
            "val_unmet_h",
            "val_unmet_c",
            "val_ev_departure_shortage_kwh",
            "val_terminal_ees_feasible_days",
            "val_terminal_ees_shortage_kwh",
            "val_mean_total_cost_plus_penalty",
            "val_reward",
        ],
        ascending=[True, True, True, True, False, True, True, False],
        kind="mergesort",
    )
    return ordered.iloc[0]


def selection_reason(selected: pd.Series, validation_df: pd.DataFrame) -> str:
    n_val = int(validation_df["val_n_days"].max())
    full_count = int((validation_df["val_terminal_ees_feasible_days"].astype(int) >= n_val).sum())
    if full_count > 0:
        return (
            f"Selected from {full_count} validation-full-EES-feasible checkpoints "
            f"({n_val}/{n_val}); sorted by unmet load, EV departure shortage, EES shortage, "
            "cost+penalty, then reward."
        )
    max_feasible = int(validation_df["val_terminal_ees_feasible_days"].max())
    return (
        f"No validation-full-EES-feasible checkpoint; selected from checkpoints with up to "
        f"{max_feasible}/{n_val} feasible days by unmet load, EV departure shortage, "
        "feasible days, shortage, cost+penalty, then reward."
    )


def write_test_outputs(
    *,
    method: str,
    seed: int,
    checkpoint_path: Path,
    result_dir: Path,
    selected: pd.Series,
    loader: YearlyCSVDataLoader,
    ev_provider: YearlyEVProvider,
    device: str,
    output_dir: Path,
    skip_existing: bool,
) -> dict[str, Any]:
    required = [
        result_dir / "daily_summary.csv",
        result_dir / "timeseries_detail.csv",
        result_dir / "runtime_summary.json",
        result_dir / "test_export.xlsx",
    ]
    if skip_existing and all(path.exists() for path in required):
        print(f"[skip] {method} seed={seed} selected test exists: {result_dir}", flush=True)
        daily = pd.read_csv(result_dir / "daily_summary.csv")
    else:
        result_dir.mkdir(parents=True, exist_ok=True)
        started = perf_counter()
        daily, steps = evaluate_model(
            method=method,
            model_path=checkpoint_path,
            split="test",
            loader=loader,
            ev_provider=ev_provider,
            device=device,
            collect_steps=True,
        )
        duration = max(perf_counter() - started, 0.0)
        daily.to_csv(result_dir / "daily_summary.csv", index=False, encoding="utf-8-sig")
        steps.to_csv(result_dir / "timeseries_detail.csv", index=False, encoding="utf-8-sig")
        metadata = {
            "method": f"{method}-feasible-selected",
            "seed": int(seed),
            "selected_checkpoint_path": str(checkpoint_path),
            "selected_checkpoint_step": int(selected["checkpoint_step"]),
            "selection_rule": SELECTION_RULE,
            "deterministic": True,
            "device": str(device),
            "result_dir": str(result_dir),
            "n_days": int(len(daily)),
            "n_steps": int(len(steps)),
            "test_duration_seconds": round(float(duration), 6),
            "time_per_day_seconds": round(float(duration / max(len(daily), 1)), 6),
            "time_per_step_seconds": round(float(duration / max(len(steps), 1)), 6),
            "val_terminal_ees_feasible_days": int(selected["val_terminal_ees_feasible_days"]),
            "val_terminal_ees_shortage_kwh": float(selected["val_terminal_ees_shortage_kwh"]),
            "val_mean_total_cost_plus_penalty": float(selected["val_mean_total_cost_plus_penalty"]),
            "val_reward": float(selected["val_reward"]),
        }
        (result_dir / "runtime_summary.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with pd.ExcelWriter(result_dir / "test_export.xlsx", engine="openpyxl") as writer:
            daily.to_excel(writer, sheet_name="daily_summary", index=False)
            steps.to_excel(writer, sheet_name="timeseries_detail", index=False)
            pd.DataFrame([metadata]).to_excel(writer, sheet_name="runtime_summary", index=False)
            pd.DataFrame([selected.to_dict()]).to_excel(writer, sheet_name="selection", index=False)
    metrics = metrics_from_daily(daily)
    return {
        "test_EES_feasible": f"{int(metrics['terminal_ees_feasible_days'])}/{int(metrics['n_days'])}",
        "test_terminal_shortage": float(metrics["terminal_ees_shortage_kwh"]),
        "test_cost_plus_penalty": float(metrics["mean_total_cost_plus_penalty"]),
        "test_EV_departure_shortage": float(metrics["ev_departure_shortage_kwh"]),
        "test_unmet_e": float(metrics["unmet_e"]),
        "test_unmet_h": float(metrics["unmet_h"]),
        "test_unmet_c": float(metrics["unmet_c"]),
        "test_result_dir": str(result_dir),
    }


def main() -> None:
    args = parse_args()
    methods = normalize_methods(args.methods)
    seeds = [int(seed) for seed in args.seeds]
    device = validate_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(42)

    loader = YearlyCSVDataLoader(YEARLY_CSV_PATH, val_days_per_month=VAL_DAYS_PER_MONTH)
    loader.load()
    ev_provider = YearlyEVProvider(YEARLY_EV_PATH)

    selected_rows: list[dict[str, Any]] = []
    for method in methods:
        for seed in seeds:
            paths = checkpoint_paths(method, seed)
            if not paths:
                warning = f"missing checkpoints for {method} seed={seed}"
                print(f"WARNING: {warning}", flush=True)
                selected_rows.append(
                    {
                        "method": method,
                        "seed": int(seed),
                        "checkpoint_found": False,
                        "warning": warning,
                    }
                )
                continue

            validation_df = scan_checkpoints(
                method=method,
                seed=seed,
                loader=loader,
                ev_provider=ev_provider,
                device=device,
            )
            selected = select_checkpoint(validation_df).copy()
            selected["selected_reason"] = selection_reason(selected, validation_df)
            checkpoint_path = Path(str(selected["checkpoint_path"]))
            result_dir = result_dir_for(method, seed)
            test_metrics = write_test_outputs(
                method=method,
                seed=seed,
                checkpoint_path=checkpoint_path,
                result_dir=result_dir,
                selected=selected,
                loader=loader,
                ev_provider=ev_provider,
                device=device,
                output_dir=output_dir,
                skip_existing=bool(args.skip_existing_tests),
            )
            row = {
                "method": method,
                "seed": int(seed),
                "checkpoint_found": True,
                "selected_checkpoint_step": int(selected["checkpoint_step"]),
                "selected_checkpoint_path": str(checkpoint_path),
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
                "val_cost_plus_penalty": float(selected["val_mean_total_cost_plus_penalty"]),
                "val_mean_total_system_cost": float(selected["val_mean_total_system_cost"]),
                "val_mean_total_penalties": float(selected["val_mean_total_penalties"]),
                "val_ev_departure_shortage_kwh": float(
                    selected["val_ev_departure_shortage_kwh"]
                ),
                "val_unmet_e": float(selected["val_unmet_e"]),
                "val_unmet_h": float(selected["val_unmet_h"]),
                "val_unmet_c": float(selected["val_unmet_c"]),
                "val_reward": float(selected["val_reward"]),
                "warning": "",
                **test_metrics,
            }
            selected_rows.append(row)
            print(
                f"[select] {method} seed={seed} step={row['selected_checkpoint_step']} "
                f"val={row['val_EES_feasible']} test={row['test_EES_feasible']} "
                f"test_cost={row['test_cost_plus_penalty']:.6f}",
                flush=True,
            )

    selected_df = pd.DataFrame(selected_rows)
    for column in SELECTED_COLUMNS:
        if column not in selected_df.columns:
            selected_df[column] = ""
    selected_df = selected_df[SELECTED_COLUMNS]
    selected_path = output_dir / "selected_ddpg_ppo_checkpoints.csv"
    selected_df.to_csv(selected_path, index=False, encoding="utf-8-sig")
    print(f"[output] selected checkpoints: {selected_path}")


if __name__ == "__main__":
    main()
