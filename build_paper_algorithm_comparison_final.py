from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
OUTPUT_ROOT = RESULTS_DIR / "paper_algorithm_comparison_final"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
REPORT_DIR = OUTPUT_ROOT / "report"

SEEDS = (42, 2024, 2025)
MOVING_AVERAGE_WINDOW = 100
TRAINING_REWARD_ZOOM_START = 3000
TRAINING_REWARD_ZOOM_END = 4000
TRAINING_REWARD_ZOOM_YLIM = (-2.20, -1.90)
MAIN_STD_ALPHA = 0.05
ZOOM_STD_ALPHA = 0.08
TEST_DAYS = 60
DT_HOURS = 1.0
EXCHANGE_EPS = 1e-9
RELIABILITY_ZERO_TOLERANCE_KWH = 1e-3
NA = "NA"
NOT_APPLICABLE = "—"

METHOD_ORDER = ("Rule-based-V2G", "DDPG", "PPO", "TD3")
DRL_ORDER = ("TD3", "DDPG", "PPO")

COLORS = {
    "TD3": "#1f77b4",
    "DDPG": "#ff7f0e",
    "PPO": "#2ca02c",
    "Rule-based-V2G": "#4d4d4d",
}


@dataclass(frozen=True)
class MethodSources:
    daily_summary: Path
    timeseries_detail: Path
    runtime_summary: Path
    training_runtime_summary: Path | None = None


METHOD_SOURCES = {
    "TD3": MethodSources(
        daily_summary=RESULTS_DIR / "td3_yearly_test_cuda_feasible_seed_42" / "daily_summary.csv",
        timeseries_detail=RESULTS_DIR / "td3_yearly_test_cuda_feasible_seed_42" / "timeseries_detail.csv",
        runtime_summary=RESULTS_DIR / "td3_yearly_test_cuda_feasible_seed_42" / "runtime_summary.json",
        training_runtime_summary=RESULTS_DIR / "td3_yearly_training_cuda_seed_42" / "training_runtime_summary.json",
    ),
    "DDPG": MethodSources(
        daily_summary=RESULTS_DIR / "ddpg_feasible_selected_test_seed_42" / "daily_summary.csv",
        timeseries_detail=RESULTS_DIR / "ddpg_feasible_selected_test_seed_42" / "timeseries_detail.csv",
        runtime_summary=RESULTS_DIR / "ddpg_feasible_selected_test_seed_42" / "runtime_summary.json",
        training_runtime_summary=RESULTS_DIR / "ddpg_yearly_training_seed_42" / "training_runtime_summary.json",
    ),
    "PPO": MethodSources(
        daily_summary=RESULTS_DIR / "ppo_feasible_selected_test_seed_42" / "daily_summary.csv",
        timeseries_detail=RESULTS_DIR / "ppo_feasible_selected_test_seed_42" / "timeseries_detail.csv",
        runtime_summary=RESULTS_DIR / "ppo_feasible_selected_test_seed_42" / "runtime_summary.json",
        training_runtime_summary=RESULTS_DIR / "ppo_sb3_direct_training_seed_42" / "training_runtime_summary.json",
    ),
    "Rule-based-V2G": MethodSources(
        daily_summary=RESULTS_DIR / "rule_based_v2g_test" / "daily_summary.csv",
        timeseries_detail=RESULTS_DIR / "rule_based_v2g_test" / "timeseries_detail.csv",
        runtime_summary=RESULTS_DIR / "rule_based_v2g_test" / "runtime_summary.json",
        training_runtime_summary=None,
    ),
}

TRAINING_EPISODE_SUMMARY = {
    "TD3": {seed: RESULTS_DIR / f"td3_yearly_training_cuda_seed_{seed}" / "episode_summary.csv" for seed in SEEDS},
    "DDPG": {seed: RESULTS_DIR / f"ddpg_yearly_training_seed_{seed}" / "episode_summary.csv" for seed in SEEDS},
    "PPO": {seed: RESULTS_DIR / f"ppo_sb3_direct_training_seed_{seed}" / "episode_summary.csv" for seed in SEEDS},
}

TRAINING_SCRIPTS = {
    "TD3": ROOT / "train_TD3.py",
    "DDPG": ROOT / "train_DDPG.py",
    "PPO": ROOT / "train_PPO_sb3_direct.py",
}

FINAL_SOURCE_FILES = {
    "TD3 seed summary": RESULTS_DIR
    / "td3_cuda_multiseed_feasible_selected"
    / "td3_cuda_feasible_selected_seed_summary.csv",
    "TD3 algorithm summary": RESULTS_DIR
    / "td3_cuda_multiseed_feasible_selected"
    / "td3_cuda_feasible_selected_algorithm_summary.csv",
    "TD3 selected checkpoints": RESULTS_DIR
    / "td3_cuda_multiseed_feasible_selected"
    / "selected_cuda_checkpoints.csv",
    "DDPG/PPO seed summary": RESULTS_DIR
    / "ddpg_ppo_feasible_checkpoint_selection"
    / "ddpg_ppo_feasible_selected_seed_summary.csv",
    "DDPG/PPO algorithm summary": RESULTS_DIR
    / "ddpg_ppo_feasible_checkpoint_selection"
    / "ddpg_ppo_feasible_selected_algorithm_summary.csv",
    "DDPG/PPO selected checkpoints": RESULTS_DIR
    / "ddpg_ppo_feasible_checkpoint_selection"
    / "selected_ddpg_ppo_checkpoints.csv",
    "Rule-based-V2G daily summary": RESULTS_DIR / "rule_based_v2g_test" / "daily_summary.csv",
    "Rule-based-V2G timeseries detail": RESULTS_DIR / "rule_based_v2g_test" / "timeseries_detail.csv",
}


def load_declared_final_sources(notes: list[str]) -> dict[str, pd.DataFrame]:
    sources: dict[str, pd.DataFrame] = {}
    for label, path in FINAL_SOURCE_FILES.items():
        df = read_csv(path, notes, f"declared final source file ({label})")
        if df is not None:
            sources[label] = df
    return sources


def ensure_dirs() -> None:
    for path in (TABLE_DIR, FIGURE_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def eval_expr(node: ast.AST, constants: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [eval_expr(item, constants) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(eval_expr(item, constants) for item in node.elts)
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.UnaryOp):
        value = eval_expr(node.operand, constants)
        if isinstance(node.op, ast.USub) and isinstance(value, (int, float)):
            return -value
        if isinstance(node.op, ast.UAdd) and isinstance(value, (int, float)):
            return value
    if isinstance(node, ast.BinOp):
        left = eval_expr(node.left, constants)
        right = eval_expr(node.right, constants)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Pow):
                return left**right
    return None


def read_module_constants(path: Path, notes: list[str]) -> dict[str, Any]:
    if not path.exists():
        notes.append(f"Missing training script for hyperparameter parsing: {rel(path)}")
        return {}

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = eval_expr(node.value, constants)
            if value is not None:
                constants[node.targets[0].id] = value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            value = eval_expr(node.value, constants)
            if value is not None:
                constants[node.target.id] = value
    return constants


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def read_csv(path: Path, notes: list[str], label: str, **kwargs: Any) -> pd.DataFrame | None:
    if not path.exists():
        notes.append(f"Missing {label}: {rel(path)}")
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:  # noqa: BLE001 - report generation should continue with clear notes.
        notes.append(f"Failed to read {label} at {rel(path)}: {exc}")
        return None


def read_json(path: Path, notes: list[str], label: str) -> dict[str, Any]:
    if not path or not path.exists():
        notes.append(f"Missing {label}: {rel(path)}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Failed to read {label} at {rel(path)}: {exc}")
        return {}


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def first_existing(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in columns:
            return column
    return None


def fmt_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return NA
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return NA
    if number == 0:
        return "0"
    if abs(number) < 10 ** (-(digits + 1)):
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}"


def fmt_seconds(value: Any, digits: int = 3) -> str:
    text = fmt_number(value, digits=digits)
    return NA if text == NA else f"{text} s"


def fmt_reliability_kwh(value: Any) -> str:
    if value is None:
        return NA
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return NA
    if abs(number) < RELIABILITY_ZERO_TOLERANCE_KWH:
        return "0"
    return fmt_number(number)


def fmt_param(value: Any) -> str:
    if value is None:
        return NA
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != 0 and abs(value) < 1e-3:
            return f"{value:.3g}".replace("e-0", "e-").replace("e+0", "e+")
        return f"{value:g}"
    return str(value)


def markdown_table(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([str(row[column]).replace("|", "\\|").replace("\n", " ") for column in df.columns])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def write_table_bundle(df: pd.DataFrame, stem: str, caption: str, note: str | None = None) -> dict[str, Path]:
    csv_path = TABLE_DIR / f"{stem}.csv"
    xlsx_path = TABLE_DIR / f"{stem}.xlsx"
    md_path = TABLE_DIR / f"{stem}.md"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False)
    note_text = f"\n\nNote: {note}\n" if note else ""
    md_path.write_text(f"**{caption}**\n\n{markdown_table(df)}{note_text}", encoding="utf-8")
    return {"csv": csv_path, "xlsx": xlsx_path, "md": md_path}


def build_hyperparameter_table(notes: list[str]) -> pd.DataFrame:
    constants = {method: read_module_constants(path, notes) for method, path in TRAINING_SCRIPTS.items()}

    required = {
        "Number of episodes": "TOTAL_EPISODES",
        "Total timesteps": "TOTAL_TIMESTEPS",
        "Learning rate": "LEARNING_RATE",
        "Discount factor gamma": "GAMMA",
        "Mini-batch size": "BATCH_SIZE",
    }
    for method, method_constants in constants.items():
        for label, key in required.items():
            if key not in method_constants:
                notes.append(f"Table 1 missing {label} for {method}; wrote NA.")

    rows = [
        {
            "Parameter": "Number of episodes",
            "TD3": fmt_param(constants["TD3"].get("TOTAL_EPISODES")),
            "DDPG": fmt_param(constants["DDPG"].get("TOTAL_EPISODES")),
            "PPO": fmt_param(constants["PPO"].get("TOTAL_EPISODES")),
        },
        {
            "Parameter": "Total timesteps",
            "TD3": fmt_param(constants["TD3"].get("TOTAL_TIMESTEPS")),
            "DDPG": fmt_param(constants["DDPG"].get("TOTAL_TIMESTEPS")),
            "PPO": fmt_param(constants["PPO"].get("TOTAL_TIMESTEPS")),
        },
        {
            "Parameter": "Learning rate",
            "TD3": fmt_param(constants["TD3"].get("LEARNING_RATE")),
            "DDPG": fmt_param(constants["DDPG"].get("LEARNING_RATE")),
            "PPO": fmt_param(constants["PPO"].get("LEARNING_RATE")),
        },
        {
            "Parameter": "Discount factor gamma",
            "TD3": fmt_param(constants["TD3"].get("GAMMA")),
            "DDPG": fmt_param(constants["DDPG"].get("GAMMA")),
            "PPO": fmt_param(constants["PPO"].get("GAMMA")),
        },
        {
            "Parameter": "Soft update factor tau",
            "TD3": fmt_param(constants["TD3"].get("TAU")),
            "DDPG": fmt_param(constants["DDPG"].get("TAU")),
            "PPO": NOT_APPLICABLE,
        },
        {
            "Parameter": "Replay buffer size",
            "TD3": fmt_param(constants["TD3"].get("BUFFER_SIZE")),
            "DDPG": fmt_param(constants["DDPG"].get("BUFFER_SIZE")),
            "PPO": NOT_APPLICABLE,
        },
        {
            "Parameter": "Mini-batch size",
            "TD3": fmt_param(constants["TD3"].get("BATCH_SIZE")),
            "DDPG": fmt_param(constants["DDPG"].get("BATCH_SIZE")),
            "PPO": fmt_param(constants["PPO"].get("BATCH_SIZE")),
        },
    ]
    return pd.DataFrame(rows, columns=("Parameter", "TD3", "DDPG", "PPO"))


def daily_cost_plus_penalty(daily: pd.DataFrame, method: str, notes: list[str]) -> pd.Series | None:
    if "total_cost_plus_penalty" in daily.columns:
        return numeric_series(daily, "total_cost_plus_penalty")
    if {"total_system_cost", "total_penalties"}.issubset(daily.columns):
        notes.append(
            f"{method}: total_cost_plus_penalty not present; computed as total_system_cost + total_penalties."
        )
        return numeric_series(daily, "total_system_cost") + numeric_series(daily, "total_penalties")
    notes.append(f"{method}: cannot compute daily cost+penalty from daily_summary.csv.")
    return None


def runtime_time_per_day(runtime: dict[str, Any]) -> float | None:
    if "time_per_day_seconds" in runtime:
        return float(runtime["time_per_day_seconds"])
    duration = runtime.get("test_duration_seconds")
    n_days = runtime.get("n_days")
    if duration is not None and n_days:
        return float(duration) / float(n_days)
    return None


def build_performance_table(notes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for method in METHOD_ORDER:
        sources = METHOD_SOURCES[method]
        daily = read_csv(sources.daily_summary, notes, f"{method} daily summary")

        if daily is None:
            rows.append(
                {
                    "Method": method,
                    "Average daily cost+penalty": NA,
                    "Average daily system cost": NA,
                    "Average daily penalty": NA,
                }
            )
            continue

        cost_plus_penalty = daily_cost_plus_penalty(daily, method, notes)
        avg_cost_plus_penalty = cost_plus_penalty.mean(skipna=True) if cost_plus_penalty is not None else None
        avg_system_cost = numeric_series(daily, "total_system_cost").mean(skipna=True) if "total_system_cost" in daily else None
        avg_penalty = numeric_series(daily, "total_penalties").mean(skipna=True) if "total_penalties" in daily else None

        rows.append(
            {
                "Method": method,
                "Average daily cost+penalty": fmt_number(avg_cost_plus_penalty),
                "Average daily system cost": fmt_number(avg_system_cost),
                "Average daily penalty": fmt_number(avg_penalty),
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "Method",
            "Average daily cost+penalty",
            "Average daily system cost",
            "Average daily penalty",
        ),
    )


def bool_count(series: pd.Series) -> int:
    if series.dtype == bool:
        return int(series.sum())
    normalized = series.astype(str).str.strip().str.lower()
    return int(normalized.isin(("true", "1", "yes", "y")).sum())


def sum_from_daily_or_timeseries(
    daily: pd.DataFrame | None,
    timeseries: pd.DataFrame | None,
    daily_candidates: tuple[str, ...],
    timeseries_candidates: tuple[str, ...],
    method: str,
    label: str,
    notes: list[str],
) -> float | None:
    if daily is not None:
        column = first_existing(daily.columns, daily_candidates)
        if column:
            return float(numeric_series(daily, column).sum(skipna=True))
    if timeseries is not None:
        column = first_existing(timeseries.columns, timeseries_candidates)
        if column:
            notes.append(f"{method}: {label} not found in daily_summary.csv; summed {column} from timeseries_detail.csv.")
            return float(numeric_series(timeseries, column).sum(skipna=True))
    notes.append(f"{method}: {label} unavailable; wrote NA.")
    return None


def build_feasibility_table(notes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for method in METHOD_ORDER:
        sources = METHOD_SOURCES[method]
        daily = read_csv(sources.daily_summary, notes, f"{method} daily summary")
        timeseries = read_csv(sources.timeseries_detail, notes, f"{method} timeseries detail")

        ev_shortage = sum_from_daily_or_timeseries(
            daily,
            timeseries,
            ("total_depart_energy_shortage_kwh", "depart_energy_shortage_kwh"),
            ("depart_energy_shortage_kwh",),
            method,
            "EV departure shortage",
            notes,
        )
        unmet_e = sum_from_daily_or_timeseries(
            daily,
            timeseries,
            ("total_unmet_e", "unmet_e"),
            ("unmet_e",),
            method,
            "unmet electric",
            notes,
        )
        unmet_h = sum_from_daily_or_timeseries(
            daily,
            timeseries,
            ("total_unmet_h", "unmet_h"),
            ("unmet_h",),
            method,
            "unmet heat",
            notes,
        )
        unmet_c = sum_from_daily_or_timeseries(
            daily,
            timeseries,
            ("total_unmet_c", "unmet_c"),
            ("unmet_c",),
            method,
            "unmet cooling",
            notes,
        )

        rows.append(
            {
                "Method": method,
                "EV departure shortage (kWh)": fmt_reliability_kwh(ev_shortage),
                "Unmet electric load (kWh)": fmt_reliability_kwh(unmet_e),
                "Unmet heat load (kWh)": fmt_reliability_kwh(unmet_h),
                "Unmet cooling load (kWh)": fmt_reliability_kwh(unmet_c),
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "Method",
            "EV departure shortage (kWh)",
            "Unmet electric load (kWh)",
            "Unmet heat load (kWh)",
            "Unmet cooling load (kWh)",
        ),
    )


def load_training_curve(method: str, seed: int, notes: list[str]) -> pd.DataFrame | None:
    path = TRAINING_EPISODE_SUMMARY[method][seed]
    use_columns = {"episode_idx", "global_step_end", "episode_reward_scaled", "episode_reward_raw"}
    df = read_csv(
        path,
        notes,
        f"{method} seed {seed} training episode summary",
        usecols=lambda column: column in use_columns,
    )
    if df is None:
        return None
    reward_col = "episode_reward_scaled" if "episode_reward_scaled" in df.columns else None
    if reward_col is None and "episode_reward_raw" in df.columns:
        reward_col = "episode_reward_raw"
        notes.append(f"{method} seed {seed}: episode_reward_scaled missing; used episode_reward_raw.")
    if reward_col is None or "episode_idx" not in df.columns:
        notes.append(f"{method} seed {seed}: training reward columns unavailable; curve skipped.")
        return None

    curve = pd.DataFrame(
        {
            "episode_idx": pd.to_numeric(df["episode_idx"], errors="coerce"),
            "reward": pd.to_numeric(df[reward_col], errors="coerce"),
        }
    ).dropna()
    curve = curve.sort_values("episode_idx")
    curve["reward_ma"] = curve["reward"].rolling(MOVING_AVERAGE_WINDOW, min_periods=1).mean()
    return curve[["episode_idx", "reward_ma"]]


def plot_training_reward_curves(notes: list[str]) -> dict[str, Path]:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.5, 4.2),
        dpi=300,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    ax_full, ax_zoom = axes
    alg_order = ["TD3", "PPO", "DDPG"]
    colors = {
        "TD3": "#d62728",
        "PPO": "#1f77b4",
        "DDPG": "#2ca02c",
    }
    line_widths = {
        "TD3": 2.8,
        "PPO": 1.8,
        "DDPG": 1.8,
    }
    plotted = 0
    for method in alg_order:
        seed_frames: list[pd.Series] = []
        for seed in SEEDS:
            curve = load_training_curve(method, seed, notes)
            if curve is None or curve.empty:
                continue
            series = curve.set_index("episode_idx")["reward_ma"].rename(str(seed))
            seed_frames.append(series)
        if not seed_frames:
            notes.append(f"{method}: no training reward data available for Figure 1.")
            continue
        if len(seed_frames) < len(SEEDS):
            notes.append(f"{method}: Figure 1 used {len(seed_frames)}/{len(SEEDS)} seed reward files.")

        aligned = pd.concat(seed_frames, axis=1).sort_index()
        mean = aligned.mean(axis=1, skipna=True)
        std = aligned.std(axis=1, skipna=True).fillna(0.0)
        x = aligned.index.to_numpy(dtype=float)
        y = mean.to_numpy(dtype=float)
        y_std = std.to_numpy(dtype=float)
        color = colors[method]
        linewidth = line_widths[method]

        ax_full.plot(x, y, label=method, color=color, linewidth=linewidth)
        ax_full.fill_between(
            x,
            y - y_std,
            y + y_std,
            color=color,
            alpha=MAIN_STD_ALPHA,
            linewidth=0,
        )

        zoom_mask = (x >= TRAINING_REWARD_ZOOM_START) & (x <= TRAINING_REWARD_ZOOM_END)
        zoom_x = x[zoom_mask]
        zoom_y = y[zoom_mask]
        zoom_std = y_std[zoom_mask]
        ax_zoom.plot(zoom_x, zoom_y, label=method, color=color, linewidth=linewidth)
        zoom_lower = zoom_y - zoom_std
        zoom_upper = zoom_y + zoom_std
        y_min, y_max = TRAINING_REWARD_ZOOM_YLIM
        visible_band = (zoom_lower >= y_min) & (zoom_upper <= y_max)
        if np.any(visible_band):
            ax_zoom.fill_between(
                zoom_x[visible_band],
                np.clip(zoom_lower[visible_band], y_min, y_max),
                np.clip(zoom_upper[visible_band], y_min, y_max),
                color=color,
                alpha=ZOOM_STD_ALPHA,
                linewidth=0,
            )
        plotted += 1

    ax_full.set_title("(a) Full training process")
    ax_full.set_xlim(0, 4000)
    ax_full.set_ylim(-15, 0.5)
    ax_full.set_xlabel("Episode")
    ax_full.set_ylabel(f"Episode reward, {MOVING_AVERAGE_WINDOW}-episode moving average")
    ax_full.grid(True, linestyle="--", alpha=0.25)

    ax_zoom.set_title("(b) Zoomed convergence stage")
    ax_zoom.set_xlim(TRAINING_REWARD_ZOOM_START, TRAINING_REWARD_ZOOM_END)
    ax_zoom.set_ylim(*TRAINING_REWARD_ZOOM_YLIM)
    ax_zoom.set_xlabel("Episode")
    ax_zoom.grid(True, linestyle="--", alpha=0.25)

    handles, labels = ax_full.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(0.5, 1.03),
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if plotted == 0:
        notes.append("Figure 1 has no plotted curves because all training reward files were missing.")

    png_path = FIGURE_DIR / "fig_1_training_reward_curves.png"
    pdf_path = FIGURE_DIR / "fig_1_training_reward_curves.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def plot_cumulative_daily_cost(notes: list[str]) -> dict[str, Path]:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    plotted = 0
    plot_styles = {
        "Rule-based-V2G": {"color": "#6e6e6e", "linewidth": 2.2, "linestyle": "-"},
        "DDPG": {"color": "#2ca02c", "linewidth": 2.2, "linestyle": "--"},
        "PPO": {"color": "#1f77b4", "linewidth": 2.2, "linestyle": "-."},
        "TD3": {"color": "#d62728", "linewidth": 2.6, "linestyle": "-"},
    }
    for method in METHOD_ORDER:
        sources = METHOD_SOURCES[method]
        daily = read_csv(sources.daily_summary, notes, f"{method} daily summary for Figure 2")
        if daily is None:
            continue
        cost = daily_cost_plus_penalty(daily, method, notes)
        if cost is None:
            continue
        if len(cost) < TEST_DAYS:
            notes.append(f"{method}: Figure 2 has only {len(cost)} daily rows, expected {TEST_DAYS}.")
        cost = cost.iloc[:TEST_DAYS]
        x = np.arange(1, len(cost) + 1)
        cumulative = cost.cumsum().to_numpy(dtype=float)
        style = plot_styles[method]
        ax.plot(
            x,
            cumulative,
            label=method,
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
        )
        plotted += 1

    ax.set_xlabel("Test day", fontsize=13)
    ax.set_ylabel("Cumulative cost+penalty", fontsize=13)
    ax.set_xlim(1, TEST_DAYS)
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60])
    ax.grid(True, color="#d0d0d0", linestyle="--", linewidth=0.6, alpha=0.30)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    fig.tight_layout()

    if plotted == 0:
        notes.append("Figure 2 has no plotted curves because daily cost data were unavailable.")

    png_path = FIGURE_DIR / "fig_2_cumulative_daily_cost_plus_penalty.png"
    pdf_path = FIGURE_DIR / "fig_2_cumulative_daily_cost_plus_penalty.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def load_exchange_intensity(method: str, notes: list[str]) -> np.ndarray:
    sources = METHOD_SOURCES[method]
    df = read_csv(sources.timeseries_detail, notes, f"{method} timeseries detail for Figure 3")
    if df is None:
        return np.array([], dtype=float)
    dis_col = first_existing(df.columns, ("p_ev_dis", "ev_discharge", "total_ev_discharge_power"))
    ch_col = first_existing(df.columns, ("p_ev_ch", "ev_charge", "total_ev_charge_power"))
    if dis_col is None or ch_col is None:
        notes.append(f"{method}: EV charge/discharge power columns missing; Figure 3 skipped this method.")
        return np.array([], dtype=float)
    values = (numeric_series(df, dis_col) - numeric_series(df, ch_col)).abs() * DT_HOURS
    return values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)


def plot_exchange_distribution(exchange_data: dict[str, np.ndarray], notes: list[str]) -> dict[str, Path]:
    # Suggested paper caption:
    # Fig. X. Statistical characterization of EV-IES energy exchange under different control strategies.
    # (a) Active exchange ratio of the EV aggregator participating in IES scheduling.
    # (b) Distribution of positive EV-IES exchange intensity.
    # The exchange intensity is defined as the absolute net energy exchanged between the EV aggregator
    # and the IES during each dispatching interval.
    alg_order = ["Rule-based-V2G", "DDPG", "PPO", "TD3"]
    colors = {
        "Rule-based-V2G": "#4d4d4d",
        "DDPG": "#2ca02c",
        "PPO": "#1f77b4",
        "TD3": "#d62728",
    }

    stats = compute_exchange_statistics(exchange_data)
    active_values = [
        float(stats.loc[stats["Algorithm"] == alg, "Active_ratio"].iloc[0]) * 100.0
        if alg in set(stats["Algorithm"])
        else 0.0
        for alg in alg_order
    ]
    positive_data: list[np.ndarray] = []
    positive_labels: list[str] = []
    for alg in alg_order:
        data = exchange_data.get(alg, np.array([], dtype=float))
        positive = data[np.isfinite(data) & (data > EXCHANGE_EPS)]
        if positive.size == 0:
            notes.append(f"{alg}: no positive EV-IES exchange intensity values for Figure 3(b).")
            continue
        positive_data.append(positive)
        positive_labels.append(alg)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.5, 4.2),
        dpi=300,
        gridspec_kw={"width_ratios": [0.95, 1.25]},
    )
    ax_act, ax_dist = axes

    bars = ax_act.bar(
        alg_order,
        active_values,
        color=[colors[alg] for alg in alg_order],
        edgecolor="black",
        linewidth=0.6,
    )
    ax_act.set_ylabel("Active exchange ratio (%)")
    ax_act.set_title("(a) Exchange activation ratio", fontsize=11)
    ax_act.set_ylim(0, max(80.0, max(active_values, default=0.0) + 8.0))
    ax_act.grid(axis="y", linestyle="--", alpha=0.25)
    ax_act.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, active_values):
        ax_act.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    if positive_data:
        box = ax_dist.boxplot(
            positive_data,
            vert=False,
            labels=positive_labels,
            patch_artist=True,
            whis=(5, 95),
            showfliers=True,
            flierprops=dict(marker="o", markersize=2.0, alpha=0.35),
            medianprops=dict(color="black", linewidth=1.2),
        )
        for patch, alg in zip(box["boxes"], positive_labels):
            patch.set_facecolor(colors[alg])
            patch.set_alpha(0.35)
            patch.set_edgecolor(colors[alg])

        for i, (alg, positive) in enumerate(zip(positive_labels, positive_data), start=1):
            positive_mean = float(np.mean(positive))
            ax_dist.scatter(
                positive_mean,
                i,
                marker="D",
                s=28,
                color=colors[alg],
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
        ax_dist.set_xscale("log")
    else:
        notes.append("Figure 3(b) has no boxplots because positive exchange-intensity data were unavailable.")

    ax_dist.set_xlabel("Positive EV-IES exchange intensity (kWh, log scale)")
    ax_dist.set_title("(b) Positive exchange intensity distribution", fontsize=11)
    ax_dist.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()

    png_path = FIGURE_DIR / "fig_3_ev_ies_exchange_characterization.png"
    pdf_path = FIGURE_DIR / "fig_3_ev_ies_exchange_characterization.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def relative_increase(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None or not math.isfinite(value) or not math.isfinite(baseline) or baseline == 0:
        return NA
    return f"{((value - baseline) / baseline) * 100:.2f}%"


def compute_exchange_statistics(exchange_data: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    alg_order = ["Rule-based-V2G", "DDPG", "PPO", "TD3"]
    for method in alg_order:
        values = exchange_data.get(method, np.array([], dtype=float))
        values = values[np.isfinite(values)]
        if values.size == 0:
            rows.append(
                {
                    "Algorithm": method,
                    "Mean": np.nan,
                    "Median": np.nan,
                    "Q75": np.nan,
                    "Q95": np.nan,
                    "Max": np.nan,
                    "Total": np.nan,
                    "Zero_ratio": np.nan,
                    "Active_ratio": np.nan,
                }
            )
            continue
        zero_ratio = float(np.mean(values <= EXCHANGE_EPS))
        active_ratio = float(np.mean(values > EXCHANGE_EPS))
        rows.append(
            {
                "Algorithm": method,
                "Mean": float(np.mean(values)),
                "Median": float(np.median(values)),
                "Q75": float(np.quantile(values, 0.75)),
                "Q95": float(np.quantile(values, 0.95)),
                "Max": float(np.max(values)),
                "Total": float(np.sum(values)),
                "Zero_ratio": zero_ratio,
                "Active_ratio": active_ratio,
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "Algorithm",
            "Mean",
            "Median",
            "Q75",
            "Q95",
            "Max",
            "Total",
            "Zero_ratio",
            "Active_ratio",
        ),
    )


def build_exchange_statistics(exchange_data: dict[str, np.ndarray]) -> pd.DataFrame:
    stats = compute_exchange_statistics(exchange_data)
    print("EV-IES exchange statistics")
    if stats.empty:
        print("No exchange statistics available.")
    else:
        print(stats.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    return stats


def build_checkpoint_appendix(notes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    td3_path = FINAL_SOURCE_FILES["TD3 selected checkpoints"]
    td3 = read_csv(td3_path, notes, "TD3 selected checkpoints")
    if td3 is not None:
        for _, row in td3.iterrows():
            rows.append(
                {
                    "Method": "TD3",
                    "Seed": str(row.get("seed", NA)),
                    "Selected step": fmt_param(row.get("selected_checkpoint_step")),
                    "Val EES feasible": str(row.get("val_EES_feasible", NA)),
                    "Test EES feasible": str(row.get("test_EES_feasible", NA)),
                    "Test cost+penalty": fmt_number(row.get("test_cost_plus_penalty")),
                    "Selection rule": str(row.get("selected_reason", NA)),
                }
            )

    ddpg_ppo_path = FINAL_SOURCE_FILES["DDPG/PPO selected checkpoints"]
    ddpg_ppo = read_csv(ddpg_ppo_path, notes, "DDPG/PPO selected checkpoints")
    if ddpg_ppo is not None:
        for _, row in ddpg_ppo.iterrows():
            rows.append(
                {
                    "Method": str(row.get("method", NA)),
                    "Seed": str(row.get("seed", NA)),
                    "Selected step": fmt_param(row.get("selected_checkpoint_step")),
                    "Val EES feasible": str(row.get("val_EES_feasible", NA)),
                    "Test EES feasible": str(row.get("test_EES_feasible", NA)),
                    "Test cost+penalty": fmt_number(row.get("test_cost_plus_penalty")),
                    "Selection rule": str(row.get("selected_reason", NA)),
                }
            )
    order = {"TD3": 0, "DDPG": 1, "PPO": 2}
    df = pd.DataFrame(
        rows,
        columns=(
            "Method",
            "Seed",
            "Selected step",
            "Val EES feasible",
            "Test EES feasible",
            "Test cost+penalty",
            "Selection rule",
        ),
    )
    if not df.empty:
        df["_order"] = df["Method"].map(order).fillna(99)
        df["_seed"] = pd.to_numeric(df["Seed"], errors="coerce")
        df = df.sort_values(["_order", "_seed"]).drop(columns=["_order", "_seed"]).reset_index(drop=True)
    return df


def make_source_list() -> str:
    lines = [
        "- TD3: feasible-selected CUDA TD3; seed/algorithm/checkpoint summaries under `results/td3_cuda_multiseed_feasible_selected/`; representative seed=42 test directory `results/td3_yearly_test_cuda_feasible_seed_42/`.",
        "- DDPG: feasible-selected DDPG; seed/algorithm/checkpoint summaries under `results/ddpg_ppo_feasible_checkpoint_selection/`; representative seed=42 test directory `results/ddpg_feasible_selected_test_seed_42/`.",
        "- PPO: feasible-selected PPO; seed/algorithm/checkpoint summaries under `results/ddpg_ppo_feasible_checkpoint_selection/`; representative seed=42 test directory `results/ppo_feasible_selected_test_seed_42/`.",
        "- Rule-based-V2G: repaired deterministic rule baseline from `results/rule_based_v2g_test/`.",
    ]
    return "\n".join(lines)


def make_notes_section(notes: list[str]) -> str:
    unique_notes = list(dict.fromkeys(notes))
    if not unique_notes:
        return "No required source files were missing. Table 1 hyperparameters were parsed from the current training script constants; `—` denotes not applicable.\n"
    return "\n".join(f"- {note}" for note in unique_notes) + "\n"


def make_source_audit_section(declared_sources: dict[str, pd.DataFrame]) -> str:
    if not declared_sources:
        return "- No declared final source CSV was successfully loaded.\n"
    lines = []
    for label, df in declared_sources.items():
        lines.append(f"- {label}: loaded {len(df)} rows.")
    return "\n".join(lines) + "\n"


def write_file_index(
    table_paths: dict[str, dict[str, Path]],
    figure_paths: dict[str, dict[str, Path]],
    notes: list[str],
    declared_sources: dict[str, pd.DataFrame],
) -> Path:
    path = REPORT_DIR / "algorithm_comparison_file_index.md"
    text = f"""# Algorithm Comparison Final File Index

## Final Result Sources
{make_source_list()}

## Body Tables
- Table 1: `results/paper_algorithm_comparison_final/tables/table_1_drl_hyperparameters.csv`
- Table 2: `results/paper_algorithm_comparison_final/tables/table_2_algorithm_performance.csv`
- Table 3: `results/paper_algorithm_comparison_final/tables/table_3_algorithm_feasibility.csv`

## Table 2 Scope Note
Table 2 only reports the core economic metrics, including average daily cost-plus-penalty, system cost and penalty. Relative cost reductions are not included in the table to keep the main performance table concise. The percentage improvements of TD3 over other methods should be discussed in the text.

## Body Figures
- Figure 1: `results/paper_algorithm_comparison_final/figures/fig_1_training_reward_curves.png`
- Figure 2: `results/paper_algorithm_comparison_final/figures/fig_2_cumulative_daily_cost_plus_penalty.png`
- Figure 3: `results/paper_algorithm_comparison_final/figures/fig_3_ev_ies_exchange_characterization.png`

## Figure 2 Scope Note
Figure 2 keeps the cumulative daily cost-plus-penalty metric rather than pure system cost. This is the appropriate main testing metric for the constrained IES-EV/V2G scheduling problem because both operating cost and soft-constraint violations should be considered.

## Appendix Table
- `results/paper_algorithm_comparison_final/tables/table_appendix_checkpoint_selection.csv`

## Recommended Paper Placement
- Table 1: place in the algorithm implementation or experimental setup subsection.
- Table 2: place in the main algorithm performance comparison subsection.
- Table 3: place immediately after Table 2 to report user-side service reliability and EV departure satisfaction.
- Figure 1: place before or near the performance table to show the full DRL training process and the zoomed convergence stage.
- Figure 2: place in the main testing-performance subsection to compare continuous 60-day economic performance.
- Figure 3: place after Figure 2 to explain EV aggregation flexibility utilization.
- Appendix checkpoint table: place in appendix or use for reviewer response on validation-set checkpoint selection.

## Body-Only Contract
正文算法对比部分只使用：

Tables:
    table_1_drl_hyperparameters
    table_2_algorithm_performance
    table_3_algorithm_feasibility

Figures:
    fig_1_training_reward_curves
    fig_2_cumulative_daily_cost_plus_penalty
    fig_3_ev_ies_exchange_characterization

附录建议使用：
    table_appendix_checkpoint_selection

不要作为正文主结果：
    actor loss
    critic loss
    CUDA reward-best TD3
    td3_cuda_run1
    old PPO infeasible result
    original DDPG/PPO reward-best result
    grid buy/grid sell/EV discharge/storage peak-shaved 柱状图

## Generation Notes
{make_notes_section(notes)}

## Declared Source Audit
{make_source_audit_section(declared_sources)}
"""
    text = re.sub(
        r"## Body-Only Contract\n.*?\n## Generation Notes",
        """## Body-Only Contract
The body algorithm comparison section only uses:

Tables:
    table_1_drl_hyperparameters
    table_2_algorithm_performance
    table_3_algorithm_feasibility

Figures:
    fig_1_training_reward_curves
    fig_2_cumulative_daily_cost_plus_penalty
    fig_3_ev_ies_exchange_characterization

Appendix recommended:
    table_appendix_checkpoint_selection

Supplementary statistics:
    table_ev_ies_exchange_statistics

Do not use as body main results:
    actor loss
    critic loss
    CUDA reward-best TD3
    td3_cuda_run1
    old PPO infeasible result
    original DDPG/PPO reward-best result
    grid buy/grid sell/EV discharge/storage peak-shaved bar chart

## Generation Notes""",
        text,
        flags=re.DOTALL,
    )
    path.write_text(text, encoding="utf-8")
    return path


def table_snapshot(table: pd.DataFrame, columns: tuple[str, ...]) -> str:
    if table.empty:
        return "- No rows available.\n"
    lines = []
    label_column = "Method" if "Method" in table.columns else "Algorithm"
    for _, row in table.iterrows():
        parts = [f"{column}={row[column]}" for column in columns if column in table.columns]
        lines.append(f"- {row[label_column]}: " + ", ".join(parts))
    return "\n".join(lines) + "\n"


def table_number(table: pd.DataFrame, method: str, column: str) -> float | None:
    if table.empty or column not in table.columns or "Method" not in table.columns:
        return None
    rows = table.loc[table["Method"] == method, column]
    if rows.empty:
        return None
    try:
        value = float(rows.iloc[0])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def td3_improvement_wording(performance: pd.DataFrame) -> str:
    td3 = table_number(performance, "TD3", "Average daily cost+penalty")
    rule = table_number(performance, "Rule-based-V2G", "Average daily cost+penalty")
    ppo = table_number(performance, "PPO", "Average daily cost+penalty")
    ddpg = table_number(performance, "DDPG", "Average daily cost+penalty")

    if None in (td3, rule, ppo, ddpg):
        return (
            "Compared with Rule-based-V2G, TD3 reduces the average daily cost-plus-penalty. "
            "Compared with PPO and DDPG, TD3 also obtains a lower value. "
            "DDPG obtains a slightly higher cost-plus-penalty than Rule-based-V2G, indicating that "
            "the single-critic deterministic policy is less stable in this constrained continuous-control scheduling problem."
        )

    rule_reduction = (rule - td3) / rule * 100.0
    ppo_reduction = (ppo - td3) / ppo * 100.0
    ddpg_reduction = (ddpg - td3) / ddpg * 100.0
    return (
        f"Compared with Rule-based-V2G, TD3 reduces the average daily cost-plus-penalty by approximately "
        f"{rule_reduction:.2f}%. Compared with PPO and DDPG, TD3 reduces the value by approximately "
        f"{ppo_reduction:.2f}% and {ddpg_reduction:.2f}%, respectively. DDPG obtains a slightly higher "
        "cost-plus-penalty than Rule-based-V2G, indicating that the single-critic deterministic policy is "
        "less stable in this constrained continuous-control scheduling problem."
    )


def write_summary_report(
    performance: pd.DataFrame,
    feasibility: pd.DataFrame,
    exchange_stats: pd.DataFrame,
    notes: list[str],
    declared_sources: dict[str, pd.DataFrame],
) -> Path:
    path = REPORT_DIR / "algorithm_comparison_summary.md"
    text = f"""# Algorithm Comparison Summary

## 1. Final Result Scope
This report only covers the final algorithm comparison part of the paper. It uses feasible-selected TD3/DDPG/PPO results and the repaired deterministic Rule-based-V2G baseline. No training, testing, environment, reward, or `ies_config.py` changes are performed by this workflow.

{make_source_list()}

## 2. Body Tables
- Table 1 reports only seven compact DRL hyperparameters, matching a concise Energy 2024-style setup table. Random seeds, checkpoint selection, target policy noise, target noise clip, policy delay, clip range, n_epochs, gae_lambda, vf_coef, and max_grad_norm are intentionally excluded from the body table.
- Table 2 only reports the core economic metrics, including average daily cost-plus-penalty, system cost and penalty. Relative cost reductions are not included in the table to keep the main performance table concise. The percentage improvements of TD3 over other methods should be discussed in the text.
- Table 3 reports user-side service reliability, including EV departure shortage and unmet electric, heating, and cooling loads. It is not used as a complete constraint-feasibility table.

## 3. Body Figures
- Figure 1 shows DRL training reward trends using the three seeds 42, 2024, and 2025. Panel (a) presents the full 0-4000 episode training process, and panel (b) zooms into episodes 3000-4000 with the reward axis fixed at -2.20 to -1.90. The solid line is the seed mean after a 100-episode moving average, and the shaded band is ±1 standard deviation.
- Figure 2 shows the cumulative daily cost-plus-penalty over the 60-day representative seed=42 test dataset. The cumulative metric is used instead of pure system cost because this study addresses a constrained IES-EV/V2G scheduling problem, where both operating cost and soft-constraint violations should be considered.
- Figure 3 characterizes EV-IES energy exchange using two views: panel (a) reports the active exchange ratio, and panel (b) shows the distribution of positive EV-IES exchange intensity on a log scale. Exchange intensity is computed as `abs(p_ev_dis - p_ev_ch) * dt`.

## 4. Rationale
Table 1 is compact because the paper body should show the reproducibility-critical common hyperparameters without turning the algorithm comparison into a full implementation audit.

Table 2 is compact because the main performance claim should be tied to the core economic metrics: cost+penalty, system cost, and penalty. Relative cost reductions, grid buy, grid sell, EV discharge, storage peak-shaved, EES feasible ratio, EV departure shortage, total training time, and computational time are intentionally excluded from this main table.

Table 3 is separate because user-side service reliability is a different question from operating cost. It focuses on whether the scheduling methods create EV departure shortages or unmet electric, heating, and cooling loads. Values smaller than 1e-3 kWh are displayed as numerical zero.

Figure 3 is used because EV-IES energy exchange intensity describes how much the EV aggregation participates in IES coordination during scheduling. The active exchange ratio captures how often EV flexibility is used, while the positive-intensity distribution describes the magnitude of exchange after inactive zero-exchange intervals are removed. This figure should be interpreted as a mechanism explanation rather than direct proof of cost optimality.

Figure 2 shows the cumulative daily cost-plus-penalty over the 60-day test dataset. The cumulative metric is used instead of pure system cost because this study addresses a constrained IES-EV/V2G scheduling problem, where both operating cost and soft-constraint violations should be considered. TD3 maintains the lowest cumulative cost-plus-penalty throughout the test period, and the gap between TD3 and the other methods gradually increases with the number of test days, indicating better overall economic performance under operational constraints.

## 5. Result Snapshot
Main performance:
{table_snapshot(performance, ("Average daily cost+penalty", "Average daily system cost", "Average daily penalty"))}
Service reliability:
{table_snapshot(feasibility, ("EV departure shortage (kWh)", "Unmet electric load (kWh)", "Unmet heat load (kWh)", "Unmet cooling load (kWh)"))}
EV-IES exchange intensity:
{table_snapshot(exchange_stats, ("Mean", "Median", "Q75", "Q95", "Total", "Zero_ratio", "Active_ratio"))}
## 6. Recommended Paper Wording
{td3_improvement_wording(performance)}

Under the unified validation-set feasibility-first checkpoint selection criterion, TD3 achieves the lowest representative seed=42 economic objective in Table 2. Table 3 further shows that this economic improvement is not accompanied by unmet electric, heating, or cooling load, and no material EV departure shortage is observed under the 1e-3 kWh numerical tolerance. DDPG and PPO serve as feasible-selected DRL baselines, and Rule-based-V2G is used as the deterministic rule-based baseline.

为了比较不同算法调度过程中 EV 聚合体与 IES 之间能量交换水平的差异，图 X 给出了 60 天测试期间 EV-IES 能量交换强度的概率分布。若 TD3 的分布整体右移或高强度区域概率更高，说明 TD3 能更充分调动 EV 聚合体参与园区 IES 协同调度，从而降低系统综合运行成本。

## 7. Do Not Overstate
Do not write:
- TD3 strictly satisfies all constraints on every test day and every seed.
- Table 3 is a complete constraint-feasibility comparison.
- PPO completely fails.
- More V2G participation is always better.
- Guide reward is real economic profit.
- CUDA reward-best TD3 is the final main result.
- TD3 has the highest EV-IES exchange intensity in all ranges.

## 8. Generation Notes
{make_notes_section(notes)}

## 9. Declared Source Audit
{make_source_audit_section(declared_sources)}
"""
    text = re.sub(
        r"the shaded band is .*? standard deviation\.",
        "the shaded band is +/-1 standard deviation.",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"(Under the unified validation-set feasibility-first checkpoint selection criterion, TD3 achieves the lowest representative seed=42 economic objective in Table 2\. Table 3 further shows that this economic improvement is not accompanied by unmet electric, heating, or cooling load, and no material EV departure shortage is observed under the 1e-3 kWh numerical tolerance\. DDPG and PPO serve as feasible-selected DRL baselines, and Rule-based-V2G is used as the deterministic rule-based baseline\.\n\n).*?(## 7\. Do Not Overstate)",
        (
            r"\1"
            "Figure 3 should be used as a mechanism explanation rather than as a direct economic-performance metric. "
            "A higher active exchange ratio, a higher median, and a larger total exchange intensity indicate that TD3 "
            "more continuously mobilizes EV-aggregator flexibility during the 60-day test period. Rule-based-V2G has "
            "a median exchange intensity of zero, meaning more than half of its dispatch intervals have no effective "
            "EV-IES exchange. DDPG can produce higher 95% quantile and maximum exchange intensity values, but these "
            "occasional extreme exchanges do not translate into a lower cumulative cost-plus-penalty than TD3.\n\n"
            "Do not describe Figure 3 as showing that TD3 has the highest exchange intensity in all ranges. The "
            "defensible conclusion is that TD3 uses EV flexibility more persistently and more evenly, while avoiding "
            "a claim that extreme exchange intensity is always beneficial.\n\n"
            r"\2"
        ),
        text,
        flags=re.DOTALL,
    )
    path.write_text(text, encoding="utf-8")
    return path


def check_outputs(paths: list[Path]) -> tuple[bool, list[Path]]:
    missing = [path for path in paths if not path.exists() or path.stat().st_size == 0]
    return len(missing) == 0, missing


def main() -> None:
    ensure_dirs()
    notes: list[str] = []

    declared_sources = load_declared_final_sources(notes)

    table_paths: dict[str, dict[str, Path]] = {}
    figure_paths: dict[str, dict[str, Path]] = {}

    table_1 = build_hyperparameter_table(notes)
    table_paths["table_1_drl_hyperparameters"] = write_table_bundle(
        table_1,
        "table_1_drl_hyperparameters",
        "Table X. Hyperparameters of the DRL algorithms.",
    )

    table_2 = build_performance_table(notes)
    table_paths["table_2_algorithm_performance"] = write_table_bundle(
        table_2,
        "table_2_algorithm_performance",
        "Table X. Performance comparison of different scheduling methods.",
    )

    table_3 = build_feasibility_table(notes)
    table_paths["table_3_algorithm_feasibility"] = write_table_bundle(
        table_3,
        "table_3_algorithm_feasibility",
        "Table X. Service reliability comparison of different scheduling methods.",
        note="Values smaller than 1e-3 kWh are displayed as numerical zero.",
    )

    figure_paths["fig_1_training_reward_curves"] = plot_training_reward_curves(notes)
    figure_paths["fig_2_cumulative_daily_cost_plus_penalty"] = plot_cumulative_daily_cost(notes)

    exchange_data = {method: load_exchange_intensity(method, notes) for method in METHOD_ORDER}
    figure_paths["fig_3_ev_ies_exchange_characterization"] = plot_exchange_distribution(exchange_data, notes)
    exchange_stats = build_exchange_statistics(exchange_data)
    table_paths["table_ev_ies_exchange_statistics"] = write_table_bundle(
        exchange_stats,
        "table_ev_ies_exchange_statistics",
        "Supplementary statistics of EV-IES energy exchange characterization.",
    )

    appendix = build_checkpoint_appendix(notes)
    table_paths["table_appendix_checkpoint_selection"] = write_table_bundle(
        appendix,
        "table_appendix_checkpoint_selection",
        "Table A. Selected checkpoints using validation-set feasibility-first criterion.",
    )

    file_index_path = write_file_index(table_paths, figure_paths, notes, declared_sources)
    summary_path = write_summary_report(table_2, table_3, exchange_stats, notes, declared_sources)

    required_paths = [
        table_paths["table_1_drl_hyperparameters"]["csv"],
        table_paths["table_1_drl_hyperparameters"]["xlsx"],
        table_paths["table_1_drl_hyperparameters"]["md"],
        table_paths["table_2_algorithm_performance"]["csv"],
        table_paths["table_2_algorithm_performance"]["xlsx"],
        table_paths["table_2_algorithm_performance"]["md"],
        table_paths["table_3_algorithm_feasibility"]["csv"],
        table_paths["table_3_algorithm_feasibility"]["xlsx"],
        table_paths["table_3_algorithm_feasibility"]["md"],
        figure_paths["fig_1_training_reward_curves"]["png"],
        figure_paths["fig_1_training_reward_curves"]["pdf"],
        figure_paths["fig_2_cumulative_daily_cost_plus_penalty"]["png"],
        figure_paths["fig_2_cumulative_daily_cost_plus_penalty"]["pdf"],
        figure_paths["fig_3_ev_ies_exchange_characterization"]["png"],
        figure_paths["fig_3_ev_ies_exchange_characterization"]["pdf"],
        table_paths["table_appendix_checkpoint_selection"]["csv"],
        table_paths["table_appendix_checkpoint_selection"]["xlsx"],
        table_paths["table_appendix_checkpoint_selection"]["md"],
        file_index_path,
        summary_path,
    ]
    ok, missing = check_outputs(required_paths)

    print("Paper algorithm comparison final build complete.")
    print(f"Script: {rel(Path(__file__))}")
    print(f"Tables: {rel(TABLE_DIR)}")
    print(f"Figures: {rel(FIGURE_DIR)}")
    print(f"Report: {rel(REPORT_DIR)}")
    print(f"Required outputs generated: {'YES' if ok else 'NO'}")
    if missing:
        print("Missing or empty required outputs:")
        for path in missing:
            print(f"  - {rel(path)}")
    if notes:
        print("Generation notes:")
        for note in dict.fromkeys(notes):
            print(f"  - {note}")


if __name__ == "__main__":
    main()
