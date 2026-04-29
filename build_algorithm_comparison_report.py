from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def ensure_runtime_dependencies() -> None:
    required = ["numpy", "pandas", "matplotlib"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    should_reexec = bool(missing) or sys.version_info[:2] >= (3, 14)
    fallback_python = Path(r"D:\Anaconda\python.exe")
    already_bootstrapped = os.environ.get("ALGORITHM_COMPARISON_BOOTSTRAPPED") == "1"

    if should_reexec and fallback_python.exists() and not already_bootstrapped:
        os.environ["ALGORITHM_COMPARISON_BOOTSTRAPPED"] = "1"
        completed = subprocess.run([str(fallback_python), *sys.argv], check=False)
        raise SystemExit(completed.returncode)

    if missing:
        raise ModuleNotFoundError(
            "Missing required packages for report generation: "
            f"{', '.join(missing)}. Run this script in the project Python environment "
            "that has numpy, pandas, and matplotlib installed."
        )


ensure_runtime_dependencies()

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results" / "comparison"
N_EXCHANGE_BINS = 40
PNG_DPI = 300
EPS = 1e-9
METHOD_COLORS = {
    "Rule-based-V2G": "#7F7F7F",
    "PPO": "#4E79A7",
    "TD3": "#F28E2B",
    "LSTM-TD3": "#59A14F",
}

ECONOMIC_COST_COLUMNS = [
    "total_cost_grid",
    "total_cost_gas",
    "total_cost_om",
    "total_cost_deg",
]

PENALTY_COLUMNS = [
    "total_penalty_unserved_e",
    "total_penalty_unserved_h",
    "total_penalty_unserved_c",
    "total_penalty_depart_energy",
    "total_penalty_depart_risk",
    "total_penalty_surplus_e",
    "total_penalty_surplus_h",
    "total_penalty_surplus_c",
    "total_penalty_export_e",
    "total_penalty_ev_export_guard",
    "total_penalty_terminal_ees_soc",
]

PENALTY_LABELS = {
    "total_penalty_terminal_ees_soc": "EES terminal SOC penalty",
}

EES_TERMINAL_DAILY_COLUMNS = [
    "final_ees_soc",
    "ees_soc_init",
    "terminal_ees_required_soc",
    "terminal_ees_shortage_kwh",
    "total_penalty_terminal_ees_soc",
    "ees_terminal_soc_feasible",
]

EES_TERMINAL_CRITICAL_COLUMNS = [
    "final_ees_soc",
    "terminal_ees_shortage_kwh",
    "total_penalty_terminal_ees_soc",
    "ees_terminal_soc_feasible",
]

ECONOMIC_OUTPUT_COLUMNS = [
    "method",
    "n_days",
    "mean_total_system_cost",
    "sum_total_system_cost",
    "mean_total_cost_grid",
    "sum_total_cost_grid",
    "mean_total_cost_gas",
    "sum_total_cost_gas",
    "mean_total_cost_om",
    "sum_total_cost_om",
    "mean_total_cost_deg",
    "sum_total_cost_deg",
    "mean_total_cost_plus_penalty",
    "sum_total_cost_plus_penalty",
    "cost_reduction_vs_rule_based_pct",
]

PENALTY_OUTPUT_COLUMNS = [
    "method",
    "mean_total_penalties",
    *[f"mean_{column}" for column in PENALTY_COLUMNS],
]

PERFORMANCE_OUTPUT_COLUMNS = [
    "method",
    "n_days",
    "n_steps",
    "avg_daily_system_cost",
    "avg_daily_penalty",
    "avg_daily_cost_plus_penalty",
    "training_duration_seconds",
    "training_duration_hms",
    "test_duration_seconds",
    "time_per_day_seconds",
    "time_per_step_seconds",
    "mean_final_ees_soc",
    "min_final_ees_soc",
    "terminal_ees_feasible_days",
    "terminal_ees_feasible_ratio",
    "sum_terminal_ees_shortage_kwh",
    "mean_terminal_ees_shortage_kwh",
    "sum_penalty_terminal_ees_soc",
]

VALIDATION_REWARD_OUTPUT_COLUMNS = [
    "method",
    "eval_timestep",
    "validation_mean_reward",
    "validation_std_reward",
    "n_eval_episodes",
]

VALIDATION_REWARD_NOTE = (
    "NOTE: Validation reward is computed from SB3 EvalCallback evaluations.npz:\n"
    "validation_mean_reward = mean(results, axis=1)\n"
    "validation_std_reward = std(results, axis=1)\n"
    "It is not SB3 rollout/ep_rew_mean and should not be mixed with training "
    "episode_reward_scaled."
)


@dataclass(frozen=True)
class MethodConfig:
    method: str
    test_dir: Path
    training_dir: Path | None = None
    training_export_xlsx: Path | None = None
    training_episode_csv: Path | None = None
    eval_npz_path: Path | None = None


METHODS = [
    MethodConfig(
        method="Rule-based-V2G",
        test_dir=BASE_DIR / "results" / "rule_based_v2g_test",
        eval_npz_path=None,
    ),
    MethodConfig(
        method="PPO",
        test_dir=BASE_DIR / "results" / "ppo_sb3_direct_test",
        training_dir=BASE_DIR / "results" / "ppo_sb3_direct_training",
        training_export_xlsx=BASE_DIR / "results" / "ppo_sb3_direct_training" / "ppo_sb3_direct_training_export.xlsx",
        training_episode_csv=BASE_DIR / "results" / "ppo_sb3_direct_training" / "episode_summary.csv",
        eval_npz_path=BASE_DIR / "logs" / "ppo_sb3_direct" / "evaluations.npz",
    ),
    MethodConfig(
        method="TD3",
        test_dir=BASE_DIR / "results" / "td3_yearly_test",
        training_dir=BASE_DIR / "results" / "td3_yearly_training",
        training_export_xlsx=BASE_DIR / "results" / "td3_yearly_training" / "td3_training_export.xlsx",
        training_episode_csv=BASE_DIR / "results" / "td3_yearly_training" / "episode_summary.csv",
        eval_npz_path=BASE_DIR / "logs" / "td3_yearly_single" / "evaluations.npz",
    ),
    MethodConfig(
        method="LSTM-TD3",
        test_dir=BASE_DIR / "results" / "lstm_td3_yearly_test",
        training_dir=BASE_DIR / "results" / "lstm_td3_yearly_training",
        training_export_xlsx=BASE_DIR / "results" / "lstm_td3_yearly_training" / "lstm_td3_training_export.xlsx",
        training_episode_csv=BASE_DIR / "results" / "lstm_td3_yearly_training" / "episode_summary.csv",
        eval_npz_path=BASE_DIR / "logs" / "lstm_td3_yearly_single" / "evaluations.npz",
    ),
]


class WarningLog:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def warn(self, message: str) -> None:
        text = f"WARNING: {message}"
        self.messages.append(text)
        print(text, file=sys.stderr)

    def write(self, path: Path) -> None:
        content = VALIDATION_REWARD_NOTE
        if self.messages:
            content += "\n\n" + "\n".join(self.messages)
        path.write_text(content + "\n", encoding="utf-8")


def read_json(path: Path, warning_log: WarningLog, *, label: str) -> dict[str, Any]:
    if not path.exists():
        warning_log.warn(f"{label} not found: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warning_log.warn(f"Failed to read {label}: {path} ({exc})")
        return {}


def display_path(path: Path) -> str:
    try:
        return path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(path)


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.zeros(len(df), dtype=np.float64), index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def numeric_nan(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.full(len(df), np.nan, dtype=np.float64), index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def finite_mean(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else np.nan


def finite_min(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.min()) if len(clean) else np.nan


def finite_sum(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.sum()) if len(clean) else np.nan


def boolean_flag(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.zeros(len(df), dtype=bool), index=df.index)
    values = df[column]
    numeric_values = pd.to_numeric(values, errors="coerce")
    text_values = values.astype(str).str.strip().str.lower()
    return text_values.isin({"true", "t", "yes", "y"}) | (numeric_values.fillna(0.0) > 0.5)


def boolean_flag_nan(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="boolean")
    values = df[column]
    text_values = values.astype(str).str.strip().str.lower()
    numeric_values = pd.to_numeric(values, errors="coerce")
    result = pd.Series([pd.NA] * len(df), index=df.index, dtype="boolean")
    true_mask = text_values.isin({"true", "t", "yes", "y"}) | (numeric_values == 1)
    false_mask = text_values.isin({"false", "f", "no", "n"}) | (numeric_values == 0)
    result[true_mask] = True
    result[false_mask] = False
    return result


def ensure_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    method: str,
    source_name: str,
    warning_log: WarningLog,
) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            warning_log.warn(f"{method} {source_name} missing column {column}; filled with 0.")
            df[column] = 0.0
    return df


def ensure_ees_terminal_columns(
    df: pd.DataFrame,
    *,
    method: str,
    source_name: str,
    warning_log: WarningLog,
) -> pd.DataFrame:
    df = df.copy()
    missing = [column for column in EES_TERMINAL_DAILY_COLUMNS if column not in df.columns]
    critical_missing = [column for column in EES_TERMINAL_CRITICAL_COLUMNS if column not in df.columns]
    if critical_missing:
        warning_log.warn(
            f"{method} {source_name}: EES terminal SOC columns are missing; "
            "the comparison report is incomplete for SCI reporting. "
            f"Missing columns: {critical_missing}"
        )
    elif missing:
        warning_log.warn(
            f"{method} {source_name} missing optional EES terminal SOC columns: {missing}."
        )
    for column in missing:
        df[column] = np.nan
    return df


def load_method_data(config: MethodConfig, warning_log: WarningLog) -> dict[str, Any] | None:
    if not config.test_dir.exists():
        warning_log.warn(f"{config.method} test directory not found, skipped: {config.test_dir}")
        return None

    daily_path = config.test_dir / "daily_summary.csv"
    timeseries_path = config.test_dir / "timeseries_detail.csv"
    runtime_path = config.test_dir / "runtime_summary.json"

    if not daily_path.exists():
        warning_log.warn(f"{config.method} daily_summary.csv not found, skipped: {daily_path}")
        return None

    daily_df = pd.read_csv(daily_path)
    daily_df = ensure_columns(
        daily_df,
        [
            "total_system_cost",
            "total_penalties",
            *ECONOMIC_COST_COLUMNS,
            *[column for column in PENALTY_COLUMNS if column != "total_penalty_terminal_ees_soc"],
        ],
        method=config.method,
        source_name="daily_summary.csv",
        warning_log=warning_log,
    )
    daily_df = ensure_ees_terminal_columns(
        daily_df,
        method=config.method,
        source_name="daily_summary.csv",
        warning_log=warning_log,
    )
    daily_df["total_cost_plus_penalty"] = (
        numeric(daily_df, "total_system_cost") + numeric(daily_df, "total_penalties")
    )

    if timeseries_path.exists():
        timeseries_df = pd.read_csv(timeseries_path)
    else:
        warning_log.warn(f"{config.method} timeseries_detail.csv not found: {timeseries_path}")
        timeseries_df = pd.DataFrame()

    runtime = read_json(runtime_path, warning_log, label=f"{config.method} runtime_summary.json")
    training_runtime = {}
    if config.training_dir is not None:
        training_runtime_path = config.training_dir / "training_runtime_summary.json"
        training_runtime = read_json(
            training_runtime_path,
            warning_log,
            label=f"{config.method} training_runtime_summary.json",
        )

    return {
        "config": config,
        "daily": daily_df,
        "timeseries": timeseries_df,
        "runtime": runtime,
        "training_runtime": training_runtime,
    }


def build_economic_summary(method_data: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in method_data:
        method = item["config"].method
        df = item["daily"]
        row: dict[str, Any] = {
            "method": method,
            "n_days": int(len(df)),
            "mean_total_system_cost": float(numeric(df, "total_system_cost").mean()) if len(df) else np.nan,
            "sum_total_system_cost": float(numeric(df, "total_system_cost").sum()),
            "mean_total_cost_grid": float(numeric(df, "total_cost_grid").mean()) if len(df) else np.nan,
            "sum_total_cost_grid": float(numeric(df, "total_cost_grid").sum()),
            "mean_total_cost_gas": float(numeric(df, "total_cost_gas").mean()) if len(df) else np.nan,
            "sum_total_cost_gas": float(numeric(df, "total_cost_gas").sum()),
            "mean_total_cost_om": float(numeric(df, "total_cost_om").mean()) if len(df) else np.nan,
            "sum_total_cost_om": float(numeric(df, "total_cost_om").sum()),
            "mean_total_cost_deg": float(numeric(df, "total_cost_deg").mean()) if len(df) else np.nan,
            "sum_total_cost_deg": float(numeric(df, "total_cost_deg").sum()),
            "mean_total_cost_plus_penalty": float(numeric(df, "total_cost_plus_penalty").mean()) if len(df) else np.nan,
            "sum_total_cost_plus_penalty": float(numeric(df, "total_cost_plus_penalty").sum()),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return pd.DataFrame(columns=ECONOMIC_OUTPUT_COLUMNS)

    baseline_rows = summary[summary["method"] == "Rule-based-V2G"]
    baseline = (
        float(baseline_rows["mean_total_cost_plus_penalty"].iloc[0])
        if not baseline_rows.empty
        else np.nan
    )
    if not np.isfinite(baseline) or abs(baseline) <= EPS:
        summary["cost_reduction_vs_rule_based_pct"] = np.nan
    else:
        summary["cost_reduction_vs_rule_based_pct"] = (
            (baseline - summary["mean_total_cost_plus_penalty"]) / baseline * 100.0
        )
    return summary.reindex(columns=ECONOMIC_OUTPUT_COLUMNS)


def build_penalty_breakdown(method_data: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in method_data:
        method = item["config"].method
        df = item["daily"]
        row: dict[str, Any] = {
            "method": method,
            "mean_total_penalties": float(numeric(df, "total_penalties").mean()) if len(df) else np.nan,
        }
        for column in PENALTY_COLUMNS:
            values = (
                numeric_nan(df, column)
                if column == "total_penalty_terminal_ees_soc"
                else numeric(df, column)
            )
            row[f"mean_{column}"] = finite_mean(values) if len(df) else np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=PENALTY_OUTPUT_COLUMNS)
    return pd.DataFrame(rows).reindex(columns=PENALTY_OUTPUT_COLUMNS)


def build_performance_summary(method_data: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in method_data:
        config: MethodConfig = item["config"]
        df = item["daily"]
        runtime = item["runtime"]
        training_runtime = item["training_runtime"]
        timeseries_df = item["timeseries"]

        n_days = int(runtime.get("n_days", len(df)))
        n_steps = int(runtime.get("n_steps", len(timeseries_df)))
        final_ees_soc = numeric_nan(df, "final_ees_soc")
        terminal_shortage = numeric_nan(df, "terminal_ees_shortage_kwh")
        terminal_penalty = numeric_nan(df, "total_penalty_terminal_ees_soc")
        feasible_flags = boolean_flag_nan(df, "ees_terminal_soc_feasible")
        feasible_days = (
            int(feasible_flags.fillna(False).sum())
            if len(df) and bool(feasible_flags.notna().all())
            else np.nan
        )
        rows.append(
            {
                "method": config.method,
                "n_days": n_days,
                "n_steps": n_steps,
                "avg_daily_system_cost": float(numeric(df, "total_system_cost").mean()) if len(df) else np.nan,
                "avg_daily_penalty": float(numeric(df, "total_penalties").mean()) if len(df) else np.nan,
                "avg_daily_cost_plus_penalty": float(numeric(df, "total_cost_plus_penalty").mean()) if len(df) else np.nan,
                "training_duration_seconds": training_runtime.get("training_duration_seconds", np.nan),
                "training_duration_hms": training_runtime.get("training_duration_hms", ""),
                "test_duration_seconds": runtime.get("test_duration_seconds", np.nan),
                "time_per_day_seconds": runtime.get("time_per_day_seconds", np.nan),
                "time_per_step_seconds": runtime.get("time_per_step_seconds", np.nan),
                "mean_final_ees_soc": finite_mean(final_ees_soc) if len(df) else np.nan,
                "min_final_ees_soc": finite_min(final_ees_soc) if len(df) else np.nan,
                "terminal_ees_feasible_days": feasible_days,
                "terminal_ees_feasible_ratio": float(feasible_days / len(df))
                if len(df) and np.isfinite(feasible_days)
                else np.nan,
                "sum_terminal_ees_shortage_kwh": finite_sum(terminal_shortage),
                "mean_terminal_ees_shortage_kwh": finite_mean(terminal_shortage) if len(df) else np.nan,
                "sum_penalty_terminal_ees_soc": finite_sum(terminal_penalty),
            }
        )
    if not rows:
        return pd.DataFrame(columns=PERFORMANCE_OUTPUT_COLUMNS)
    return pd.DataFrame(rows).reindex(columns=PERFORMANCE_OUTPUT_COLUMNS)


def compute_ev_exchange(timeseries_df: pd.DataFrame, method: str, warning_log: WarningLog) -> np.ndarray:
    if timeseries_df.empty:
        return np.array([], dtype=np.float64)

    df = timeseries_df.copy()
    if "p_ev_dis" not in df.columns:
        warning_log.warn(f"{method} timeseries_detail.csv missing p_ev_dis; filled with 0.")
        df["p_ev_dis"] = 0.0

    if "p_ev_ch" not in df.columns:
        charge_components = ["p_ev_rigid_ch", "p_ev_flex_target_ch", "p_ev_buffer_ch"]
        df = ensure_columns(
            df,
            charge_components,
            method=method,
            source_name="timeseries_detail.csv",
            warning_log=warning_log,
        )
        df["p_ev_ch"] = sum(numeric(df, column) for column in charge_components)

    exchange = numeric(df, "p_ev_dis") - numeric(df, "p_ev_ch")
    return exchange.to_numpy(dtype=np.float64)


def build_exchange_distribution(
    method_data: list[dict[str, Any]],
    warning_log: WarningLog,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    method_values: dict[str, np.ndarray] = {}
    for item in method_data:
        method = item["config"].method
        values = compute_ev_exchange(item["timeseries"], method, warning_log)
        if len(values):
            method_values[method] = values

    if not method_values:
        columns = [
            "method",
            "bin_left",
            "bin_right",
            "probability",
            "mean_exchange_kw",
            "std_exchange_kw",
            "positive_exchange_ratio",
            "negative_exchange_ratio",
            "zero_exchange_ratio",
        ]
        return pd.DataFrame(columns=columns), method_values, np.array([], dtype=np.float64)

    all_values = np.concatenate(list(method_values.values()))
    min_value = float(np.nanmin(all_values))
    max_value = float(np.nanmax(all_values))
    if abs(max_value - min_value) <= EPS:
        min_value -= 1.0
        max_value += 1.0
    bins = np.linspace(min_value, max_value, N_EXCHANGE_BINS + 1)

    rows: list[dict[str, Any]] = []
    for method, values in method_values.items():
        finite_values = values[np.isfinite(values)]
        if not len(finite_values):
            continue
        counts, edges = np.histogram(finite_values, bins=bins)
        probabilities = counts / max(int(counts.sum()), 1)
        positive_ratio = float(np.mean(finite_values > EPS))
        negative_ratio = float(np.mean(finite_values < -EPS))
        zero_ratio = float(np.mean(np.abs(finite_values) <= EPS))
        mean_exchange = float(np.mean(finite_values))
        std_exchange = float(np.std(finite_values))
        for idx, probability in enumerate(probabilities):
            rows.append(
                {
                    "method": method,
                    "bin_left": float(edges[idx]),
                    "bin_right": float(edges[idx + 1]),
                    "probability": float(probability),
                    "mean_exchange_kw": mean_exchange,
                    "std_exchange_kw": std_exchange,
                    "positive_exchange_ratio": positive_ratio,
                    "negative_exchange_ratio": negative_ratio,
                    "zero_exchange_ratio": zero_ratio,
                }
            )
    return pd.DataFrame(rows), method_values, bins


def save_placeholder_figure(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cost_breakdown(economic_df: pd.DataFrame, path: Path) -> None:
    if economic_df.empty:
        save_placeholder_figure(path, "Average Daily Cost Breakdown", "No method data available.")
        return

    methods = economic_df["method"].astype(str).tolist()
    specs = [
        ("mean_total_cost_grid", "Grid", "#7BA6D8"),
        ("mean_total_cost_gas", "Gas", "#F4B548"),
        ("mean_total_cost_om", "O&M", "#8E7C6E"),
        ("mean_total_cost_deg", "Degradation", "#CF8DD9"),
    ]
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods), dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for column, label, color in specs:
        values = pd.to_numeric(economic_df[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.set_ylabel("average daily cost")
    ax.set_title("Algorithm Average Daily Cost Breakdown")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_penalty_breakdown(penalty_df: pd.DataFrame, path: Path, *, log_scale: bool = False) -> None:
    title_suffix = " (log scale)" if log_scale else ""
    if penalty_df.empty:
        save_placeholder_figure(path, f"Average Daily Penalty Breakdown{title_suffix}", "No method data available.")
        return

    methods = penalty_df["method"].astype(str).tolist()
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods), dtype=float)
    colors = [
        "#D62728",
        "#FF7F0E",
        "#9467BD",
        "#E15759",
        "#B279A2",
        "#F28E2B",
        "#EDC948",
        "#59A14F",
        "#4E79A7",
        "#76B7B2",
    ]
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    for idx, column in enumerate(PENALTY_COLUMNS):
        mean_col = f"mean_{column}"
        values = pd.to_numeric(penalty_df[mean_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        label = PENALTY_LABELS.get(column, column.replace("total_penalty_", ""))
        ax.bar(x, values, bottom=bottom, label=label, color=colors[idx % len(colors)])
        bottom += values
    if log_scale:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.set_ylabel("average daily penalty")
    ax.set_title(f"Algorithm Average Daily Penalty Breakdown{title_suffix}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def needs_penalty_log_plot(penalty_df: pd.DataFrame) -> bool:
    if penalty_df.empty or "mean_total_penalties" not in penalty_df.columns:
        return False
    values = pd.to_numeric(penalty_df["mean_total_penalties"], errors="coerce").fillna(0.0)
    positive_values = values[values > EPS]
    if len(positive_values) < 2:
        return False
    rule_values = penalty_df.loc[penalty_df["method"] == "Rule-based-V2G", "mean_total_penalties"]
    if rule_values.empty:
        return False
    rule_value = float(rule_values.iloc[0])
    others = positive_values[penalty_df.loc[positive_values.index, "method"] != "Rule-based-V2G"]
    return len(others) > 0 and rule_value > 10.0 * float(others.max())


def plot_exchange_distribution(
    method_values: dict[str, np.ndarray],
    bins: np.ndarray,
    path: Path,
) -> None:
    if not method_values or len(bins) == 0:
        save_placeholder_figure(path, "EV-IES Exchange Distribution", "No timeseries data available.")
        return

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    centers = (bins[:-1] + bins[1:]) / 2.0
    for method, values in method_values.items():
        counts, _ = np.histogram(values[np.isfinite(values)], bins=bins)
        probabilities = counts / max(int(counts.sum()), 1)
        ax.plot(centers, probabilities, marker="o", linewidth=1.7, markersize=3.0, label=method)

    ax.axvline(0.0, color="#222222", linewidth=1.2, linestyle="--")
    ax.text(
        0.02,
        0.95,
        "left side: IES/grid to EV charging",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    ax.text(
        0.98,
        0.95,
        "right side: EV to IES discharging",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    ax.set_xlabel("p_ev_exchange = p_ev_dis - p_ev_ch (kW)")
    ax.set_ylabel("probability")
    ax.set_title("EV-IES Exchange Probability Distribution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def load_training_episode_curve(
    config: MethodConfig,
    warning_log: WarningLog,
    metric_columns: tuple[str, ...],
    *,
    source_label: str,
) -> pd.DataFrame:
    if config.method == "Rule-based-V2G" or config.training_dir is None:
        return pd.DataFrame()

    candidates: list[tuple[str, Path]] = []
    if config.training_export_xlsx is not None:
        candidates.append(("xlsx", config.training_export_xlsx))
    if config.training_episode_csv is not None:
        candidates.append(("csv", config.training_episode_csv))

    for kind, path in candidates:
        if not path.exists():
            continue
        try:
            if kind == "xlsx":
                df = pd.read_excel(path, sheet_name="episode_summary")
            else:
                df = pd.read_csv(path)
        except Exception as exc:
            warning_log.warn(f"Failed to read {config.method} {source_label} from {path}: {exc}")
            continue

        missing = [column for column in metric_columns if column not in df.columns]
        if missing:
            warning_log.warn(f"{config.method} {source_label} file missing columns {missing}: {path}")
            continue

        out = df[list(metric_columns)].copy()
        if "episode_idx" in df.columns:
            out["episode_number"] = pd.to_numeric(df["episode_idx"], errors="coerce") + 1
        else:
            out["episode_number"] = np.arange(1, len(out) + 1, dtype=np.int64)
        out["method"] = config.method
        for column in metric_columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        return out.dropna(subset=["episode_number", *metric_columns])

    warning_log.warn(f"{config.method} {source_label} file not found under {config.training_dir}")
    return pd.DataFrame()


def load_training_reward_curve(config: MethodConfig, warning_log: WarningLog) -> pd.DataFrame:
    return load_training_episode_curve(
        config,
        warning_log,
        ("episode_reward_scaled",),
        source_label="training reward",
    )


def load_training_process_curve(config: MethodConfig, warning_log: WarningLog) -> pd.DataFrame:
    return load_training_episode_curve(
        config,
        warning_log,
        ("episode_system_cost", "episode_penalty_cost"),
        source_label="training process",
    )


def load_validation_reward_curve(config: MethodConfig, warning_log: WarningLog) -> pd.DataFrame:
    if config.eval_npz_path is None:
        return pd.DataFrame(columns=VALIDATION_REWARD_OUTPUT_COLUMNS)

    path = config.eval_npz_path
    if not path.exists():
        warning_log.warn(
            f"{config.method} evaluations.npz not found at {display_path(path)}, "
            "validation reward skipped."
        )
        return pd.DataFrame(columns=VALIDATION_REWARD_OUTPUT_COLUMNS)

    try:
        with np.load(path) as data:
            if "timesteps" not in data or "results" not in data:
                warning_log.warn(
                    f"{config.method} evaluations.npz missing timesteps or results at "
                    f"{display_path(path)}, validation reward skipped."
                )
                return pd.DataFrame(columns=VALIDATION_REWARD_OUTPUT_COLUMNS)
            timesteps = np.asarray(data["timesteps"], dtype=np.int64)
            results = np.asarray(data["results"], dtype=np.float64)
    except Exception as exc:
        warning_log.warn(
            f"Failed to read {config.method} evaluations.npz at {display_path(path)}: "
            f"{exc}; validation reward skipped."
        )
        return pd.DataFrame(columns=VALIDATION_REWARD_OUTPUT_COLUMNS)

    if results.ndim != 2:
        warning_log.warn(
            f"{config.method} evaluations.npz results must be 2D, got shape "
            f"{results.shape}; validation reward skipped."
        )
        return pd.DataFrame(columns=VALIDATION_REWARD_OUTPUT_COLUMNS)
    if len(timesteps) != results.shape[0]:
        warning_log.warn(
            f"{config.method} evaluations.npz timesteps length {len(timesteps)} does "
            f"not match results rows {results.shape[0]}; validation reward skipped."
        )
        return pd.DataFrame(columns=VALIDATION_REWARD_OUTPUT_COLUMNS)

    validation_mean_reward = np.mean(results, axis=1)
    validation_std_reward = np.std(results, axis=1)
    out = pd.DataFrame(
        {
            "method": config.method,
            "eval_timestep": timesteps,
            "validation_mean_reward": validation_mean_reward,
            "validation_std_reward": validation_std_reward,
            "n_eval_episodes": int(results.shape[1]),
        }
    )
    return out.dropna(subset=["eval_timestep", "validation_mean_reward"]).reindex(
        columns=VALIDATION_REWARD_OUTPUT_COLUMNS
    )


def plot_training_rewards(configs: list[MethodConfig], path: Path, warning_log: WarningLog) -> None:
    curves = [load_training_reward_curve(config, warning_log) for config in configs]
    curves = [curve for curve in curves if not curve.empty]
    if not curves:
        save_placeholder_figure(path, "Training Rewards", "No training reward curves available.")
        return

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    for curve in curves:
        curve = curve.sort_values("episode_number")
        method = str(curve["method"].iloc[0])
        ax.plot(
            curve["episode_number"].to_numpy(dtype=float),
            curve["episode_reward_scaled"].to_numpy(dtype=float),
            linewidth=1.5,
            label=method,
        )
    ax.set_xlabel("episode")
    ax.set_ylabel("episode_reward_scaled")
    ax.set_title("Training Reward Curves")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def draw_no_data(ax: plt.Axes, title: str, xlabel: str, ylabel: str, message: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, alpha=0.15)


def plot_training_process_comparison(
    configs: list[MethodConfig],
    validation_df: pd.DataFrame,
    path: Path,
    warning_log: WarningLog,
) -> None:
    training_curves = [load_training_process_curve(config, warning_log) for config in configs]
    training_curves = [curve for curve in training_curves if not curve.empty]

    if not training_curves and validation_df.empty:
        save_placeholder_figure(
            path,
            "Training Process Comparison",
            "No training process or validation reward data available.",
        )
        return

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2))
    process_specs = [
        (axes[0], "episode_system_cost", "(a) Episode system cost", "episode", "system cost"),
        (axes[1], "episode_penalty_cost", "(b) Episode penalty cost", "episode", "penalty cost"),
    ]

    for ax, column, title, xlabel, ylabel in process_specs:
        plotted = False
        for curve in training_curves:
            curve = curve.sort_values("episode_number")
            method = str(curve["method"].iloc[0])
            ax.plot(
                curve["episode_number"].to_numpy(dtype=float),
                curve[column].to_numpy(dtype=float),
                linewidth=1.2,
                label=method,
                color=METHOD_COLORS.get(method),
            )
            plotted = True
        if plotted:
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend()
        else:
            draw_no_data(ax, title, xlabel, ylabel, "No training episode data")

    ax = axes[2]
    if validation_df.empty:
        draw_no_data(ax, "(c) Validation reward", "timestep", "mean validation reward", "No validation reward data")
    else:
        for config in configs:
            if config.eval_npz_path is None:
                continue
            method_df = validation_df[validation_df["method"] == config.method]
            if method_df.empty:
                continue
            method_df = method_df.sort_values("eval_timestep")
            x = pd.to_numeric(method_df["eval_timestep"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(method_df["validation_mean_reward"], errors="coerce").to_numpy(dtype=float)
            std = pd.to_numeric(method_df.get("validation_std_reward"), errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if not np.any(valid):
                continue
            x = x[valid]
            y = y[valid]
            std = std[valid] if len(std) == len(valid) else np.full_like(y, np.nan)
            color = METHOD_COLORS.get(config.method)
            ax.plot(x, y, linewidth=1.8, label=config.method, color=color)
            std_valid = np.isfinite(std)
            if np.any(std_valid):
                ax.fill_between(
                    x[std_valid],
                    y[std_valid] - std[std_valid],
                    y[std_valid] + std[std_valid],
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
        ax.set_title("(c) Validation reward")
        ax.set_xlabel("timestep")
        ax.set_ylabel("mean validation reward")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    warning_log = WarningLog()

    loaded = []
    for config in METHODS:
        item = load_method_data(config, warning_log)
        if item is not None:
            loaded.append(item)

    economic_df = build_economic_summary(loaded)
    penalty_df = build_penalty_breakdown(loaded)
    performance_df = build_performance_summary(loaded)
    exchange_df, method_values, exchange_bins = build_exchange_distribution(loaded, warning_log)
    validation_curves = [load_validation_reward_curve(config, warning_log) for config in METHODS]
    validation_curves = [curve for curve in validation_curves if not curve.empty]
    if validation_curves:
        validation_df = pd.concat(validation_curves, ignore_index=True).reindex(
            columns=VALIDATION_REWARD_OUTPUT_COLUMNS
        )
    else:
        warning_log.warn("No validation reward data found; validation_reward_summary.csv will be empty.")
        validation_df = pd.DataFrame(columns=VALIDATION_REWARD_OUTPUT_COLUMNS)

    economic_df.to_csv(OUTPUT_DIR / "algorithm_economic_summary.csv", index=False, encoding="utf-8-sig")
    penalty_df.to_csv(OUTPUT_DIR / "algorithm_penalty_breakdown.csv", index=False, encoding="utf-8-sig")
    performance_df.to_csv(OUTPUT_DIR / "algorithm_performance_summary.csv", index=False, encoding="utf-8-sig")
    exchange_df.to_csv(OUTPUT_DIR / "ev_ies_exchange_distribution.csv", index=False, encoding="utf-8-sig")
    validation_df.to_csv(OUTPUT_DIR / "validation_reward_summary.csv", index=False, encoding="utf-8-sig")

    plot_cost_breakdown(economic_df, OUTPUT_DIR / "fig_cost_breakdown.png")
    plot_penalty_breakdown(penalty_df, OUTPUT_DIR / "fig_penalty_breakdown.png")
    if needs_penalty_log_plot(penalty_df):
        plot_penalty_breakdown(penalty_df, OUTPUT_DIR / "fig_penalty_breakdown_log.png", log_scale=True)
    plot_exchange_distribution(method_values, exchange_bins, OUTPUT_DIR / "fig_ev_ies_exchange_distribution.png")
    plot_training_rewards(METHODS, OUTPUT_DIR / "fig_training_rewards.png", warning_log)
    plot_training_process_comparison(
        METHODS,
        validation_df,
        OUTPUT_DIR / "fig_training_process_comparison.png",
        warning_log,
    )

    warning_log.write(OUTPUT_DIR / "comparison_warnings.txt")

    print(f"Comparison outputs saved to: {OUTPUT_DIR}")
    print(f"Loaded methods: {', '.join(item['config'].method for item in loaded) if loaded else 'none'}")
    print(f"Warnings: {len(warning_log.messages)}")


if __name__ == "__main__":
    main()
