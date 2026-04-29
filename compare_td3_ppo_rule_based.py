from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TD3_SUMMARY = BASE_DIR / "results" / "td3_yearly_test" / "daily_summary.csv"
DEFAULT_PPO_SUMMARY = BASE_DIR / "results" / "ppo_sb3_direct_test" / "daily_summary.csv"
DEFAULT_RULE_SUMMARY = BASE_DIR / "results" / "rule_based_v2g_test" / "daily_summary.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "td3_ppo_rule_based_test_comparison"

METHOD_ORDER = ["TD3", "PPO", "Rule-based-V2G"]

AGG_METRICS = [
    "total_system_cost",
    "total_penalties",
    "total_cost_plus_penalty",
    "total_grid_buy_kwh",
    "total_grid_sell_kwh",
    "total_depart_energy_shortage_kwh",
    "total_unmet_e",
    "total_unmet_h",
    "total_unmet_c",
    "total_storage_peak_shaved_kwh",
    "total_ev_flex_target_charge_kwh",
    "total_ev_buffer_charge_kwh",
    "final_ees_soc",
    "terminal_ees_shortage_kwh",
    "total_penalty_terminal_ees_soc",
]

PLOT_METRICS = [
    "mean_total_system_cost",
    "mean_total_penalties",
    "mean_total_cost_plus_penalty",
    "mean_total_grid_buy_kwh",
    "mean_total_grid_sell_kwh",
    "mean_total_depart_energy_shortage_kwh",
    "mean_total_storage_peak_shaved_kwh",
    "mean_total_penalty_terminal_ees_soc",
]

EES_TERMINAL_CRITICAL_COLUMNS = [
    "final_ees_soc",
    "terminal_ees_shortage_kwh",
    "total_penalty_terminal_ees_soc",
    "ees_terminal_soc_feasible",
]

REQUIRED_COLUMNS = [
    "case_index",
    "month",
    "day_of_year",
    "season",
    "split",
    "total_system_cost",
    "total_penalties",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare TD3, PPO, and Rule-based test-set daily summaries."
    )
    parser.add_argument("--td3-summary", type=Path, default=DEFAULT_TD3_SUMMARY)
    parser.add_argument("--ppo-summary", type=Path, default=DEFAULT_PPO_SUMMARY)
    parser.add_argument("--rule-summary", type=Path, default=DEFAULT_RULE_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write comparison outputs for available methods instead of failing when a method summary is missing.",
    )
    return parser


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.zeros(len(df), dtype=np.float64), index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def numeric_nan(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.full(len(df), np.nan, dtype=np.float64), index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def bool_flags(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="boolean")
    values = df[column]
    text_values = values.astype(str).str.strip().str.lower()
    numeric_values = pd.to_numeric(values, errors="coerce")
    result = pd.Series([pd.NA] * len(df), index=df.index, dtype="boolean")
    result[text_values.isin({"true", "t", "yes", "y"}) | (numeric_values == 1)] = True
    result[text_values.isin({"false", "f", "no", "n"}) | (numeric_values == 0)] = False
    return result


def load_summary(path: Path, method: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{method} summary not found: {path}")

    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"{method} summary is missing required columns: {missing}")

    df = df.copy()
    df["method"] = method
    df["total_cost_plus_penalty"] = (
        numeric(df, "total_system_cost") + numeric(df, "total_penalties")
    )
    missing_ees = [column for column in EES_TERMINAL_CRITICAL_COLUMNS if column not in df.columns]
    if missing_ees:
        print(
            "WARNING: "
            f"{method} daily_summary.csv: EES terminal SOC columns are missing; "
            "the comparison report is incomplete for SCI reporting. "
            f"Missing columns: {missing_ees}"
        )
        for column in missing_ees:
            df[column] = np.nan
    for column in AGG_METRICS:
        if column not in df.columns:
            df[column] = np.nan if column in EES_TERMINAL_CRITICAL_COLUMNS else 0.0
    return df


def aggregate_method(df: pd.DataFrame, method: str) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "method": method,
        "n_days": int(len(df)),
    }
    for metric in AGG_METRICS:
        values = (
            numeric_nan(df, metric)
            if metric in {"final_ees_soc", "terminal_ees_shortage_kwh", "total_penalty_terminal_ees_soc"}
            else numeric(df, metric)
        )
        clean = values.dropna()
        row[f"mean_{metric}"] = float(clean.mean()) if len(clean) else float("nan")
        row[f"sum_{metric}"] = float(clean.sum()) if len(clean) else float("nan")
    feasible = bool_flags(df, "ees_terminal_soc_feasible")
    if len(feasible) and bool(feasible.notna().all()):
        feasible_days = int(feasible.fillna(False).sum())
        row["terminal_ees_feasible_days"] = feasible_days
        row["terminal_ees_feasible_ratio"] = float(feasible_days / len(df)) if len(df) else float("nan")
    else:
        row["terminal_ees_feasible_days"] = float("nan")
        row["terminal_ees_feasible_ratio"] = float("nan")
    return row


def order_methods(df: pd.DataFrame) -> pd.DataFrame:
    rank = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    return (
        df.assign(_rank=df["method"].map(rank).fillna(len(rank)).astype(int))
        .sort_values(["_rank", "method"])
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )


def build_case_comparison(all_rows: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["case_index", "month", "day_of_year", "season", "split", "method"]
    value_cols = [column for column in AGG_METRICS if column in all_rows.columns]
    compact = all_rows[id_cols + value_cols].copy()
    compact = compact.sort_values(["case_index", "method"]).reset_index(drop=True)
    return compact


def write_report(
    output_path: Path,
    *,
    loaded_methods: list[str],
    missing_methods: list[str],
    method_summary: pd.DataFrame,
) -> None:
    lines = [
        "# TD3 / PPO / Rule-based test comparison",
        "",
        f"- loaded_methods: {', '.join(loaded_methods) if loaded_methods else 'none'}",
        f"- missing_methods: {', '.join(missing_methods) if missing_methods else 'none'}",
        "",
        "Guide reward is not treated as real economic revenue. Focus on system cost, penalties, cost plus penalties, grid exchange, service shortage, EV departure shortage, and storage support.",
        "",
        "## Method Summary",
        "",
    ]

    display_cols = [
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
        "mean_final_ees_soc",
        "mean_terminal_ees_shortage_kwh",
        "sum_terminal_ees_shortage_kwh",
        "mean_total_penalty_terminal_ees_soc",
        "sum_total_penalty_terminal_ees_soc",
        "terminal_ees_feasible_days",
        "terminal_ees_feasible_ratio",
    ]
    existing_display_cols = [col for col in display_cols if col in method_summary.columns]
    if not method_summary.empty:
        lines.extend(markdown_table(method_summary[existing_display_cols]))
    else:
        lines.append("No method summaries were loaded.")

    if missing_methods:
        lines.extend(
            [
                "",
                "## Missing Method Notes",
                "",
            ]
        )
        for method in missing_methods:
            if method == "PPO":
                lines.append(
                    "- PPO is missing. In the current workspace, the existing PPO model expects observation shape 53, while the current environment observation shape is 63. Re-train PPO with the current environment, then run `test_PPO_sb3_direct.py` to generate `results/ppo_sb3_direct_test/daily_summary.csv`."
                )
            else:
                lines.append(f"- {method} summary is missing; run its test script first.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_markdown_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def markdown_table(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["No rows."]
    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(format_markdown_value(row[column]) for column in df.columns)
            + " |"
        )
    return lines


def plot_method_summary(method_summary: pd.DataFrame, output_path: Path) -> bool:
    if plt is None or method_summary.empty:
        return False

    plot_cols = [column for column in PLOT_METRICS if column in method_summary.columns]
    if not plot_cols:
        return False

    fig, axes = plt.subplots(len(plot_cols), 1, figsize=(9.5, 3.0 * len(plot_cols)))
    if len(plot_cols) == 1:
        axes = [axes]

    methods = method_summary["method"].astype(str).tolist()
    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#B07AA1"]
    for ax, column in zip(axes, plot_cols):
        values = pd.to_numeric(method_summary[column], errors="coerce").fillna(0.0)
        ax.bar(methods, values, color=colors[: len(methods)])
        ax.set_title(column)
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{float(value):.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def collect_summaries(paths: dict[str, Path], allow_missing: bool):
    loaded: list[pd.DataFrame] = []
    loaded_methods: list[str] = []
    missing_methods: list[str] = []
    errors: list[str] = []

    for method in METHOD_ORDER:
        path = paths[method]
        try:
            loaded.append(load_summary(path, method))
            loaded_methods.append(method)
        except Exception as exc:
            missing_methods.append(method)
            errors.append(str(exc))
            if not allow_missing:
                raise RuntimeError(
                    "Cannot build a three-method comparison because one or more summaries are unavailable.\n"
                    + "\n".join(errors)
                    + "\nRun missing test scripts first, or pass --allow-missing to compare available methods."
                ) from exc

    return loaded, loaded_methods, missing_methods, errors


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "TD3": args.td3_summary,
        "PPO": args.ppo_summary,
        "Rule-based-V2G": args.rule_summary,
    }
    loaded, loaded_methods, missing_methods, errors = collect_summaries(
        paths, allow_missing=args.allow_missing
    )

    all_rows = pd.concat(loaded, ignore_index=True) if loaded else pd.DataFrame()
    method_summary = (
        pd.DataFrame(
            [
                aggregate_method(all_rows[all_rows["method"] == method], method)
                for method in loaded_methods
            ]
        )
        if loaded_methods
        else pd.DataFrame()
    )
    if not method_summary.empty:
        method_summary = order_methods(method_summary)

    case_comparison = build_case_comparison(all_rows) if not all_rows.empty else pd.DataFrame()

    method_summary_path = args.output_dir / "method_summary.csv"
    case_comparison_path = args.output_dir / "case_level_comparison.csv"
    report_path = args.output_dir / "comparison_report.md"
    plot_path = args.output_dir / "method_summary.png"

    method_summary.to_csv(method_summary_path, index=False, encoding="utf-8-sig")
    case_comparison.to_csv(case_comparison_path, index=False, encoding="utf-8-sig")
    write_report(
        report_path,
        loaded_methods=loaded_methods,
        missing_methods=missing_methods,
        method_summary=method_summary,
    )
    plotted = plot_method_summary(method_summary, plot_path)

    print(f"Method summary saved to: {method_summary_path}")
    print(f"Case-level comparison saved to: {case_comparison_path}")
    print(f"Comparison report saved to: {report_path}")
    if plotted:
        print(f"Method summary plot saved to: {plot_path}")
    if missing_methods:
        print(f"Missing methods: {', '.join(missing_methods)}")
        for error in errors:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
