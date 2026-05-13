from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


OUTPUT_ROOT = Path("results/v2g_ratio_sensitivity")
DEFAULT_SEEDS = [42]
BASE_RATIO = 0.279

SEED_SUMMARY_COLUMNS = [
    "seed",
    "run_name",
    "ratio_dir",
    "v2g_ratio_target",
    "v2g_ratio_actual",
    "n_v2g_ev",
    "n_total_ev",
    "n_days",
    "mean_total_system_cost",
    "mean_total_penalties",
    "mean_total_cost_plus_penalty",
    "mean_total_grid_buy_kwh",
    "mean_total_grid_sell_kwh",
    "mean_total_ev_charge_kwh",
    "mean_total_ev_discharge_kwh",
    "mean_total_ev_buffer_charge_kwh",
    "mean_total_storage_peak_shaved_kwh",
    "sum_depart_energy_shortage_kwh",
    "mean_depart_energy_shortage_kwh",
    "terminal_ees_feasible_days",
    "terminal_ees_feasible_ratio",
    "sum_terminal_ees_shortage_kwh",
    "mean_terminal_ees_shortage_kwh",
    "sum_penalty_terminal_ees_soc",
    "sum_penalty_ev_export_guard",
    "sum_ev_export_overlap_kwh",
    "mean_ev_export_overlap_kwh",
    "model_path",
    "daily_summary_path",
    "timeseries_detail_path",
    "runtime_summary_path",
]

AGGREGATE_METRICS = [
    "v2g_ratio_actual",
    "n_v2g_ev",
    "n_total_ev",
    "mean_total_system_cost",
    "mean_total_penalties",
    "mean_total_cost_plus_penalty",
    "mean_total_grid_buy_kwh",
    "mean_total_grid_sell_kwh",
    "mean_total_ev_charge_kwh",
    "mean_total_ev_discharge_kwh",
    "mean_total_ev_buffer_charge_kwh",
    "mean_total_storage_peak_shaved_kwh",
    "sum_depart_energy_shortage_kwh",
    "mean_depart_energy_shortage_kwh",
    "terminal_ees_feasible_days",
    "terminal_ees_feasible_ratio",
    "sum_terminal_ees_shortage_kwh",
    "mean_terminal_ees_shortage_kwh",
    "sum_penalty_terminal_ees_soc",
    "sum_penalty_ev_export_guard",
    "sum_ev_export_overlap_kwh",
    "mean_ev_export_overlap_kwh",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V2G ratio sensitivity summaries.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


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
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_ratio_from_dir(path: Path) -> float:
    token = path.name.replace("ratio_", "")
    if token == "0279":
        return BASE_RATIO
    try:
        return float(int(token)) / 100.0
    except ValueError:
        return np.nan


def run_dirs_for_seed(output_root: Path, seed: int) -> list[Path]:
    run_dir = output_root / f"seed_{int(seed)}"
    if not run_dir.exists():
        print(f"WARNING: missing run directory: {run_dir}")
        return []
    return sorted([path for path in run_dir.glob("ratio_*") if path.is_dir()])


def load_timeseries_daily_metrics(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["case_index"])
    df = pd.read_csv(path)
    if "case_index" not in df.columns:
        return pd.DataFrame(columns=["case_index"])
    grouped = df.groupby("case_index", sort=True)
    rows = pd.DataFrame({"case_index": sorted(df["case_index"].unique())})
    specs = {
        "total_ev_charge_kwh": "p_ev_ch",
        "total_ev_discharge_kwh": "p_ev_dis",
        "total_ev_export_overlap_kwh": "ev_export_overlap_kwh",
    }
    for out_col, source_col in specs.items():
        if source_col in df.columns:
            rows[out_col] = grouped[source_col].sum().reset_index(drop=True)
        else:
            rows[out_col] = 0.0
    return rows


def ensure_daily_metrics(daily: pd.DataFrame, timeseries_path: Path | None) -> pd.DataFrame:
    daily = daily.copy()
    ts_metrics = load_timeseries_daily_metrics(timeseries_path)
    if not ts_metrics.empty and "case_index" in daily.columns:
        daily = daily.merge(ts_metrics, on="case_index", how="left", suffixes=("", "_from_ts"))
        for column in (
            "total_ev_charge_kwh",
            "total_ev_discharge_kwh",
            "total_ev_export_overlap_kwh",
        ):
            from_ts = f"{column}_from_ts"
            if column not in daily.columns and from_ts in daily.columns:
                daily[column] = daily[from_ts]
            elif column in daily.columns and from_ts in daily.columns:
                daily[column] = pd.to_numeric(daily[column], errors="coerce").fillna(
                    pd.to_numeric(daily[from_ts], errors="coerce")
                )
    total_system_cost = numeric_column(daily, "total_system_cost")
    total_penalties = numeric_column(daily, "total_penalties")
    if "total_cost_plus_penalty" not in daily.columns:
        daily["total_cost_plus_penalty"] = total_system_cost + total_penalties
    return daily


def build_seed_row(seed: int, ratio_dir: Path) -> dict[str, Any] | None:
    daily_path = ratio_dir / "daily_summary.csv"
    timeseries_path = ratio_dir / "timeseries_detail.csv"
    runtime_path = ratio_dir / "runtime_summary.json"
    if not daily_path.exists():
        print(f"WARNING: missing daily summary: {daily_path}")
        return None

    runtime = load_json(runtime_path)
    daily = pd.read_csv(daily_path)
    if daily.empty:
        raise ValueError(f"Empty daily summary: {daily_path}")
    daily = ensure_daily_metrics(daily, timeseries_path if timeseries_path.exists() else None)

    n_days = int(len(daily))
    feasible = bool_column(daily, "ees_terminal_soc_feasible", default=False)
    feasible_days = int(feasible.sum())
    ratio_target = float(runtime.get("ratio", parse_ratio_from_dir(ratio_dir)))
    ratio_actual = float(
        runtime.get(
            "actual_v2g_ratio",
            numeric_column(daily, "v2g_ratio_actual", ratio_target).iloc[0],
        )
    )
    n_v2g = int(
        runtime.get(
            "actual_v2g_ev_count",
            numeric_column(daily, "n_v2g_ev", np.nan).dropna().iloc[0]
            if not numeric_column(daily, "n_v2g_ev", np.nan).dropna().empty
            else round(ratio_actual * float(runtime.get("total_ev_count", 0))),
        )
    )
    n_total = int(
        runtime.get(
            "total_ev_count",
            numeric_column(daily, "n_total_ev", np.nan).dropna().iloc[0]
            if not numeric_column(daily, "n_total_ev", np.nan).dropna().empty
            else 0,
        )
    )

    return {
        "seed": int(seed),
        "run_name": runtime.get("run_name", f"seed_{int(seed)}"),
        "ratio_dir": ratio_dir.name,
        "v2g_ratio_target": ratio_target,
        "v2g_ratio_actual": ratio_actual,
        "n_v2g_ev": n_v2g,
        "n_total_ev": n_total,
        "n_days": n_days,
        "mean_total_system_cost": float(numeric_column(daily, "total_system_cost").mean()),
        "mean_total_penalties": float(numeric_column(daily, "total_penalties").mean()),
        "mean_total_cost_plus_penalty": float(
            numeric_column(daily, "total_cost_plus_penalty").mean()
        ),
        "mean_total_grid_buy_kwh": float(numeric_column(daily, "total_grid_buy_kwh").mean()),
        "mean_total_grid_sell_kwh": float(numeric_column(daily, "total_grid_sell_kwh").mean()),
        "mean_total_ev_charge_kwh": float(numeric_column(daily, "total_ev_charge_kwh").mean()),
        "mean_total_ev_discharge_kwh": float(
            numeric_column(daily, "total_ev_discharge_kwh").mean()
        ),
        "mean_total_ev_buffer_charge_kwh": float(
            numeric_column(daily, "total_ev_buffer_charge_kwh").mean()
        ),
        "mean_total_storage_peak_shaved_kwh": float(
            numeric_column(daily, "total_storage_peak_shaved_kwh").mean()
        ),
        "sum_depart_energy_shortage_kwh": float(
            numeric_column(daily, "total_depart_energy_shortage_kwh").sum()
        ),
        "mean_depart_energy_shortage_kwh": float(
            numeric_column(daily, "total_depart_energy_shortage_kwh").mean()
        ),
        "terminal_ees_feasible_days": feasible_days,
        "terminal_ees_feasible_ratio": float(feasible_days / n_days) if n_days else 0.0,
        "sum_terminal_ees_shortage_kwh": float(
            numeric_column(daily, "terminal_ees_shortage_kwh").sum()
        ),
        "mean_terminal_ees_shortage_kwh": float(
            numeric_column(daily, "terminal_ees_shortage_kwh").mean()
        ),
        "sum_penalty_terminal_ees_soc": float(
            numeric_column(daily, "total_penalty_terminal_ees_soc").sum()
        ),
        "sum_penalty_ev_export_guard": float(
            numeric_column(daily, "total_penalty_ev_export_guard").sum()
        ),
        "sum_ev_export_overlap_kwh": float(
            numeric_column(daily, "total_ev_export_overlap_kwh").sum()
        ),
        "mean_ev_export_overlap_kwh": float(
            numeric_column(daily, "total_ev_export_overlap_kwh").mean()
        ),
        "model_path": runtime.get("model_path", ""),
        "daily_summary_path": str(daily_path),
        "timeseries_detail_path": str(timeseries_path) if timeseries_path.exists() else "",
        "runtime_summary_path": str(runtime_path) if runtime_path.exists() else "",
    }


def build_aggregate_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = seed_df.groupby("v2g_ratio_target", sort=True)
    for ratio, group in grouped:
        row: dict[str, Any] = {
            "v2g_ratio_target": float(ratio),
            "ratio_label": label_ratio(float(ratio)),
            "n_seeds": int(group["seed"].nunique()),
            "seeds": ",".join(str(int(seed)) for seed in sorted(group["seed"].unique())),
        }
        for metric in AGGREGATE_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.count() > 1 else 0.0
        row["cost_plus_penalty_mean_std"] = format_mean_std(
            row["mean_total_cost_plus_penalty_mean"],
            row["mean_total_cost_plus_penalty_std"],
        )
        row["grid_buy_mean_std"] = format_mean_std(
            row["mean_total_grid_buy_kwh_mean"],
            row["mean_total_grid_buy_kwh_std"],
        )
        row["ev_discharge_mean_std"] = format_mean_std(
            row["mean_total_ev_discharge_kwh_mean"],
            row["mean_total_ev_discharge_kwh_std"],
        )
        row["terminal_feasible_ratio_mean_std"] = format_mean_std(
            row["terminal_ees_feasible_ratio_mean"],
            row["terminal_ees_feasible_ratio_std"],
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("v2g_ratio_target").reset_index(drop=True)


def format_mean_std(mean: float, std: float) -> str:
    return f"{float(mean):.6g} +/- {float(std):.6g}"


def label_ratio(ratio: float) -> str:
    if abs(float(ratio) - BASE_RATIO) <= 5e-4:
        return "base 27.9%"
    return f"{float(ratio) * 100:.1f}%"


def markdown_table(df: pd.DataFrame, columns: list[str], float_digits: int = 4) -> str:
    if df.empty:
        return "(no rows)"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{float_digits}f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def nearest_ratio_row(df: pd.DataFrame, ratio: float) -> pd.Series | None:
    if df.empty:
        return None
    idx = (pd.to_numeric(df["v2g_ratio_target"], errors="coerce") - ratio).abs().idxmin()
    return df.loc[idx]


def is_nonincreasing(values: Iterable[float], tol: float = 1e-9) -> bool:
    seq = list(values)
    return all(seq[i] <= seq[i - 1] + tol for i in range(1, len(seq)))


def is_nondecreasing(values: Iterable[float], tol: float = 1e-9) -> bool:
    seq = list(values)
    return all(seq[i] + tol >= seq[i - 1] for i in range(1, len(seq)))


def write_summary_markdown(seed_df: pd.DataFrame, aggregate_df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# V2G Ratio Sensitivity Summary",
        "",
        "## Seed Summary",
        "",
        markdown_table(
            seed_df,
            [
                "seed",
                "v2g_ratio_target",
                "v2g_ratio_actual",
                "n_v2g_ev",
                "n_total_ev",
                "mean_total_cost_plus_penalty",
                "mean_total_ev_discharge_kwh",
                "terminal_ees_feasible_ratio",
            ],
            float_digits=6,
        ),
        "",
        "## Aggregate Summary",
        "",
        markdown_table(
            aggregate_df,
            [
                "v2g_ratio_target",
                "ratio_label",
                "n_seeds",
                "cost_plus_penalty_mean_std",
                "grid_buy_mean_std",
                "ev_discharge_mean_std",
                "terminal_feasible_ratio_mean_std",
            ],
            float_digits=6,
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(seed_df: pd.DataFrame, aggregate_df: pd.DataFrame, path: Path) -> None:
    plot_df = aggregate_df.copy().sort_values("v2g_ratio_target").reset_index(drop=True)
    base_row = nearest_ratio_row(plot_df, BASE_RATIO)
    best_idx = plot_df["mean_total_cost_plus_penalty_mean"].astype(float).idxmin()
    best_row = plot_df.loc[best_idx]

    ratios = plot_df["v2g_ratio_target"].astype(float).to_list()
    cost_values = plot_df["mean_total_cost_plus_penalty_mean"].astype(float).to_list()
    ev_dis_values = plot_df["mean_total_ev_discharge_kwh_mean"].astype(float).to_list()
    grid_buy_values = plot_df["mean_total_grid_buy_kwh_mean"].astype(float).to_list()
    grid_sell_values = plot_df["mean_total_grid_sell_kwh_mean"].astype(float).to_list()
    peak_values = plot_df["mean_total_storage_peak_shaved_kwh_mean"].astype(float).to_list()

    cost_direction = "decreased monotonically" if is_nonincreasing(cost_values) else "did not decrease monotonically"
    ev_dis_direction = "increased monotonically" if is_nondecreasing(ev_dis_values) else "did not increase monotonically"
    grid_buy_direction = "decreased monotonically" if is_nonincreasing(grid_buy_values) else "did not decrease monotonically"
    grid_sell_direction = "increased monotonically" if is_nondecreasing(grid_sell_values) else "did not increase monotonically"
    peak_direction = "increased monotonically" if is_nondecreasing(peak_values) else "did not increase monotonically"

    marginal = np.diff(cost_values) * -1.0
    diminishing = bool(len(marginal) >= 2 and np.nanmean(marginal[1:]) < np.nanmean(marginal[:-1]))
    diminishing_text = (
        "The average marginal cost+penalty reduction became smaller at higher ratios."
        if diminishing
        else "A clear diminishing-return pattern is not established by these rows."
    )

    if base_row is not None:
        base_cost = float(base_row["mean_total_cost_plus_penalty_mean"])
        base_actual = float(base_row["v2g_ratio_actual_mean"])
        base_position = (
            f"The base 27.9% row has mean cost+penalty {base_cost:.6f}; "
            f"the best row in this sensitivity is {label_ratio(float(best_row['v2g_ratio_target']))} "
            f"with {float(best_row['mean_total_cost_plus_penalty_mean']):.6f}."
        )
        shortage_delta = float(best_row["sum_depart_energy_shortage_kwh_mean"]) - float(
            base_row["sum_depart_energy_shortage_kwh_mean"]
        )
        terminal_delta = float(best_row["sum_terminal_ees_shortage_kwh_mean"]) - float(
            base_row["sum_terminal_ees_shortage_kwh_mean"]
        )
        export_delta = float(best_row["sum_ev_export_overlap_kwh_mean"]) - float(
            base_row["sum_ev_export_overlap_kwh_mean"]
        )
    else:
        base_actual = np.nan
        base_position = "The base 27.9% row is missing from the generated summary."
        shortage_delta = terminal_delta = export_delta = np.nan

    constraints_text = (
        f"Relative to the base row, best-row deltas are: EV departure shortage {shortage_delta:.6f} kWh, "
        f"EES terminal shortage {terminal_delta:.6f} kWh, EV export overlap {export_delta:.6f} kWh."
    )

    main_ratio_text = (
        "The 27.9% base case should remain the main experiment scenario. "
        "Even if another sensitivity row has lower cost+penalty, that is a robustness/marginal-benefit finding, "
        "not a replacement for the park-composition base case without a separate experimental-design decision."
    )

    lines = [
        "# V2G Ratio Sensitivity Report",
        "",
        "27.9% is the current park EV-composition base scenario, not a globally optimal participation ratio.",
        "V2G ratio sensitivity is used to test robustness and marginal benefit, not to replace the base case.",
        "The main algorithm comparison and EV-mode comparison remain based on the base 27.9% scenario.",
        "",
        "## Required Questions",
        "",
        f"1. Current base V2G participation rate: {base_actual:.6f} actual ratio from the generated base row.",
        f"2. Cost+penalty trend: {cost_direction} across ratios {', '.join(f'{r:.3f}' for r in ratios)}.",
        f"3. Diminishing return: {diminishing_text}",
        f"4. EV discharge trend: {ev_dis_direction}.",
        f"5. Grid buy trend: {grid_buy_direction}.",
        f"6. Grid sell trend: {grid_sell_direction}.",
        f"7. Storage peak shaved trend: {peak_direction}.",
        f"8. EV departure shortage: total shortage by row is shown below; do not hide nonzero shortage.",
        f"9. EES terminal SOC feasible ratio: reported below and kept visible even when feasible.",
        f"10. EV export guard/export overlap: reported below; {constraints_text}",
        f"11. Base-case position: {base_position}",
        f"12. Main-ratio decision: {main_ratio_text}",
        "",
        "## Aggregate Table",
        "",
        markdown_table(
            plot_df,
            [
                "v2g_ratio_target",
                "ratio_label",
                "n_seeds",
                "mean_total_cost_plus_penalty_mean",
                "mean_total_ev_discharge_kwh_mean",
                "mean_total_grid_buy_kwh_mean",
                "mean_total_grid_sell_kwh_mean",
                "mean_total_storage_peak_shaved_kwh_mean",
                "terminal_ees_feasible_ratio_mean",
                "sum_depart_energy_shortage_kwh_mean",
                "sum_terminal_ees_shortage_kwh_mean",
                "sum_ev_export_overlap_kwh_mean",
            ],
            float_digits=6,
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    seeds = [int(seed) for seed in args.seeds]
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        for ratio_dir in run_dirs_for_seed(output_root, seed):
            row = build_seed_row(seed, ratio_dir)
            if row is not None:
                rows.append(row)

    if not rows:
        raise SystemExit("No V2G ratio sensitivity daily summaries were found.")

    seed_df = pd.DataFrame(rows, columns=SEED_SUMMARY_COLUMNS)
    seed_df = seed_df.sort_values(["seed", "v2g_ratio_target"]).reset_index(drop=True)
    aggregate_df = build_aggregate_summary(seed_df)

    seed_path = output_root / "v2g_ratio_seed_summary.csv"
    summary_path = output_root / "v2g_ratio_summary.csv"
    summary_md_path = output_root / "v2g_ratio_summary.md"
    report_path = output_root / "v2g_ratio_sensitivity_report.md"

    seed_df.to_csv(seed_path, index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_summary_markdown(seed_df, aggregate_df, summary_md_path)
    write_report(seed_df, aggregate_df, report_path)

    print(f"[summary] seed summary: {seed_path}")
    print(f"[summary] aggregate summary: {summary_path}")
    print(f"[summary] markdown summary: {summary_md_path}")
    print(f"[summary] report: {report_path}")
    print(seed_df.to_string(index=False))


if __name__ == "__main__":
    main()
