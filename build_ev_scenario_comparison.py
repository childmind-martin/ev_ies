from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ev_scenario_wrappers import EV_SCENARIOS, SCENARIO_DESCRIPTIONS


SCENARIO_ORDER = ("uncontrolled_charging", "ordered_charging", "v2g")
SCENARIO_LABELS = {
    "uncontrolled_charging": "A uncontrolled_charging",
    "ordered_charging": "B ordered_charging",
    "v2g": "C v2g",
}
DEFAULT_SEEDS = [42]
OUTPUT_DIR = Path("results/ev_scenario_comparison")

SEED_SUMMARY_COLUMNS = [
    "ev_scenario",
    "scenario_label",
    "seed",
    "n_days",
    "mean_total_system_cost",
    "mean_total_penalties",
    "mean_total_cost_plus_penalty",
    "mean_total_grid_buy_kwh",
    "mean_total_grid_sell_kwh",
    "mean_total_ev_charge_kwh",
    "mean_total_ev_discharge_kwh",
    "mean_total_storage_peak_shaved_kwh",
    "total_depart_energy_shortage_kwh",
    "mean_total_depart_energy_shortage_kwh",
    "terminal_ees_feasible_days",
    "terminal_ees_feasible_ratio",
    "sum_terminal_ees_shortage_kwh",
    "mean_terminal_ees_shortage_kwh",
    "mean_final_ees_soc",
    "min_final_ees_soc",
    "mean_total_penalty_depart_energy",
    "mean_total_penalty_depart_risk",
    "mean_total_penalty_ev_export_guard",
    "mean_total_penalty_terminal_ees_soc",
    "mean_other_penalties",
    "sum_penalty_ev_export_guard",
    "sum_ev_export_overlap_kwh",
    "mean_ev_export_overlap_kwh",
    "daily_summary_path",
    "timeseries_detail_path",
]

AGGREGATE_METRICS = [
    "mean_total_system_cost",
    "mean_total_penalties",
    "mean_total_cost_plus_penalty",
    "mean_total_grid_buy_kwh",
    "mean_total_grid_sell_kwh",
    "mean_total_ev_charge_kwh",
    "mean_total_ev_discharge_kwh",
    "mean_total_storage_peak_shaved_kwh",
    "total_depart_energy_shortage_kwh",
    "mean_total_depart_energy_shortage_kwh",
    "terminal_ees_feasible_ratio",
    "sum_terminal_ees_shortage_kwh",
    "mean_terminal_ees_shortage_kwh",
    "mean_final_ees_soc",
    "min_final_ees_soc",
    "mean_total_penalty_ev_export_guard",
    "sum_penalty_ev_export_guard",
    "sum_ev_export_overlap_kwh",
    "mean_ev_export_overlap_kwh",
]

PAIRWISE_COMPARISONS = [
    (
        "ordered_vs_uncontrolled",
        "ordered_charging",
        "uncontrolled_charging",
        "ordered compared with uncontrolled",
    ),
    ("v2g_vs_ordered", "v2g", "ordered_charging", "v2g compared with ordered"),
    (
        "v2g_vs_uncontrolled",
        "v2g",
        "uncontrolled_charging",
        "v2g compared with uncontrolled",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the EV scenario comparison tables.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
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


def first_existing_path(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def scenario_result_candidates(ev_scenario: str, seed: int, filename: str) -> list[Path]:
    if ev_scenario == "v2g":
        candidates = [
            Path(f"results/td3_yearly_test_seed_{seed}") / filename,
        ]
        if int(seed) == 42:
            candidates.append(Path("results/td3_yearly_test") / filename)
        return candidates
    return [Path(f"results/td3_yearly_test_{ev_scenario}_seed_{seed}") / filename]


def load_timeseries_daily_metrics(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["case_index"])

    df = pd.read_csv(path)
    if "case_index" not in df.columns:
        return pd.DataFrame(columns=["case_index"])

    metric_sources = {
        "total_ev_charge_kwh": "p_ev_ch",
        "total_ev_discharge_kwh": "p_ev_dis",
        "total_ev_export_overlap_kwh": "ev_export_overlap_kwh",
    }
    rows = pd.DataFrame({"case_index": sorted(df["case_index"].unique())})
    grouped = df.groupby("case_index", sort=True)
    for out_col, source_col in metric_sources.items():
        if source_col in df.columns:
            rows[out_col] = grouped[source_col].sum().reset_index(drop=True)
        else:
            rows[out_col] = 0.0
    return rows


def build_seed_row(ev_scenario: str, seed: int) -> dict[str, object] | None:
    daily_path = first_existing_path(
        scenario_result_candidates(ev_scenario, seed, "daily_summary.csv")
    )
    timeseries_path = first_existing_path(
        scenario_result_candidates(ev_scenario, seed, "timeseries_detail.csv")
    )
    if daily_path is None:
        print(f"WARNING: missing {ev_scenario} seed={seed} daily_summary.csv")
        return None

    daily = pd.read_csv(daily_path)
    if daily.empty:
        raise ValueError(f"Empty daily summary: {daily_path}")

    ts_metrics = load_timeseries_daily_metrics(timeseries_path)
    if not ts_metrics.empty:
        daily = daily.merge(ts_metrics, on="case_index", how="left")

    n_days = int(len(daily))
    total_system_cost = numeric_column(daily, "total_system_cost")
    total_penalties = numeric_column(daily, "total_penalties")
    if "total_cost_plus_penalty" in daily.columns:
        total_cost_plus_penalty = numeric_column(daily, "total_cost_plus_penalty")
    else:
        total_cost_plus_penalty = total_system_cost + total_penalties

    if "total_ev_charge_kwh" in daily.columns:
        total_ev_charge_kwh = numeric_column(daily, "total_ev_charge_kwh")
    else:
        total_ev_charge_kwh = numeric_column(daily, "avg_p_ev_ch_kw") * 24.0

    if "total_ev_discharge_kwh" in daily.columns:
        total_ev_discharge_kwh = numeric_column(daily, "total_ev_discharge_kwh")
    else:
        total_ev_discharge_kwh = numeric_column(daily, "avg_p_ev_dis_kw") * 24.0

    feasible = bool_column(daily, "ees_terminal_soc_feasible", default=False)
    feasible_days = int(feasible.sum())
    mean_penalty_depart = numeric_column(daily, "total_penalty_depart_energy").mean()
    mean_penalty_depart_risk = numeric_column(daily, "total_penalty_depart_risk").mean()
    mean_penalty_ev_guard = numeric_column(daily, "total_penalty_ev_export_guard").mean()
    mean_penalty_ees_terminal = numeric_column(
        daily, "total_penalty_terminal_ees_soc"
    ).mean()
    mean_other_penalties = float(
        total_penalties.mean()
        - mean_penalty_depart
        - mean_penalty_depart_risk
        - mean_penalty_ev_guard
        - mean_penalty_ees_terminal
    )
    total_ev_export_overlap = numeric_column(daily, "total_ev_export_overlap_kwh").sum()

    return {
        "ev_scenario": ev_scenario,
        "scenario_label": SCENARIO_LABELS[ev_scenario],
        "seed": int(seed),
        "n_days": n_days,
        "mean_total_system_cost": float(total_system_cost.mean()),
        "mean_total_penalties": float(total_penalties.mean()),
        "mean_total_cost_plus_penalty": float(total_cost_plus_penalty.mean()),
        "mean_total_grid_buy_kwh": float(numeric_column(daily, "total_grid_buy_kwh").mean()),
        "mean_total_grid_sell_kwh": float(
            numeric_column(daily, "total_grid_sell_kwh").mean()
        ),
        "mean_total_ev_charge_kwh": float(total_ev_charge_kwh.mean()),
        "mean_total_ev_discharge_kwh": float(total_ev_discharge_kwh.mean()),
        "mean_total_storage_peak_shaved_kwh": float(
            numeric_column(daily, "total_storage_peak_shaved_kwh").mean()
        ),
        "total_depart_energy_shortage_kwh": float(
            numeric_column(daily, "total_depart_energy_shortage_kwh").sum()
        ),
        "mean_total_depart_energy_shortage_kwh": float(
            numeric_column(daily, "total_depart_energy_shortage_kwh").mean()
        ),
        "terminal_ees_feasible_days": feasible_days,
        "terminal_ees_feasible_ratio": float(feasible_days / n_days),
        "sum_terminal_ees_shortage_kwh": float(
            numeric_column(daily, "terminal_ees_shortage_kwh").sum()
        ),
        "mean_terminal_ees_shortage_kwh": float(
            numeric_column(daily, "terminal_ees_shortage_kwh").mean()
        ),
        "mean_final_ees_soc": float(numeric_column(daily, "final_ees_soc", np.nan).mean()),
        "min_final_ees_soc": float(numeric_column(daily, "final_ees_soc", np.nan).min()),
        "mean_total_penalty_depart_energy": float(mean_penalty_depart),
        "mean_total_penalty_depart_risk": float(mean_penalty_depart_risk),
        "mean_total_penalty_ev_export_guard": float(mean_penalty_ev_guard),
        "mean_total_penalty_terminal_ees_soc": float(mean_penalty_ees_terminal),
        "mean_other_penalties": max(float(mean_other_penalties), 0.0),
        "sum_penalty_ev_export_guard": float(
            numeric_column(daily, "total_penalty_ev_export_guard").sum()
        ),
        "sum_ev_export_overlap_kwh": float(total_ev_export_overlap),
        "mean_ev_export_overlap_kwh": float(
            numeric_column(daily, "total_ev_export_overlap_kwh").mean()
        ),
        "daily_summary_path": str(daily_path),
        "timeseries_detail_path": str(timeseries_path) if timeseries_path else "",
    }


def build_aggregate_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ev_scenario in SCENARIO_ORDER:
        group = seed_df[seed_df["ev_scenario"] == ev_scenario]
        if group.empty:
            continue
        row: dict[str, object] = {
            "ev_scenario": ev_scenario,
            "scenario_label": SCENARIO_LABELS[ev_scenario],
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in AGGREGATE_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.count() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_pairwise_reductions(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed, seed_group in seed_df.groupby("seed", sort=True):
        by_scenario = seed_group.set_index("ev_scenario")
        for comparison, candidate, baseline, label in PAIRWISE_COMPARISONS:
            if candidate not in by_scenario.index or baseline not in by_scenario.index:
                continue
            candidate_value = float(by_scenario.loc[candidate, "mean_total_cost_plus_penalty"])
            baseline_value = float(by_scenario.loc[baseline, "mean_total_cost_plus_penalty"])
            if abs(baseline_value) <= 1e-12:
                ratio = np.nan
            else:
                ratio = (baseline_value - candidate_value) / baseline_value
            rows.append(
                {
                    "comparison": comparison,
                    "label": label,
                    "seed": int(seed),
                    "baseline_scenario": baseline,
                    "candidate_scenario": candidate,
                    "baseline_mean_total_cost_plus_penalty": baseline_value,
                    "candidate_mean_total_cost_plus_penalty": candidate_value,
                    "absolute_reduction": baseline_value - candidate_value,
                    "cost_plus_penalty_reduction_ratio": float(ratio),
                    "cost_plus_penalty_reduction_percent": float(ratio * 100.0),
                }
            )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str], float_digits: int = 4) -> str:
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


def write_report(
    seed_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    seeds: list[int],
    output_dir: Path,
) -> None:
    report_path = output_dir / "ev_scenario_comparison_report.md"
    lines = [
        "# EV Scenario Comparison Report",
        "",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "## Scenario Definitions",
        "",
        f"- {SCENARIO_DESCRIPTIONS['uncontrolled_charging']}",
        f"- {SCENARIO_DESCRIPTIONS['ordered_charging']}",
        f"- {SCENARIO_DESCRIPTIONS['v2g']}",
        "",
        "V2G uses the current TD3 main model/results. The TD3 sigma=0.10 failed version is not used.",
        "",
        "## Seed Summary",
        "",
        markdown_table(
            seed_df,
            [
                "ev_scenario",
                "seed",
                "mean_total_system_cost",
                "mean_total_penalties",
                "mean_total_cost_plus_penalty",
                "mean_total_grid_sell_kwh",
                "mean_total_ev_charge_kwh",
                "mean_total_ev_discharge_kwh",
            ],
            float_digits=3,
        ),
        "",
        "## Constraint And Guard Metrics",
        "",
        markdown_table(
            seed_df,
            [
                "ev_scenario",
                "seed",
                "total_depart_energy_shortage_kwh",
                "sum_terminal_ees_shortage_kwh",
                "terminal_ees_feasible_ratio",
                "sum_penalty_ev_export_guard",
                "sum_ev_export_overlap_kwh",
            ],
            float_digits=6,
        ),
        "",
        "## Aggregate Summary",
        "",
        markdown_table(
            aggregate_df,
            [
                "ev_scenario",
                "n_seeds",
                "mean_total_cost_plus_penalty_mean",
                "mean_total_grid_buy_kwh_mean",
                "mean_total_grid_sell_kwh_mean",
                "mean_total_ev_charge_kwh_mean",
                "mean_total_ev_discharge_kwh_mean",
                "sum_terminal_ees_shortage_kwh_mean",
            ],
            float_digits=4,
        ),
        "",
        "## Cost+Penalty Reductions",
        "",
    ]

    if pairwise_df.empty:
        lines.append("Pairwise reductions could not be computed because at least one scenario is missing.")
    else:
        lines.append(
            markdown_table(
                pairwise_df,
                [
                    "comparison",
                    "seed",
                    "baseline_scenario",
                    "candidate_scenario",
                    "cost_plus_penalty_reduction_percent",
                ],
                float_digits=4,
            )
        )

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = [int(seed) for seed in args.seeds]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ev_scenario in SCENARIO_ORDER:
        for seed in seeds:
            row = build_seed_row(ev_scenario, seed)
            if row is not None:
                rows.append(row)

    if not rows:
        raise SystemExit("No EV scenario test summaries were found.")

    seed_df = pd.DataFrame(rows, columns=SEED_SUMMARY_COLUMNS)
    seed_df["ev_scenario"] = pd.Categorical(
        seed_df["ev_scenario"], categories=SCENARIO_ORDER, ordered=True
    )
    seed_df = seed_df.sort_values(["ev_scenario", "seed"]).reset_index(drop=True)
    seed_df["ev_scenario"] = seed_df["ev_scenario"].astype(str)

    aggregate_df = build_aggregate_summary(seed_df)
    pairwise_df = build_pairwise_reductions(seed_df)

    seed_path = output_dir / "ev_scenario_seed_summary.csv"
    aggregate_path = output_dir / "ev_scenario_aggregate_summary.csv"
    pairwise_path = output_dir / "ev_scenario_pairwise_reductions.csv"

    seed_df.to_csv(seed_path, index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    pairwise_df.to_csv(pairwise_path, index=False, encoding="utf-8-sig")
    write_report(seed_df, aggregate_df, pairwise_df, seeds, output_dir)

    print(f"[summary] seed summary: {seed_path}")
    print(f"[summary] aggregate summary: {aggregate_path}")
    print(f"[summary] pairwise reductions: {pairwise_path}")
    print(f"[summary] report: {output_dir / 'ev_scenario_comparison_report.md'}")
    print(seed_df.to_string(index=False))
    if not pairwise_df.empty:
        print(pairwise_df.to_string(index=False))


if __name__ == "__main__":
    main()
