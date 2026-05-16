from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_METHODS = ["DDPG", "PPO"]
DEFAULT_SEEDS = [42, 2024, 2025]
OUTPUT_DIR = Path("results/ddpg_ppo_feasible_checkpoint_selection")

ORIGINAL_RESULT_DIRS = {
    "DDPG": "results/ddpg_yearly_test_seed_{seed}",
    "PPO": "results/ppo_sb3_direct_test_seed_{seed}",
}
SELECTED_RESULT_DIRS = {
    "DDPG": "results/ddpg_feasible_selected_test_seed_{seed}",
    "PPO": "results/ppo_feasible_selected_test_seed_{seed}",
}

SEED_COLUMNS = [
    "method",
    "variant",
    "seed",
    "n_days",
    "mean_total_system_cost",
    "mean_total_penalties",
    "mean_total_cost_plus_penalty",
    "terminal_ees_feasible_days",
    "terminal_ees_feasible_ratio",
    "sum_terminal_ees_shortage_kwh",
    "mean_terminal_ees_shortage_kwh_per_day",
    "mean_total_depart_energy_shortage_kwh",
    "sum_total_depart_energy_shortage_kwh",
    "total_unmet_e",
    "total_unmet_h",
    "total_unmet_c",
    "selected_checkpoint_step",
    "selected_checkpoint_path",
    "result_dir",
]

AGGREGATE_METRICS = [
    "mean_total_system_cost",
    "mean_total_penalties",
    "mean_total_cost_plus_penalty",
    "terminal_ees_feasible_ratio",
    "sum_terminal_ees_shortage_kwh",
    "mean_terminal_ees_shortage_kwh_per_day",
    "mean_total_depart_energy_shortage_kwh",
    "sum_total_depart_energy_shortage_kwh",
    "total_unmet_e",
    "total_unmet_h",
    "total_unmet_c",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DDPG/PPO feasible checkpoint summaries and recommendations."
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def normalize_methods(methods: list[str]) -> list[str]:
    normalized = []
    for method in methods:
        key = method.upper()
        if key not in ORIGINAL_RESULT_DIRS:
            raise ValueError(f"Unknown method {method!r}.")
        normalized.append(key)
    return normalized


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


def summarize_daily(
    *,
    method: str,
    variant: str,
    seed: int,
    result_dir: Path,
    selected_row: pd.Series | None,
) -> dict[str, Any] | None:
    daily_path = result_dir / "daily_summary.csv"
    if not daily_path.exists():
        print(f"WARNING: missing daily summary: {daily_path}")
        return None
    daily = pd.read_csv(daily_path)
    if daily.empty:
        raise ValueError(f"Empty daily summary: {daily_path}")
    system_cost = numeric_column(daily, "total_system_cost")
    penalties = numeric_column(daily, "total_penalties")
    if "total_cost_plus_penalty" in daily.columns:
        cost_plus_penalty = numeric_column(daily, "total_cost_plus_penalty")
    else:
        cost_plus_penalty = system_cost + penalties
    feasible = bool_column(daily, "ees_terminal_soc_feasible")
    shortage = numeric_column(daily, "terminal_ees_shortage_kwh")
    depart = numeric_column(daily, "total_depart_energy_shortage_kwh")
    n_days = int(len(daily))

    row = {
        "method": method,
        "variant": variant,
        "seed": int(seed),
        "n_days": n_days,
        "mean_total_system_cost": float(system_cost.mean()),
        "mean_total_penalties": float(penalties.mean()),
        "mean_total_cost_plus_penalty": float(cost_plus_penalty.mean()),
        "terminal_ees_feasible_days": int(feasible.sum()),
        "terminal_ees_feasible_ratio": float(feasible.sum() / n_days),
        "sum_terminal_ees_shortage_kwh": float(shortage.sum()),
        "mean_terminal_ees_shortage_kwh_per_day": float(shortage.mean()),
        "mean_total_depart_energy_shortage_kwh": float(depart.mean()),
        "sum_total_depart_energy_shortage_kwh": float(depart.sum()),
        "total_unmet_e": float(numeric_column(daily, "total_unmet_e").sum()),
        "total_unmet_h": float(numeric_column(daily, "total_unmet_h").sum()),
        "total_unmet_c": float(numeric_column(daily, "total_unmet_c").sum()),
        "selected_checkpoint_step": "",
        "selected_checkpoint_path": "",
        "result_dir": str(result_dir),
    }
    if selected_row is not None:
        row["selected_checkpoint_step"] = selected_row.get("selected_checkpoint_step", "")
        row["selected_checkpoint_path"] = selected_row.get("selected_checkpoint_path", "")
    return row


def mean_std_text(mean_value: float, std_value: float) -> str:
    if pd.isna(mean_value):
        return "NA"
    return f"{float(mean_value):.6g} +/- {float(std_value):.6g}"


def build_algorithm_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, variant), group in seed_df.groupby(["method", "variant"], sort=False):
        row: dict[str, Any] = {
            "method": method,
            "variant": variant,
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


def build_vs_original(algorithm_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in DEFAULT_METHODS:
        original = algorithm_df[
            (algorithm_df["method"] == method) & (algorithm_df["variant"] == "original")
        ]
        selected = algorithm_df[
            (algorithm_df["method"] == method)
            & (algorithm_df["variant"] == "feasible-selected")
        ]
        if original.empty or selected.empty:
            continue
        original_row = original.iloc[0]
        selected_row = selected.iloc[0]
        for metric in AGGREGATE_METRICS:
            old_mean = float(original_row[f"{metric}_mean"])
            old_std = float(original_row[f"{metric}_std"])
            new_mean = float(selected_row[f"{metric}_mean"])
            new_std = float(selected_row[f"{metric}_std"])
            delta = new_mean - old_mean
            delta_percent = delta / abs(old_mean) * 100.0 if abs(old_mean) > 1e-12 else np.nan
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "original_mean": old_mean,
                    "original_std": old_std,
                    "feasible_selected_mean": new_mean,
                    "feasible_selected_std": new_std,
                    "selected_minus_original": delta,
                    "selected_minus_original_percent": delta_percent,
                    "original_mean_std": mean_std_text(old_mean, old_std),
                    "feasible_selected_mean_std": mean_std_text(new_mean, new_std),
                }
            )
    return pd.DataFrame(rows)


def get_metric(algorithm_df: pd.DataFrame, method: str, variant: str, metric: str) -> float:
    row = algorithm_df[
        (algorithm_df["method"] == method) & (algorithm_df["variant"] == variant)
    ]
    if row.empty:
        return np.nan
    return float(row.iloc[0][f"{metric}_mean"])


def build_recommendation(algorithm_df: pd.DataFrame, seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in DEFAULT_METHODS:
        old_ratio = get_metric(algorithm_df, method, "original", "terminal_ees_feasible_ratio")
        new_ratio = get_metric(algorithm_df, method, "feasible-selected", "terminal_ees_feasible_ratio")
        old_cost = get_metric(algorithm_df, method, "original", "mean_total_cost_plus_penalty")
        new_cost = get_metric(algorithm_df, method, "feasible-selected", "mean_total_cost_plus_penalty")
        new_shortage_day = get_metric(
            algorithm_df, method, "feasible-selected", "mean_terminal_ees_shortage_kwh_per_day"
        )
        selected_seeds = seed_df[
            (seed_df["method"] == method) & (seed_df["variant"] == "feasible-selected")
        ]
        full_seed_count = int((selected_seeds["terminal_ees_feasible_days"].astype(int) >= 60).sum())
        cost_not_worse = bool(pd.notna(new_cost) and pd.notna(old_cost) and new_cost <= old_cost * 1.005)
        ratio_improved = bool(pd.notna(new_ratio) and pd.notna(old_ratio) and new_ratio > old_ratio + 1e-12)
        constraint_ok = bool(
            pd.notna(new_ratio)
            and (new_ratio >= 0.98 or full_seed_count >= 2)
            and pd.notna(new_shortage_day)
            and new_shortage_day <= 2.0
        )

        if method == "DDPG":
            if ratio_improved and cost_not_worse:
                source = "feasible-selected"
                reason = "EES feasible ratio improved and cost+penalty did not worsen by more than 0.5%."
            else:
                source = "original"
                if ratio_improved:
                    reason = (
                        "Feasible-selected improves EES feasibility, but cost+penalty worsens by more "
                        "than 0.5%; keep original unless strict feasibility is prioritized over economics."
                    )
                else:
                    reason = "Feasible-selected does not improve EES feasibility enough; keep original."
            can_enter_main = bool(source == "feasible-selected" and constraint_ok)
            auxiliary = not can_enter_main
        else:
            if constraint_ok:
                source = "feasible-selected"
                if cost_not_worse:
                    reason = "PPO feasible-selected meets the feasibility gate and cost+penalty is acceptable."
                else:
                    reason = (
                        "PPO feasible-selected fixes terminal-EES feasibility; cost+penalty is higher "
                        "than original, but the original PPO is not constraint-feasible."
                    )
                can_enter_main = True
                auxiliary = False
            else:
                source = "feasible-selected" if ratio_improved else "original"
                reason = "PPO remains terminal-EES infeasible enough to require auxiliary-baseline labeling."
                can_enter_main = False
                auxiliary = True

        rows.append(
            {
                "method": method,
                "recommended_result_source": source,
                "reason": reason,
                "can_enter_main_table": can_enter_main,
                "should_mark_auxiliary": auxiliary,
                "notes": (
                    f"original_ratio={old_ratio:.6g}; selected_ratio={new_ratio:.6g}; "
                    f"original_cost={old_cost:.6g}; selected_cost={new_cost:.6g}; "
                    f"selected_full_60_seed_count={full_seed_count}"
                ),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str], float_digits: int = 6) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in df.iterrows():
        values = []
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
    seed_df: pd.DataFrame,
    algorithm_df: pd.DataFrame,
    vs_df: pd.DataFrame,
    recommendation_df: pd.DataFrame,
) -> None:
    selected_metric_rows = vs_df[
        vs_df["metric"].isin(
            [
                "mean_total_cost_plus_penalty",
                "terminal_ees_feasible_ratio",
                "sum_terminal_ees_shortage_kwh",
                "mean_terminal_ees_shortage_kwh_per_day",
                "mean_total_depart_energy_shortage_kwh",
                "total_unmet_e",
                "total_unmet_h",
                "total_unmet_c",
            ]
        )
    ].copy()
    lines = [
        "# DDPG/PPO feasibility-first checkpoint selection report",
        "",
        "Checkpoint selection uses validation metrics only. Test metrics are reported after selection and are not used to choose checkpoints.",
        "",
        "## Selected Checkpoints",
        "",
        markdown_table(
            selected_df,
            [
                "method",
                "seed",
                "selected_checkpoint_step",
                "val_EES_feasible",
                "val_terminal_shortage",
                "val_cost_plus_penalty",
                "val_reward",
                "test_EES_feasible",
                "test_terminal_shortage",
                "test_cost_plus_penalty",
            ],
        ),
        "",
        "## Algorithm Summary",
        "",
        markdown_table(
            algorithm_df,
            [
                "method",
                "variant",
                "n_seeds",
                "mean_total_cost_plus_penalty_mean_std",
                "terminal_ees_feasible_ratio_mean_std",
                "sum_terminal_ees_shortage_kwh_mean_std",
                "mean_total_depart_energy_shortage_kwh_mean_std",
            ],
        ),
        "",
        "## Original Versus Feasible-Selected",
        "",
        markdown_table(
            selected_metric_rows,
            [
                "method",
                "metric",
                "original_mean_std",
                "feasible_selected_mean_std",
                "selected_minus_original",
                "selected_minus_original_percent",
            ],
        ),
        "",
        "## Final Recommendation",
        "",
        markdown_table(
            recommendation_df,
            [
                "method",
                "recommended_result_source",
                "can_enter_main_table",
                "should_mark_auxiliary",
                "reason",
            ],
        ),
        "",
    ]

    for method in DEFAULT_METHODS:
        rec = recommendation_df[recommendation_df["method"] == method]
        if rec.empty:
            continue
        rec_row = rec.iloc[0]
        if method == "DDPG":
            lines.extend(
                [
                    "## DDPG Judgment",
                    "",
                    f"DDPG feasible-selected recommendation: {rec_row['recommended_result_source']}. {rec_row['reason']}",
                    "",
                ]
            )
        if method == "PPO":
            ppo_aux = bool(rec_row["should_mark_auxiliary"])
            lines.extend(
                [
                    "## PPO Judgment",
                    "",
                    (
                        "PPO remains an auxiliary baseline."
                        if ppo_aux
                        else "PPO feasible-selected can be considered for the main table."
                    ),
                    "",
                ]
            )

    (output_dir / "selected_ddpg_ppo_checkpoint_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    methods = normalize_methods(args.methods)
    seeds = [int(seed) for seed in args.seeds]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_path = output_dir / "selected_ddpg_ppo_checkpoints.csv"
    if not selected_path.exists():
        raise SystemExit(f"Missing selected checkpoint file: {selected_path}")
    selected_df = pd.read_csv(selected_path)

    seed_rows: list[dict[str, Any]] = []
    for method in methods:
        method_selected = selected_df[selected_df["method"] == method]
        for seed in seeds:
            selected_row_df = method_selected[method_selected["seed"].astype(int) == int(seed)]
            selected_row = selected_row_df.iloc[0] if not selected_row_df.empty else None
            original_row = summarize_daily(
                method=method,
                variant="original",
                seed=seed,
                result_dir=Path(ORIGINAL_RESULT_DIRS[method].format(seed=seed)),
                selected_row=None,
            )
            if original_row is not None:
                seed_rows.append(original_row)
            selected_summary = summarize_daily(
                method=method,
                variant="feasible-selected",
                seed=seed,
                result_dir=Path(SELECTED_RESULT_DIRS[method].format(seed=seed)),
                selected_row=selected_row,
            )
            if selected_summary is not None:
                seed_rows.append(selected_summary)

    seed_df = pd.DataFrame(seed_rows, columns=SEED_COLUMNS)
    algorithm_df = build_algorithm_summary(seed_df)
    vs_df = build_vs_original(algorithm_df)
    recommendation_df = build_recommendation(algorithm_df, seed_df)

    seed_df.to_csv(
        output_dir / "ddpg_ppo_feasible_selected_seed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    algorithm_df.to_csv(
        output_dir / "ddpg_ppo_feasible_selected_algorithm_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    vs_df.to_csv(
        output_dir / "ddpg_ppo_feasible_selected_vs_original_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    recommendation_df.to_csv(
        output_dir / "final_baseline_recommendation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_report(
        output_dir=output_dir,
        selected_df=selected_df,
        seed_df=seed_df,
        algorithm_df=algorithm_df,
        vs_df=vs_df,
        recommendation_df=recommendation_df,
    )

    print(f"[summary] seed summary: {output_dir / 'ddpg_ppo_feasible_selected_seed_summary.csv'}")
    print(f"[summary] algorithm summary: {output_dir / 'ddpg_ppo_feasible_selected_algorithm_summary.csv'}")
    print(f"[summary] vs original: {output_dir / 'ddpg_ppo_feasible_selected_vs_original_summary.csv'}")
    print(f"[summary] report: {output_dir / 'selected_ddpg_ppo_checkpoint_report.md'}")
    print(f"[summary] recommendation: {output_dir / 'final_baseline_recommendation.csv'}")
    print(recommendation_df.to_string(index=False))


if __name__ == "__main__":
    main()
