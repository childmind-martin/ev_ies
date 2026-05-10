from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METHODS = ["TD3", "DDPG", "PPO"]
DEFAULT_SEEDS = [42, 2024, 2025]
OUTPUT_DIR = Path("results/comparison")

SUMMARY_PATHS = {
    "TD3": "results/td3_yearly_test_seed_{seed}/daily_summary.csv",
    "DDPG": "results/ddpg_yearly_test_seed_{seed}/daily_summary.csv",
    "PPO": "results/ppo_sb3_direct_test_seed_{seed}/daily_summary.csv",
}

SEED_SUMMARY_COLUMNS = [
    "method",
    "seed",
    "n_days",
    "mean_total_system_cost",
    "mean_total_penalties",
    "mean_total_cost_plus_penalty",
    "mean_total_grid_buy_kwh",
    "mean_total_grid_sell_kwh",
    "mean_total_storage_peak_shaved_kwh",
    "mean_total_depart_energy_shortage_kwh",
    "total_unmet_e",
    "total_unmet_h",
    "total_unmet_c",
    "terminal_ees_feasible_days",
    "terminal_ees_feasible_ratio",
    "min_final_ees_soc",
    "sum_terminal_ees_shortage_kwh",
    "sum_penalty_terminal_ees_soc",
]

AGGREGATE_METRICS = [
    "mean_total_system_cost",
    "mean_total_penalties",
    "mean_total_cost_plus_penalty",
    "terminal_ees_feasible_ratio",
    "min_final_ees_soc",
    "sum_terminal_ees_shortage_kwh",
    "sum_penalty_terminal_ees_soc",
    "mean_total_grid_buy_kwh",
    "mean_total_grid_sell_kwh",
    "mean_total_storage_peak_shaved_kwh",
]

ALGORITHM_SUMMARY_COLUMNS = [
    "method",
    "n_seeds",
    "mean_total_system_cost_mean",
    "mean_total_system_cost_std",
    "mean_total_penalties_mean",
    "mean_total_penalties_std",
    "mean_total_cost_plus_penalty_mean",
    "mean_total_cost_plus_penalty_std",
    "terminal_ees_feasible_ratio_mean",
    "terminal_ees_feasible_ratio_std",
    "min_final_ees_soc_mean",
    "min_final_ees_soc_std",
    "sum_terminal_ees_shortage_kwh_mean",
    "sum_terminal_ees_shortage_kwh_std",
    "sum_penalty_terminal_ees_soc_mean",
    "sum_penalty_terminal_ees_soc_std",
    "mean_total_grid_buy_kwh_mean",
    "mean_total_grid_buy_kwh_std",
    "mean_total_grid_sell_kwh_mean",
    "mean_total_grid_sell_kwh_std",
    "mean_total_storage_peak_shaved_kwh_mean",
    "mean_total_storage_peak_shaved_kwh_std",
    "cost_plus_penalty_mean_std",
    "system_cost_mean_std",
    "penalties_mean_std",
    "terminal_ees_feasible_ratio_mean_std",
    "terminal_ees_shortage_kwh_mean_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multi-seed method and algorithm summaries.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, help="Methods to summarize: TD3 DDPG PPO.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Random seeds to summarize.")
    return parser.parse_args()


def normalize_methods(methods: list[str]) -> list[str]:
    normalized = []
    for method in methods:
        key = method.upper()
        if key not in SUMMARY_PATHS:
            raise ValueError(f"Unknown method {method!r}. Valid methods: {', '.join(SUMMARY_PATHS)}")
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
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def build_seed_row(method: str, seed: int, path: Path) -> dict[str, float | int | str]:
    df = pd.read_csv(path)
    n_days = int(len(df))
    if n_days == 0:
        raise ValueError(f"Empty daily summary: {path}")

    total_system_cost = numeric_column(df, "total_system_cost")
    total_penalties = numeric_column(df, "total_penalties")
    if "total_cost_plus_penalty" in df.columns:
        total_cost_plus_penalty = numeric_column(df, "total_cost_plus_penalty")
    else:
        total_cost_plus_penalty = total_system_cost + total_penalties

    feasible = bool_column(df, "ees_terminal_soc_feasible")
    feasible_days = int(feasible.sum())

    return {
        "method": method,
        "seed": int(seed),
        "n_days": n_days,
        "mean_total_system_cost": float(total_system_cost.mean()),
        "mean_total_penalties": float(total_penalties.mean()),
        "mean_total_cost_plus_penalty": float(total_cost_plus_penalty.mean()),
        "mean_total_grid_buy_kwh": float(numeric_column(df, "total_grid_buy_kwh").mean()),
        "mean_total_grid_sell_kwh": float(numeric_column(df, "total_grid_sell_kwh").mean()),
        "mean_total_storage_peak_shaved_kwh": float(numeric_column(df, "total_storage_peak_shaved_kwh").mean()),
        "mean_total_depart_energy_shortage_kwh": float(numeric_column(df, "total_depart_energy_shortage_kwh").mean()),
        "total_unmet_e": float(numeric_column(df, "total_unmet_e").sum()),
        "total_unmet_h": float(numeric_column(df, "total_unmet_h").sum()),
        "total_unmet_c": float(numeric_column(df, "total_unmet_c").sum()),
        "terminal_ees_feasible_days": feasible_days,
        "terminal_ees_feasible_ratio": float(feasible_days / n_days),
        "min_final_ees_soc": float(numeric_column(df, "final_ees_soc", np.nan).min()),
        "sum_terminal_ees_shortage_kwh": float(numeric_column(df, "terminal_ees_shortage_kwh").sum()),
        "sum_penalty_terminal_ees_soc": float(numeric_column(df, "total_penalty_terminal_ees_soc").sum()),
    }


def mean_std_text(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.2f} ± {std_value:.2f}"


def build_algorithm_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in seed_df.groupby("method", sort=False):
        row: dict[str, float | int | str] = {"method": method, "n_seeds": int(group["seed"].nunique())}
        for metric in AGGREGATE_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values.dropna()) > 1 else 0.0

        row["cost_plus_penalty_mean_std"] = mean_std_text(
            float(row["mean_total_cost_plus_penalty_mean"]),
            float(row["mean_total_cost_plus_penalty_std"]),
        )
        row["system_cost_mean_std"] = mean_std_text(
            float(row["mean_total_system_cost_mean"]),
            float(row["mean_total_system_cost_std"]),
        )
        row["penalties_mean_std"] = mean_std_text(
            float(row["mean_total_penalties_mean"]),
            float(row["mean_total_penalties_std"]),
        )
        row["terminal_ees_feasible_ratio_mean_std"] = mean_std_text(
            float(row["terminal_ees_feasible_ratio_mean"]),
            float(row["terminal_ees_feasible_ratio_std"]),
        )
        row["terminal_ees_shortage_kwh_mean_std"] = mean_std_text(
            float(row["sum_terminal_ees_shortage_kwh_mean"]),
            float(row["sum_terminal_ees_shortage_kwh_std"]),
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=ALGORITHM_SUMMARY_COLUMNS)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def write_report(seed_df: pd.DataFrame, algorithm_df: pd.DataFrame, seeds: list[int]) -> None:
    report_path = OUTPUT_DIR / "multiseed_comparison_report.md"
    table_columns = [
        "method",
        "n_seeds",
        "cost_plus_penalty_mean_std",
        "system_cost_mean_std",
        "penalties_mean_std",
        "terminal_ees_feasible_ratio_mean_std",
        "terminal_ees_shortage_kwh_mean_std",
    ]
    seed_feasibility_columns = [
        "method",
        "seed",
        "terminal_ees_feasible_days",
        "terminal_ees_feasible_ratio",
        "sum_terminal_ees_shortage_kwh",
        "sum_penalty_terminal_ees_soc",
    ]

    ppo_note = "PPO results were not available."
    if "PPO" in set(seed_df["method"]):
        ppo_rows = seed_df[seed_df["method"] == "PPO"]
        ppo_min_ratio = float(ppo_rows["terminal_ees_feasible_ratio"].min())
        if ppo_min_ratio < 1.0:
            ppo_note = (
                "PPO is an infeasible auxiliary baseline in the current setting because at least one seed "
                f"has terminal EES SOC feasible ratio below 100% (minimum={ppo_min_ratio:.4f})."
            )
        else:
            ppo_note = "PPO satisfies the terminal EES SOC constraint for all available seeds."

    infeasible_methods = []
    for method, group in seed_df.groupby("method", sort=False):
        min_ratio = float(group["terminal_ees_feasible_ratio"].min())
        if min_ratio < 1.0:
            infeasible_methods.append(f"{method} minimum seed feasible ratio={min_ratio:.4f}")
    if infeasible_methods:
        feasibility_note = (
            "At least one method is not 100% terminal-EES feasible across all seeds: "
            + "; ".join(infeasible_methods)
            + ". Do not describe these multi-seed results as fully feasible without the feasibility column."
        )
    else:
        feasibility_note = "All available methods are 100% terminal-EES feasible across all seeds."

    lines = [
        "# Multi-seed comparison report",
        "",
        f"Random seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "Training length: 4000 episodes / 96000 timesteps for TD3, DDPG, and PPO.",
        "",
        "Testing: deterministic=True.",
        "",
        "Reward-curve smoothing: moving average window = 100 episodes.",
        "",
        "Rule-based-V2G is a deterministic rule baseline. It is not included in multi-seed training statistics or reward curves, but it can remain in the final performance table as a traditional baseline.",
        "",
        ppo_note,
        "",
        feasibility_note,
        "",
        "Reward curves show training trends only. Economic conclusions should use test-set system cost, penalties, cost+penalty, and constraint feasibility, not guide_reward.",
        "",
        "## Mean ± std summary",
        "",
        markdown_table(algorithm_df, table_columns),
        "",
        "## Terminal EES SOC feasibility by seed",
        "",
        markdown_table(seed_df, seed_feasibility_columns),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    methods = normalize_methods(args.methods)
    seeds = [int(seed) for seed in args.seeds]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for method in methods:
        for seed in seeds:
            path = Path(SUMMARY_PATHS[method].format(seed=seed))
            if not path.exists():
                missing.append(str(path))
                print(f"WARNING: missing daily summary: {path}")
                continue
            rows.append(build_seed_row(method, seed, path))

    if not rows:
        raise SystemExit("No multi-seed daily_summary.csv files were found.")

    seed_df = pd.DataFrame(rows, columns=SEED_SUMMARY_COLUMNS)
    algorithm_df = build_algorithm_summary(seed_df)

    seed_path = OUTPUT_DIR / "multiseed_method_seed_summary.csv"
    algorithm_path = OUTPUT_DIR / "multiseed_algorithm_summary.csv"
    algorithm_md_path = OUTPUT_DIR / "multiseed_algorithm_summary.md"

    seed_df.to_csv(seed_path, index=False, encoding="utf-8-sig")
    algorithm_df.to_csv(algorithm_path, index=False, encoding="utf-8-sig")
    algorithm_md_path.write_text(
        markdown_table(
            algorithm_df,
            [
                "method",
                "n_seeds",
                "cost_plus_penalty_mean_std",
                "system_cost_mean_std",
                "penalties_mean_std",
                "terminal_ees_feasible_ratio_mean_std",
                "terminal_ees_shortage_kwh_mean_std",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(seed_df, algorithm_df, seeds)

    print(f"[summary] seed summary: {seed_path}")
    print(f"[summary] algorithm summary: {algorithm_path}")
    print(f"[summary] markdown summary: {algorithm_md_path}")
    print(f"[summary] report: {OUTPUT_DIR / 'multiseed_comparison_report.md'}")
    if missing:
        print(f"WARNING: {len(missing)} expected files were missing.")
    print(algorithm_df.to_string(index=False))


if __name__ == "__main__":
    main()
