from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_SEEDS = [42, 2024, 2025]
OUTPUT_DIR = Path("results/td3_cuda_multiseed_comparison")
FIGURE_DIR = OUTPUT_DIR / "figures"
TEST_REQUIRED_FILES = [
    "daily_summary.csv",
    "timeseries_detail.csv",
    "runtime_summary.json",
    "td3_test_export.xlsx",
]

SEED_SUMMARY_COLUMNS = [
    "method",
    "seed",
    "run_name",
    "n_days",
    "train_complete",
    "test_complete",
    "best_eval_timestep",
    "best_validation_reward",
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
    "mean_total_ev_charge_kwh",
    "mean_total_ev_discharge_kwh",
    "mean_total_storage_peak_shaved_kwh",
    "mean_total_depart_energy_shortage_kwh",
    "total_unmet_e",
    "total_unmet_h",
    "total_unmet_c",
    "total_penalty_ev_export_guard",
    "model_path",
    "result_dir",
    "training_result_dir",
    "log_dir",
    "daily_summary_path",
    "runtime_summary_path",
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
    "mean_total_ev_charge_kwh",
    "mean_total_ev_discharge_kwh",
    "mean_total_storage_peak_shaved_kwh",
    "mean_total_depart_energy_shortage_kwh",
    "total_unmet_e",
    "total_unmet_h",
    "total_unmet_c",
    "total_penalty_ev_export_guard",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the isolated CUDA TD3 multiseed summary and original TD3 comparison."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def cuda_paths(seed: int) -> dict[str, Path]:
    suffix = f"cuda_seed_{int(seed)}"
    return {
        "model_path": Path(f"models/td3_yearly_single_{suffix}/best/best_model.zip"),
        "training_result_dir": Path(f"results/td3_yearly_training_{suffix}"),
        "result_dir": Path(f"results/td3_yearly_test_{suffix}"),
        "log_dir": Path(f"logs/td3_yearly_single_{suffix}"),
    }


def original_paths(seed: int) -> dict[str, Path]:
    suffix = f"seed_{int(seed)}"
    return {
        "model_path": Path(f"models/td3_yearly_single_{suffix}/best/best_model.zip"),
        "training_result_dir": Path(f"results/td3_yearly_training_{suffix}"),
        "result_dir": Path(f"results/td3_yearly_test_{suffix}"),
        "log_dir": Path(f"logs/td3_yearly_single_{suffix}"),
    }


def run_name(method: str, seed: int) -> str:
    if method == "TD3-CUDA":
        return f"td3_cuda_seed_{int(seed)}"
    return f"td3_original_seed_{int(seed)}"


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


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def test_complete(result_dir: Path) -> bool:
    return all((result_dir / filename).exists() for filename in TEST_REQUIRED_FILES)


def train_complete(paths: dict[str, Path]) -> bool:
    return (
        paths["model_path"].exists()
        and (paths["training_result_dir"] / "episode_summary.csv").exists()
    )


def load_best_eval(log_dir: Path) -> tuple[float, float]:
    eval_path = log_dir / "evaluations.npz"
    if not eval_path.exists():
        return np.nan, np.nan
    try:
        data = np.load(eval_path)
        timesteps = np.asarray(data["timesteps"], dtype=float)
        results = np.asarray(data["results"], dtype=float)
        if results.ndim == 1:
            mean_rewards = results
        else:
            mean_rewards = np.nanmean(results, axis=1)
        if len(timesteps) == 0 or len(mean_rewards) == 0:
            return np.nan, np.nan
        best_index = int(np.nanargmax(mean_rewards))
        return float(timesteps[best_index]), float(mean_rewards[best_index])
    except Exception:
        return np.nan, np.nan


def load_timeseries_metrics(timeseries_path: Path) -> pd.DataFrame:
    if not timeseries_path.exists():
        return pd.DataFrame(columns=["case_index"])
    ts = pd.read_csv(timeseries_path)
    if "case_index" not in ts.columns:
        return pd.DataFrame(columns=["case_index"])
    grouped = ts.groupby("case_index", sort=True)
    rows = pd.DataFrame({"case_index": sorted(ts["case_index"].unique())})
    sources = {
        "total_ev_charge_kwh": "p_ev_ch",
        "total_ev_discharge_kwh": "p_ev_dis",
    }
    for out_col, source_col in sources.items():
        if source_col in ts.columns:
            rows[out_col] = grouped[source_col].sum().reset_index(drop=True)
    return rows


def load_daily_with_ev_metrics(result_dir: Path) -> pd.DataFrame:
    daily_path = result_dir / "daily_summary.csv"
    daily = pd.read_csv(daily_path)
    ts_metrics = load_timeseries_metrics(result_dir / "timeseries_detail.csv")
    if not ts_metrics.empty:
        daily = daily.merge(ts_metrics, on="case_index", how="left", suffixes=("", "_from_ts"))
        for column in ("total_ev_charge_kwh", "total_ev_discharge_kwh"):
            ts_column = f"{column}_from_ts"
            if ts_column in daily.columns:
                if column not in daily.columns:
                    daily[column] = daily[ts_column]
                else:
                    daily[column] = daily[column].fillna(daily[ts_column])
                daily = daily.drop(columns=[ts_column])

    if "total_ev_charge_kwh" not in daily.columns:
        if "avg_p_ev_ch_kw" in daily.columns:
            daily["total_ev_charge_kwh"] = numeric_column(daily, "avg_p_ev_ch_kw") * 24.0
        else:
            daily["total_ev_charge_kwh"] = np.nan
    if "total_ev_discharge_kwh" not in daily.columns:
        if "avg_p_ev_dis_kw" in daily.columns:
            daily["total_ev_discharge_kwh"] = numeric_column(daily, "avg_p_ev_dis_kw") * 24.0
        else:
            daily["total_ev_discharge_kwh"] = np.nan
    return daily


def empty_seed_row(method: str, seed: int, paths: dict[str, Path]) -> dict[str, object]:
    best_eval_timestep, best_validation_reward = load_best_eval(paths["log_dir"])
    result_dir = paths["result_dir"]
    return {
        "method": method,
        "seed": int(seed),
        "run_name": run_name(method, seed),
        "n_days": 0,
        "train_complete": bool(train_complete(paths)),
        "test_complete": bool(test_complete(result_dir)),
        "best_eval_timestep": best_eval_timestep,
        "best_validation_reward": best_validation_reward,
        "model_path": str(paths["model_path"]),
        "result_dir": str(result_dir),
        "training_result_dir": str(paths["training_result_dir"]),
        "log_dir": str(paths["log_dir"]),
        "daily_summary_path": str(result_dir / "daily_summary.csv"),
        "runtime_summary_path": str(result_dir / "runtime_summary.json"),
    }


def build_seed_row(method: str, seed: int, paths: dict[str, Path]) -> dict[str, object]:
    row = empty_seed_row(method, seed, paths)
    daily_path = paths["result_dir"] / "daily_summary.csv"
    if not daily_path.exists():
        return row

    daily = load_daily_with_ev_metrics(paths["result_dir"])
    n_days = int(len(daily))
    if n_days <= 0:
        return row

    total_system_cost = numeric_column(daily, "total_system_cost")
    total_penalties = numeric_column(daily, "total_penalties")
    if "total_cost_plus_penalty" in daily.columns:
        total_cost_plus_penalty = numeric_column(daily, "total_cost_plus_penalty")
    else:
        total_cost_plus_penalty = total_system_cost + total_penalties
    feasible = bool_column(daily, "ees_terminal_soc_feasible")
    feasible_days = int(feasible.sum())
    terminal_shortage = numeric_column(daily, "terminal_ees_shortage_kwh")

    row.update(
        {
            "n_days": n_days,
            "test_complete": bool(test_complete(paths["result_dir"])),
            "mean_total_system_cost": float(total_system_cost.mean()),
            "mean_total_penalties": float(total_penalties.mean()),
            "mean_total_cost_plus_penalty": float(total_cost_plus_penalty.mean()),
            "terminal_ees_feasible_days": feasible_days,
            "terminal_ees_feasible_ratio": float(feasible_days / n_days),
            "min_final_ees_soc": float(numeric_column(daily, "final_ees_soc", np.nan).min()),
            "sum_terminal_ees_shortage_kwh": float(terminal_shortage.sum()),
            "mean_terminal_ees_shortage_kwh_per_day": float(terminal_shortage.mean()),
            "sum_penalty_terminal_ees_soc": float(
                numeric_column(daily, "total_penalty_terminal_ees_soc").sum()
            ),
            "mean_total_grid_buy_kwh": float(numeric_column(daily, "total_grid_buy_kwh").mean()),
            "mean_total_grid_sell_kwh": float(numeric_column(daily, "total_grid_sell_kwh").mean()),
            "mean_total_ev_charge_kwh": float(numeric_column(daily, "total_ev_charge_kwh", np.nan).mean()),
            "mean_total_ev_discharge_kwh": float(
                numeric_column(daily, "total_ev_discharge_kwh", np.nan).mean()
            ),
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
        }
    )
    return row


def mean_std_text(mean_value: float, std_value: float) -> str:
    if pd.isna(mean_value):
        return "NA"
    if pd.isna(std_value):
        std_value = 0.0
    return f"{float(mean_value):.6g} +/- {float(std_value):.6g}"


def build_algorithm_summary(seed_df: pd.DataFrame, seeds: Iterable[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in ["TD3-original", "TD3-CUDA"]:
        group = seed_df[(seed_df["method"] == method) & (seed_df["test_complete"] == True)]
        row: dict[str, object] = {
            "method": method,
            "n_seeds_expected": len(list(seeds)),
            "n_complete_seeds": int(group["seed"].nunique()) if not group.empty else 0,
        }
        for metric in AGGREGATE_METRICS:
            values = pd.to_numeric(group.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
            mean_value = float(values.mean()) if not values.empty else np.nan
            std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_mean_std"] = mean_std_text(mean_value, std_value)
        rows.append(row)
    return pd.DataFrame(rows)


def build_vs_summary(algorithm_df: pd.DataFrame) -> pd.DataFrame:
    if algorithm_df.empty:
        return pd.DataFrame()
    by_method = algorithm_df.set_index("method")
    rows: list[dict[str, object]] = []
    for metric in AGGREGATE_METRICS:
        old_mean = by_method.at["TD3-original", f"{metric}_mean"] if "TD3-original" in by_method.index else np.nan
        old_std = by_method.at["TD3-original", f"{metric}_std"] if "TD3-original" in by_method.index else np.nan
        cuda_mean = by_method.at["TD3-CUDA", f"{metric}_mean"] if "TD3-CUDA" in by_method.index else np.nan
        cuda_std = by_method.at["TD3-CUDA", f"{metric}_std"] if "TD3-CUDA" in by_method.index else np.nan
        delta = float(cuda_mean - old_mean) if pd.notna(cuda_mean) and pd.notna(old_mean) else np.nan
        if pd.notna(delta) and pd.notna(old_mean) and abs(float(old_mean)) > 1e-12:
            delta_percent = float(delta / abs(float(old_mean)) * 100.0)
        else:
            delta_percent = np.nan
        rows.append(
            {
                "metric": metric,
                "original_mean": old_mean,
                "original_std": old_std,
                "cuda_mean": cuda_mean,
                "cuda_std": cuda_std,
                "cuda_minus_original": delta,
                "cuda_minus_original_percent": delta_percent,
                "original_mean_std": mean_std_text(old_mean, old_std),
                "cuda_mean_std": mean_std_text(cuda_mean, cuda_std),
            }
        )
    return pd.DataFrame(rows)


def condition_status(seed_df: pd.DataFrame, algorithm_df: pd.DataFrame, expected_seed_count: int) -> tuple[bool, list[str], dict[str, bool]]:
    by_method = algorithm_df.set_index("method")
    old_row = by_method.loc["TD3-original"] if "TD3-original" in by_method.index else pd.Series(dtype=object)
    cuda_row = by_method.loc["TD3-CUDA"] if "TD3-CUDA" in by_method.index else pd.Series(dtype=object)
    cuda_seeds = seed_df[(seed_df["method"] == "TD3-CUDA") & (seed_df["test_complete"] == True)]

    old_complete = int(old_row.get("n_complete_seeds", 0)) == expected_seed_count
    cuda_complete = int(cuda_row.get("n_complete_seeds", 0)) == expected_seed_count
    old_cost = old_row.get("mean_total_cost_plus_penalty_mean", np.nan)
    cuda_cost = cuda_row.get("mean_total_cost_plus_penalty_mean", np.nan)
    cuda_ratio = cuda_row.get("terminal_ees_feasible_ratio_mean", np.nan)
    cuda_shortage_per_day = cuda_row.get("mean_terminal_ees_shortage_kwh_per_day_mean", np.nan)
    cuda_depart_shortage = cuda_row.get("mean_total_depart_energy_shortage_kwh_mean", np.nan)

    full_feasible_seed_count = 0
    for _, row in cuda_seeds.iterrows():
        if int(row.get("terminal_ees_feasible_days", 0)) >= 60 and int(row.get("n_days", 0)) >= 60:
            full_feasible_seed_count += 1

    conditions = {
        "all_expected_original_seeds_complete": old_complete,
        "all_expected_cuda_seeds_complete": cuda_complete,
        "cost_not_worse_than_0p5pct": bool(
            pd.notna(cuda_cost) and pd.notna(old_cost) and float(cuda_cost) <= float(old_cost) * 1.005
        ),
        "ees_terminal_feasibility_ok": bool(
            (pd.notna(cuda_ratio) and float(cuda_ratio) >= 0.98)
            or full_feasible_seed_count >= 2
        ),
        "terminal_shortage_per_day_ok": bool(
            pd.notna(cuda_shortage_per_day) and float(cuda_shortage_per_day) <= 2.0
        ),
        "ev_departure_shortage_ok": bool(
            pd.notna(cuda_depart_shortage) and abs(float(cuda_depart_shortage)) <= 1e-3
        ),
        "unmet_load_ok": bool(
            not cuda_seeds.empty
            and abs(float(pd.to_numeric(cuda_seeds["total_unmet_e"], errors="coerce").fillna(np.inf).sum())) <= 1e-6
            and abs(float(pd.to_numeric(cuda_seeds["total_unmet_h"], errors="coerce").fillna(np.inf).sum())) <= 1e-6
            and abs(float(pd.to_numeric(cuda_seeds["total_unmet_c"], errors="coerce").fillna(np.inf).sum())) <= 1e-6
        ),
    }
    reasons = [name for name, ok in conditions.items() if not ok]
    candidate = all(conditions.values())
    return candidate, reasons, conditions


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


def write_figures(algorithm_df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: matplotlib is unavailable; skipping figures: {exc}")
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    by_method = algorithm_df.set_index("method")
    labels = ["TD3-original", "TD3-CUDA"]
    display_labels = ["Old TD3", "CUDA TD3"]

    def values(metric: str) -> tuple[list[float], list[float]]:
        means = [float(by_method.at[label, f"{metric}_mean"]) for label in labels]
        stds = [float(by_method.at[label, f"{metric}_std"]) for label in labels]
        return means, stds

    for metric, ylabel, filename in [
        (
            "mean_total_cost_plus_penalty",
            "Mean daily cost+penalty",
            "td3_cuda_vs_original_cost.png",
        ),
        (
            "terminal_ees_feasible_ratio",
            "EES terminal feasible ratio",
            "td3_cuda_vs_original_feasibility.png",
        ),
    ]:
        means, stds = values(metric)
        fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
        ax.bar(display_labels, means, yerr=stds, capsize=6, color=["#4c78a8", "#f58518"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "figures" / filename)
        plt.close(fig)


def write_report(
    *,
    seed_df: pd.DataFrame,
    algorithm_df: pd.DataFrame,
    vs_df: pd.DataFrame,
    seeds: list[int],
    output_dir: Path,
    candidate: bool,
    reasons: list[str],
    conditions: dict[str, bool],
) -> None:
    report_path = output_dir / "td3_cuda_comparison_report.md"
    completion_columns = [
        "method",
        "seed",
        "train_complete",
        "test_complete",
        "best_eval_timestep",
        "best_validation_reward",
        "mean_total_cost_plus_penalty",
        "terminal_ees_feasible_days",
        "model_path",
        "result_dir",
    ]
    mean_std_columns = [
        "metric",
        "original_mean_std",
        "cuda_mean_std",
        "cuda_minus_original",
        "cuda_minus_original_percent",
    ]
    selected_metrics = vs_df[
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
    ]
    condition_rows = pd.DataFrame(
        [{"condition": name, "passed": passed} for name, passed in conditions.items()]
    )

    lines = [
        "# TD3 CUDA multiseed comparison report",
        "",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "CUDA run names: td3_cuda_seed_42, td3_cuda_seed_2024, td3_cuda_seed_2025.",
        "",
        "Output roots are isolated under models/logs/results with the cuda_seed_* suffix.",
        "",
        "td3_cuda_run1 is excluded from this CUDA multiseed summary and remains only an old test record.",
        "",
        f"candidate_for_paper_main_result = {candidate}",
        "",
    ]
    if candidate:
        lines.append("CUDA multi-seed run satisfies the automatic replacement checks.")
    else:
        lines.append(
            "CUDA multi-seed run is valid as a reproducibility check, but it should not replace the original TD3 result as the paper main result."
        )
        if reasons:
            lines.append("")
            lines.append("Reasons not passed: " + ", ".join(reasons) + ".")

    lines.extend(
        [
            "",
            "## Completion And Seed Metrics",
            "",
            markdown_table(seed_df, completion_columns, float_digits=6),
            "",
            "## CUDA Mean +/- Std Versus Original TD3",
            "",
            markdown_table(selected_metrics, mean_std_columns, float_digits=6),
            "",
            "## Replacement Conditions",
            "",
            markdown_table(condition_rows, ["condition", "passed"], float_digits=6),
            "",
            "## Algorithm Summary",
            "",
            markdown_table(
                algorithm_df,
                [
                    "method",
                    "n_complete_seeds",
                    "mean_total_cost_plus_penalty_mean_std",
                    "terminal_ees_feasible_ratio_mean_std",
                    "sum_terminal_ees_shortage_kwh_mean_std",
                    "mean_total_depart_energy_shortage_kwh_mean_std",
                ],
                float_digits=6,
            ),
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = [int(seed) for seed in args.seeds]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for seed in seeds:
        rows.append(build_seed_row("TD3-CUDA", seed, cuda_paths(seed)))
    for seed in seeds:
        rows.append(build_seed_row("TD3-original", seed, original_paths(seed)))

    seed_df = pd.DataFrame(rows)
    for column in SEED_SUMMARY_COLUMNS:
        if column not in seed_df.columns:
            seed_df[column] = np.nan
    seed_df = seed_df[SEED_SUMMARY_COLUMNS]

    algorithm_df = build_algorithm_summary(seed_df, seeds)
    vs_df = build_vs_summary(algorithm_df)
    candidate, reasons, conditions = condition_status(seed_df, algorithm_df, len(seeds))

    seed_path = output_dir / "td3_cuda_seed_summary.csv"
    algorithm_path = output_dir / "td3_cuda_algorithm_summary.csv"
    vs_path = output_dir / "td3_cuda_vs_original_summary.csv"

    seed_df.to_csv(seed_path, index=False, encoding="utf-8-sig")
    algorithm_df.to_csv(algorithm_path, index=False, encoding="utf-8-sig")
    vs_df.to_csv(vs_path, index=False, encoding="utf-8-sig")
    write_report(
        seed_df=seed_df,
        algorithm_df=algorithm_df,
        vs_df=vs_df,
        seeds=seeds,
        output_dir=output_dir,
        candidate=candidate,
        reasons=reasons,
        conditions=conditions,
    )

    try:
        if set(["TD3-original", "TD3-CUDA"]).issubset(set(algorithm_df["method"])):
            write_figures(algorithm_df, output_dir)
    except Exception as exc:
        print(f"WARNING: could not create comparison figures: {exc}")

    print(f"[summary] seed summary: {seed_path}")
    print(f"[summary] algorithm summary: {algorithm_path}")
    print(f"[summary] vs original summary: {vs_path}")
    print(f"[summary] report: {output_dir / 'td3_cuda_comparison_report.md'}")
    print(algorithm_df.to_string(index=False))


if __name__ == "__main__":
    main()
