from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIO_ORDER = ("uncontrolled_charging", "ordered_charging", "v2g")
SCENARIO_LABELS = {
    "uncontrolled_charging": "A uncontrolled",
    "ordered_charging": "B ordered",
    "v2g": "C v2g",
}
DEFAULT_SEEDS = [42]
OUTPUT_DIR = Path("results/ev_scenario_comparison")
PNG_DPI = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot EV scenario comparison figures.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def load_seed_summary(output_dir: Path, seeds: list[int]) -> pd.DataFrame:
    path = output_dir / "ev_scenario_seed_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run build_ev_scenario_comparison.py first."
        )
    df = pd.read_csv(path)
    df = df[df["seed"].astype(int).isin([int(seed) for seed in seeds])].copy()
    if df.empty:
        raise ValueError(f"No EV scenario rows found for seeds={seeds}")
    return df


def metric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def build_plot_df(seed_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in seed_df.columns
        if col not in {"ev_scenario", "scenario_label", "daily_summary_path", "timeseries_detail_path"}
    ]
    grouped = (
        seed_df.groupby("ev_scenario", sort=False)[numeric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    grouped["ev_scenario"] = pd.Categorical(
        grouped["ev_scenario"], categories=SCENARIO_ORDER, ordered=True
    )
    grouped = grouped.sort_values("ev_scenario").reset_index(drop=True)
    grouped["label"] = grouped["ev_scenario"].astype(str).map(SCENARIO_LABELS)
    return grouped


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cost_breakdown(df: pd.DataFrame, output_dir: Path) -> None:
    components = [
        ("mean_total_system_cost", "System cost", "#4E79A7"),
        ("mean_other_penalties", "Other penalties", "#BAB0AC"),
        ("mean_total_penalty_depart_energy", "EV departure shortage", "#E15759"),
        ("mean_total_penalty_depart_risk", "EV depart risk", "#B07AA1"),
        ("mean_total_penalty_ev_export_guard", "EV export guard", "#76B7B2"),
        ("mean_total_penalty_terminal_ees_soc", "EES terminal shortage", "#F28E2B"),
    ]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(df), dtype=float)
    for column, label, color in components:
        values = metric_series(df, column).to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=label, color=color, width=0.64)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=0)
    ax.set_ylabel("Mean daily cost + penalty")
    ax.set_title("EV scenario cost and penalty breakdown")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, output_dir / "ev_scenario_cost_breakdown.png")


def plot_grid_interaction(df: pd.DataFrame, output_dir: Path) -> None:
    x = np.arange(len(df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(
        x - width / 2,
        metric_series(df, "mean_total_grid_buy_kwh"),
        width,
        label="Grid buy",
        color="#59A14F",
    )
    ax.bar(
        x + width / 2,
        metric_series(df, "mean_total_grid_sell_kwh"),
        width,
        label="Grid sell",
        color="#F28E2B",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"])
    ax.set_ylabel("Mean daily energy (kWh)")
    ax.set_title("Grid interaction")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, output_dir / "ev_scenario_grid_interaction.png")


def plot_ev_energy(df: pd.DataFrame, output_dir: Path) -> None:
    x = np.arange(len(df))
    width = 0.2
    specs = [
        ("mean_total_ev_charge_kwh", "EV charge", "#4E79A7"),
        ("mean_total_ev_discharge_kwh", "EV discharge", "#B07AA1"),
        ("mean_total_depart_energy_shortage_kwh", "Departure shortage", "#E15759"),
        ("mean_ev_export_overlap_kwh", "EV export overlap", "#76B7B2"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for idx, (column, label, color) in enumerate(specs):
        offset = (idx - 1.5) * width
        ax.bar(x + offset, metric_series(df, column), width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"])
    ax.set_ylabel("Mean daily energy (kWh)")
    ax.set_title("EV energy, shortage, and export guard exposure")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, output_dir / "ev_scenario_ev_energy.png")


def plot_terminal_ees_soc(df: pd.DataFrame, output_dir: Path) -> None:
    x = np.arange(len(df))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].bar(
        x - width / 2,
        metric_series(df, "mean_final_ees_soc"),
        width,
        label="Mean final SOC",
        color="#4E79A7",
    )
    axes[0].bar(
        x + width / 2,
        metric_series(df, "min_final_ees_soc"),
        width,
        label="Min final SOC",
        color="#F28E2B",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["label"], rotation=15, ha="right")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("SOC")
    axes[0].set_title("Terminal EES SOC")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        x,
        metric_series(df, "mean_terminal_ees_shortage_kwh"),
        width=0.52,
        label="Mean terminal shortage",
        color="#E15759",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["label"], rotation=15, ha="right")
    axes[1].set_ylabel("Mean daily shortage (kWh)")
    axes[1].set_title("Terminal shortage and feasibility")
    axes[1].grid(axis="y", alpha=0.25)
    ax_ratio = axes[1].twinx()
    ax_ratio.plot(
        x,
        metric_series(df, "terminal_ees_feasible_ratio"),
        marker="o",
        color="#59A14F",
        label="Feasible ratio",
    )
    ax_ratio.set_ylim(0.0, 1.05)
    ax_ratio.set_ylabel("Feasible ratio")

    lines, labels = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax_ratio.get_legend_handles_labels()
    axes[1].legend(lines + lines2, labels + labels2, frameon=False, fontsize=8)
    save_fig(fig, output_dir / "ev_scenario_terminal_ees_soc.png")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in args.seeds]

    seed_df = load_seed_summary(output_dir, seeds)
    plot_df = build_plot_df(seed_df)

    plot_cost_breakdown(plot_df, output_dir)
    plot_grid_interaction(plot_df, output_dir)
    plot_ev_energy(plot_df, output_dir)
    plot_terminal_ees_soc(plot_df, output_dir)

    print(f"[plot] {output_dir / 'ev_scenario_cost_breakdown.png'}")
    print(f"[plot] {output_dir / 'ev_scenario_grid_interaction.png'}")
    print(f"[plot] {output_dir / 'ev_scenario_ev_energy.png'}")
    print(f"[plot] {output_dir / 'ev_scenario_terminal_ees_soc.png'}")


if __name__ == "__main__":
    main()
