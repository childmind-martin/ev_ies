from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


OUTPUT_ROOT = Path("results/v2g_ratio_sensitivity")
DEFAULT_SEEDS = [42]
BASE_RATIO = 0.279
PNG_DPI = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot V2G ratio sensitivity figures.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def load_seed_summary(output_root: Path, seeds: list[int]) -> pd.DataFrame:
    path = output_root / "v2g_ratio_seed_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run build_v2g_ratio_sensitivity.py first.")
    df = pd.read_csv(path)
    df = df[df["seed"].astype(int).isin([int(seed) for seed in seeds])].copy()
    if df.empty:
        raise ValueError(f"No V2G ratio sensitivity rows found for seeds={seeds}")
    return df


def metric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def build_plot_df(seed_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in seed_df.columns
        if col
        not in {
            "run_name",
            "ratio_dir",
            "model_path",
            "daily_summary_path",
            "timeseries_detail_path",
            "runtime_summary_path",
        }
    ]
    grouped = (
        seed_df.groupby("v2g_ratio_target", sort=True)[numeric_cols]
        .mean(numeric_only=True)
        .reset_index(drop=True)
    )
    grouped["v2g_ratio_target"] = sorted(seed_df["v2g_ratio_target"].astype(float).unique())
    grouped = grouped.sort_values("v2g_ratio_target").reset_index(drop=True)
    return grouped


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def mark_base(ax: plt.Axes) -> None:
    ax.axvline(BASE_RATIO, color="#4D4D4D", linestyle="--", linewidth=1.2, alpha=0.8)
    ymin, ymax = ax.get_ylim()
    y = ymin + (ymax - ymin) * 0.95
    ax.text(
        BASE_RATIO,
        y,
        "base 27.9%",
        rotation=90,
        va="top",
        ha="right",
        fontsize=8,
        color="#4D4D4D",
    )


def format_ratio_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("V2G participation ratio")
    ax.grid(True, axis="y", alpha=0.25)


def plot_cost(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = metric_series(df, "v2g_ratio_target")
    specs = [
        ("mean_total_system_cost", "System cost", "#4E79A7"),
        ("mean_total_cost_plus_penalty", "Cost + penalty", "#E15759"),
        ("mean_total_penalties", "Penalties", "#F28E2B"),
    ]
    for column, label, color in specs:
        ax.plot(x, metric_series(df, column), marker="o", linewidth=2.0, label=label, color=color)
    ax.set_ylabel("Mean daily cost")
    ax.set_title("V2G participation ratio vs cost")
    format_ratio_axis(ax)
    mark_base(ax)
    ax.legend(frameon=False)
    path = output_dir / "v2g_ratio_cost.png"
    save_fig(fig, path)
    return path


def plot_grid_interaction(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = metric_series(df, "v2g_ratio_target")
    ax.plot(x, metric_series(df, "mean_total_grid_buy_kwh"), marker="o", linewidth=2.0, label="Grid buy", color="#59A14F")
    ax.plot(x, metric_series(df, "mean_total_grid_sell_kwh"), marker="s", linewidth=2.0, label="Grid sell", color="#F28E2B")
    ax.set_ylabel("Mean daily energy (kWh)")
    ax.set_title("V2G participation ratio vs grid interaction")
    format_ratio_axis(ax)
    mark_base(ax)
    ax.legend(frameon=False)
    path = output_dir / "v2g_ratio_grid_interaction.png"
    save_fig(fig, path)
    return path


def plot_ev_discharge_peak_shaving(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = metric_series(df, "v2g_ratio_target")
    ax.plot(
        x,
        metric_series(df, "mean_total_ev_discharge_kwh"),
        marker="o",
        linewidth=2.0,
        label="EV discharge",
        color="#B07AA1",
    )
    ax.plot(
        x,
        metric_series(df, "mean_total_storage_peak_shaved_kwh"),
        marker="s",
        linewidth=2.0,
        label="Storage peak shaved",
        color="#4E79A7",
    )
    ax.set_ylabel("Mean daily energy (kWh)")
    ax.set_title("V2G participation ratio vs EV discharge and peak shaving")
    format_ratio_axis(ax)
    mark_base(ax)
    ax.legend(frameon=False)
    path = output_dir / "v2g_ratio_ev_discharge_peak_shaving.png"
    save_fig(fig, path)
    return path


def plot_constraints(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = metric_series(df, "v2g_ratio_target")
    ax.plot(
        x,
        metric_series(df, "sum_terminal_ees_shortage_kwh"),
        marker="o",
        linewidth=2.0,
        label="EES terminal shortage",
        color="#E15759",
    )
    ax.plot(
        x,
        metric_series(df, "sum_depart_energy_shortage_kwh"),
        marker="s",
        linewidth=2.0,
        label="EV departure shortage",
        color="#F28E2B",
    )
    ax.plot(
        x,
        metric_series(df, "sum_ev_export_overlap_kwh"),
        marker="^",
        linewidth=2.0,
        label="EV export overlap",
        color="#76B7B2",
    )
    ax.set_ylabel("Total energy over tested days (kWh)")
    ax.set_title("V2G participation ratio vs constraints and guards")
    format_ratio_axis(ax)
    mark_base(ax)

    ax_ratio = ax.twinx()
    ax_ratio.plot(
        x,
        metric_series(df, "terminal_ees_feasible_ratio"),
        marker="D",
        linewidth=2.0,
        label="EES feasible ratio",
        color="#59A14F",
    )
    ax_ratio.set_ylabel("EES terminal feasible ratio")
    ax_ratio.set_ylim(-0.02, 1.05)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_ratio.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="best")
    path = output_dir / "v2g_ratio_constraints.png"
    save_fig(fig, path)
    return path


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    seeds = [int(seed) for seed in args.seeds]
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    seed_df = load_seed_summary(output_root, seeds)
    plot_df = build_plot_df(seed_df)

    paths = [
        plot_cost(plot_df, figures_dir),
        plot_grid_interaction(plot_df, figures_dir),
        plot_ev_discharge_peak_shaving(plot_df, figures_dir),
        plot_constraints(plot_df, figures_dir),
    ]
    for path in paths:
        print(f"[plot] {path}")


if __name__ == "__main__":
    main()
