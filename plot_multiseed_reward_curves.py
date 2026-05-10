from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METHODS = ["TD3", "DDPG", "PPO"]
DEFAULT_SEEDS = [42, 2024, 2025]
WINDOW = 100
FIGURE_DIR = Path("results/comparison/figures")

TRAINING_PATHS = {
    "TD3": "results/td3_yearly_training_seed_{seed}/episode_summary.csv",
    "DDPG": "results/ddpg_yearly_training_seed_{seed}/episode_summary.csv",
    "PPO": "results/ppo_sb3_direct_training_seed_{seed}/episode_summary.csv",
}

EVAL_PATHS = {
    "TD3": "logs/td3_yearly_single_seed_{seed}/evaluations.npz",
    "DDPG": "logs/ddpg_yearly_single_seed_{seed}/evaluations.npz",
    "PPO": "logs/ppo_sb3_direct_seed_{seed}/evaluations.npz",
}

COLORS = {
    "TD3": "#1f77b4",
    "DDPG": "#2ca02c",
    "PPO": "#d62728",
}

PPO_NOTE = "PPO is an auxiliary baseline because it fails to satisfy the terminal EES SOC constraint in the current setting."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot multi-seed training and validation reward curves.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, help="Methods to plot: TD3 DDPG PPO.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Random seeds to plot.")
    parser.add_argument("--window", type=int, default=WINDOW, help="Moving-average window in episodes.")
    return parser.parse_args()


def normalize_methods(methods: list[str]) -> list[str]:
    normalized = []
    for method in methods:
        key = method.upper()
        if key not in TRAINING_PATHS:
            raise ValueError(f"Unknown method {method!r}. Valid methods: {', '.join(TRAINING_PATHS)}")
        normalized.append(key)
    return normalized


def read_training_curve(method: str, seed: int, window: int) -> pd.Series | None:
    path = Path(TRAINING_PATHS[method].format(seed=seed))
    if not path.exists():
        print(f"WARNING: missing training reward file: {path}")
        return None

    df = pd.read_csv(path)
    reward_column = "episode_reward_scaled" if "episode_reward_scaled" in df.columns else "episode_reward_raw"
    if reward_column not in df.columns:
        print(f"WARNING: no reward column in {path}")
        return None

    if "episode_idx" in df.columns:
        x = pd.to_numeric(df["episode_idx"], errors="coerce")
    else:
        x = pd.Series(np.arange(1, len(df) + 1), index=df.index)
    y = pd.to_numeric(df[reward_column], errors="coerce")
    valid = x.notna() & y.notna()
    if not valid.any():
        print(f"WARNING: no valid reward rows in {path}")
        return None

    series = pd.Series(y[valid].to_numpy(dtype=float), index=x[valid].astype(int).to_numpy())
    series = series.sort_index()
    return series.rolling(window=window, min_periods=1).mean()


def read_validation_curve(method: str, seed: int) -> pd.Series | None:
    path = Path(EVAL_PATHS[method].format(seed=seed))
    if not path.exists():
        print(f"WARNING: missing validation reward file: {path}")
        return None

    data = np.load(path)
    if "timesteps" not in data or "results" not in data:
        print(f"WARNING: evaluations.npz missing timesteps/results: {path}")
        return None
    timesteps = np.asarray(data["timesteps"], dtype=int)
    results = np.asarray(data["results"], dtype=float)
    if results.ndim == 1:
        rewards = results
    else:
        rewards = np.nanmean(results, axis=1)
    if len(timesteps) != len(rewards):
        print(f"WARNING: mismatched validation arrays in {path}")
        return None
    return pd.Series(rewards, index=timesteps).sort_index()


def aggregate_curves(curves: list[pd.Series]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not curves:
        return None
    frame = pd.concat(curves, axis=1).sort_index()
    mean = frame.mean(axis=1, skipna=True)
    std = frame.std(axis=1, skipna=True, ddof=1).fillna(0.0)
    return frame.index.to_numpy(dtype=float), mean.to_numpy(dtype=float), std.to_numpy(dtype=float)


def collect_curves(methods: list[str], seeds: list[int], window: int):
    training = {}
    validation = {}
    for method in methods:
        training_curves = []
        validation_curves = []
        for seed in seeds:
            training_curve = read_training_curve(method, seed, window)
            if training_curve is not None:
                training_curves.append(training_curve)
            validation_curve = read_validation_curve(method, seed)
            if validation_curve is not None:
                validation_curves.append(validation_curve)
        training[method] = aggregate_curves(training_curves)
        validation[method] = aggregate_curves(validation_curves)
    return training, validation


def plot_curve_set(curve_set, *, title: str, xlabel: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
    for method, aggregate in curve_set.items():
        if aggregate is None:
            continue
        x, mean, std = aggregate
        color = COLORS.get(method, None)
        ax.plot(x, mean, label=method, color=color, linewidth=2.0)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.text(0.01, 0.02, PPO_NOTE, transform=ax.transAxes, fontsize=8, va="bottom", ha="left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved: {output_path}")


def plot_combined(training, validation, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=300)
    panels = [
        (axes[0], training, "Training reward curves over 3 random seeds", "Episode", "Smoothed reward"),
        (axes[1], validation, "Validation reward curves over 3 random seeds", "Timesteps", "Validation reward"),
    ]
    for ax, curve_set, title, xlabel, ylabel in panels:
        for method, aggregate in curve_set.items():
            if aggregate is None:
                continue
            x, mean, std = aggregate
            color = COLORS.get(method, None)
            ax.plot(x, mean, label=method, color=color, linewidth=2.0)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.text(0.01, 0.01, PPO_NOTE, fontsize=8, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved: {output_path}")


def main() -> None:
    args = parse_args()
    methods = normalize_methods(args.methods)
    seeds = [int(seed) for seed in args.seeds]
    training, validation = collect_curves(methods, seeds, int(args.window))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_curve_set(
        training,
        title="Training reward curves over 3 random seeds",
        xlabel="Episode",
        ylabel=f"Reward, moving average window={args.window}",
        output_path=FIGURE_DIR / "multiseed_training_reward.png",
    )
    plot_curve_set(
        validation,
        title="Validation reward curves over 3 random seeds",
        xlabel="Timesteps",
        ylabel="Mean validation reward",
        output_path=FIGURE_DIR / "multiseed_validation_reward.png",
    )
    plot_combined(training, validation, FIGURE_DIR / "multiseed_learning_curves_combined.png")


if __name__ == "__main__":
    main()
