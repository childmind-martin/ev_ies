from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:
    EventAccumulator = None


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_TD3_TB_LOG_DIR = BASE_DIR / "tb" / "td3_yearly_single"
DEFAULT_TD3_EVAL_NPZ_PATH = BASE_DIR / "logs" / "td3_yearly_single" / "evaluations.npz"
DEFAULT_TD3_TRAINING_XLSX_PATH = BASE_DIR / "results" / "td3_yearly_training" / "td3_training_export.xlsx"

DEFAULT_PPO_TB_LOG_DIR = BASE_DIR / "tb" / "ppo_sb3_direct"
DEFAULT_PPO_EVAL_NPZ_PATH = BASE_DIR / "logs" / "ppo_sb3_direct" / "evaluations.npz"
DEFAULT_PPO_TRAINING_XLSX_PATH = (
    BASE_DIR / "results" / "ppo_sb3_direct_training" / "ppo_sb3_direct_training_export.xlsx"
)

DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "td3_ppo_reward_comparison"

TIMESTEP_SCALE = 1e4
TRAIN_REWARD_SMOOTH_WINDOW = 100
FALLBACK_REWARD_SMOOTH_WINDOW = 5
EVAL_REWARD_SMOOTH_WINDOW = 3

SCALAR_TAGS = {
    "episode_reward_scaled": ["custom/episode_reward_scaled"],
    "sb3_reward_mean": ["rollout/ep_rew_mean"],
    "eval_reward": ["eval/mean_reward"],
}

ALGO_STYLES = {
    "TD3": {
        "color": "#1F77B4",
        "raw_color": "#B7CCE8",
        "fill_color": "#1F77B4",
    },
    "PPO": {
        "color": "#FF7F0E",
        "raw_color": "#F6C89F",
        "fill_color": "#FF7F0E",
    },
}


def configure_matplotlib() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.copy()
    left_pad = window // 2
    right_pad = window - 1 - left_pad
    padded = np.pad(values, (left_pad, right_pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def scaled_steps(steps: np.ndarray) -> np.ndarray:
    return steps.astype(np.float64) / TIMESTEP_SCALE


def resolve_latest_event_scope(tb_log_dir: Path) -> Tuple[Path, List[Path]]:
    if not tb_log_dir.exists():
        return tb_log_dir, []

    run_dirs = [path for path in tb_log_dir.iterdir() if path.is_dir()]
    search_root = max(run_dirs, key=lambda path: path.stat().st_mtime) if run_dirs else tb_log_dir
    event_files = sorted(search_root.rglob("events.out.tfevents.*"))
    if not event_files and search_root != tb_log_dir:
        event_files = sorted(tb_log_dir.rglob("events.out.tfevents.*"))
        search_root = tb_log_dir
    return search_root, event_files


def dedupe_series(values: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    step_to_value: dict[int, float] = {}
    for step, value in values:
        step_to_value[int(step)] = float(value)
    return sorted(step_to_value.items(), key=lambda item: item[0])


def load_scalar_series(tb_log_dir: Path) -> Tuple[Dict[str, List[Tuple[int, float]]], Path | None]:
    if EventAccumulator is None:
        print("TensorBoard is not available; TensorBoard fallback curves will be skipped.")
        return {}, None

    run_dir, event_files = resolve_latest_event_scope(tb_log_dir)
    if not event_files:
        print(f"No TensorBoard event files found under: {tb_log_dir}")
        return {}, run_dir

    series: Dict[str, List[Tuple[int, float]]] = {key: [] for key in SCALAR_TAGS}
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        scalar_tags = set(accumulator.Tags().get("scalars", []))
        for key, tag_names in SCALAR_TAGS.items():
            for tag_name in tag_names:
                if tag_name not in scalar_tags:
                    continue
                values = accumulator.Scalars(tag_name)
                series[key].extend((int(item.step), float(item.value)) for item in values)
                break

    return {key: dedupe_series(values) for key, values in series.items()}, run_dir


def load_episode_reward_from_xlsx(training_xlsx_path: Path) -> List[Tuple[int, float]]:
    if not training_xlsx_path.exists():
        return []
    if pd is None:
        print("pandas is not available; Excel-based episode reward loading will be skipped.")
        return []

    try:
        episode_df = pd.read_excel(training_xlsx_path, sheet_name="episode_summary")
    except Exception as exc:
        print(f"Failed to read episode_summary from {training_xlsx_path}: {exc}")
        return []

    required_columns = {"global_step_end", "episode_reward_scaled"}
    if not required_columns.issubset(set(episode_df.columns)):
        print(
            f"Sheet 'episode_summary' in {training_xlsx_path} is missing required columns: "
            f"{sorted(required_columns - set(episode_df.columns))}"
        )
        return []

    step_values = pd.to_numeric(episode_df["global_step_end"], errors="coerce")
    reward_values = pd.to_numeric(episode_df["episode_reward_scaled"], errors="coerce")
    valid_mask = step_values.notna() & reward_values.notna()
    if not bool(valid_mask.any()):
        return []

    steps = step_values[valid_mask].to_numpy(dtype=np.int64)
    values = reward_values[valid_mask].to_numpy(dtype=np.float64)
    return dedupe_series(list(zip(steps.tolist(), values.tolist())))


def load_eval_npz(eval_npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not eval_npz_path.exists():
        return None
    data = np.load(eval_npz_path)
    if "timesteps" not in data or "results" not in data:
        return None
    timesteps = np.asarray(data["timesteps"], dtype=np.int64)
    results = np.asarray(data["results"], dtype=np.float64)
    return timesteps, results.mean(axis=1), results.std(axis=1)


def series_to_xy(series: List[Tuple[int, float]]) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.array([step for step, _ in series], dtype=np.int64),
        np.array([value for _, value in series], dtype=np.float64),
    )


def choose_training_reward_series(
    *,
    training_xlsx_path: Path,
    tb_scalar_series: Dict[str, List[Tuple[int, float]]],
) -> Tuple[List[Tuple[int, float]], str]:
    xlsx_series = load_episode_reward_from_xlsx(training_xlsx_path)
    if xlsx_series:
        return xlsx_series, f"Excel episode_summary: {training_xlsx_path}"

    custom_series = tb_scalar_series.get("episode_reward_scaled", [])
    if custom_series:
        return custom_series, "TensorBoard: custom/episode_reward_scaled"

    fallback_series = tb_scalar_series.get("sb3_reward_mean", [])
    if fallback_series:
        return fallback_series, "TensorBoard fallback: rollout/ep_rew_mean"

    return [], "unavailable"


def choose_eval_reward_source(
    *,
    eval_npz_path: Path,
    tb_scalar_series: Dict[str, List[Tuple[int, float]]],
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray] | None, List[Tuple[int, float]], str]:
    eval_data = load_eval_npz(eval_npz_path)
    if eval_data is not None:
        return eval_data, [], f"NPZ: {eval_npz_path}"

    fallback_series = tb_scalar_series.get("eval_reward", [])
    if fallback_series:
        return None, fallback_series, "TensorBoard fallback: eval/mean_reward"

    return None, [], "unavailable"


def draw_empty_panel(ax: plt.Axes, title: str, ylabel: str, message: str = "No data") -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Timesteps (x1e4)")
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#666666")
    ax.grid(True, alpha=0.15)


def plot_training_comparison_panel(
    ax: plt.Axes,
    algorithm_to_series: Dict[str, List[Tuple[int, float]]],
    algorithm_to_source: Dict[str, str],
) -> None:
    plotted = False
    for algorithm_name, series in algorithm_to_series.items():
        if not series:
            continue
        x, y = series_to_xy(series)
        style = ALGO_STYLES[algorithm_name]
        ax.plot(
            scaled_steps(x),
            y,
            color=style["raw_color"],
            linewidth=1.0,
            alpha=0.8,
            label=f"{algorithm_name} raw",
        )
        source_text = algorithm_to_source.get(algorithm_name, "")
        smooth_window = (
            TRAIN_REWARD_SMOOTH_WINDOW
            if ("Excel" in source_text or "custom/episode_reward_scaled" in source_text)
            else FALLBACK_REWARD_SMOOTH_WINDOW
        )
        if len(y) >= smooth_window:
            y = moving_average(y, smooth_window)
        ax.plot(
            scaled_steps(x),
            y,
            color=style["color"],
            linewidth=2.2,
            label=f"{algorithm_name} smooth",
        )
        plotted = True

    if not plotted:
        draw_empty_panel(ax, "Training reward comparison", "Episode reward")
        return

    ax.set_title("Training reward comparison")
    ax.set_xlabel("Timesteps (x1e4)")
    ax.set_ylabel("Episode reward")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


def plot_validation_comparison_panel(
    ax: plt.Axes,
    *,
    eval_npz_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray] | None],
    eval_fallback_series: Dict[str, List[Tuple[int, float]]],
) -> None:
    plotted = False
    for algorithm_name in ("TD3", "PPO"):
        style = ALGO_STYLES[algorithm_name]
        eval_data = eval_npz_data.get(algorithm_name)
        if eval_data is not None:
            x, mean_reward, std_reward = eval_data
            x_scaled = scaled_steps(x)
            best_idx = int(np.argmax(mean_reward))
            ax.plot(
                x_scaled,
                mean_reward,
                color=style["color"],
                linewidth=2.0,
                marker="o",
                label=f"{algorithm_name} mean reward",
            )
            ax.fill_between(
                x_scaled,
                mean_reward - std_reward,
                mean_reward + std_reward,
                color=style["fill_color"],
                alpha=0.12,
                label=f"{algorithm_name} mean +/- std",
            )
            ax.scatter(
                [x_scaled[best_idx]],
                [mean_reward[best_idx]],
                color=style["color"],
                s=42,
                zorder=5,
            )
            plotted = True
            continue

        fallback_series = eval_fallback_series.get(algorithm_name, [])
        if fallback_series:
            x, y = series_to_xy(fallback_series)
            if len(y) >= EVAL_REWARD_SMOOTH_WINDOW:
                y_smooth = moving_average(y, EVAL_REWARD_SMOOTH_WINDOW)
            else:
                y_smooth = y
            ax.plot(
                scaled_steps(x),
                y_smooth,
                color=style["color"],
                linewidth=2.0,
                label=f"{algorithm_name} eval fallback",
            )
            plotted = True

    if not plotted:
        draw_empty_panel(ax, "Validation reward comparison", "Mean reward")
        return

    ax.set_title("Validation reward comparison")
    ax.set_xlabel("Timesteps (x1e4)")
    ax.set_ylabel("Mean reward")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


def plot_reward_comparison(
    *,
    td3_train_series: List[Tuple[int, float]],
    ppo_train_series: List[Tuple[int, float]],
    td3_train_source: str,
    ppo_train_source: str,
    td3_eval_data: Tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    ppo_eval_data: Tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    td3_eval_fallback_series: List[Tuple[int, float]],
    ppo_eval_fallback_series: List[Tuple[int, float]],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8))

    plot_training_comparison_panel(
        axes[0],
        {
            "TD3": td3_train_series,
            "PPO": ppo_train_series,
        },
        {
            "TD3": td3_train_source,
            "PPO": ppo_train_source,
        },
    )
    plot_validation_comparison_panel(
        axes[1],
        eval_npz_data={"TD3": td3_eval_data, "PPO": ppo_eval_data},
        eval_fallback_series={"TD3": td3_eval_fallback_series, "PPO": ppo_eval_fallback_series},
    )

    fig.suptitle("TD3 vs PPO reward comparison", fontsize=15)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot TD3 and PPO reward curves on one comparison figure."
    )
    parser.add_argument("--td3-tb-log-dir", type=Path, default=DEFAULT_TD3_TB_LOG_DIR)
    parser.add_argument("--td3-eval-npz", type=Path, default=DEFAULT_TD3_EVAL_NPZ_PATH)
    parser.add_argument("--td3-training-xlsx", type=Path, default=DEFAULT_TD3_TRAINING_XLSX_PATH)
    parser.add_argument("--ppo-tb-log-dir", type=Path, default=DEFAULT_PPO_TB_LOG_DIR)
    parser.add_argument("--ppo-eval-npz", type=Path, default=DEFAULT_PPO_EVAL_NPZ_PATH)
    parser.add_argument("--ppo-training-xlsx", type=Path, default=DEFAULT_PPO_TRAINING_XLSX_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    td3_tb_series, td3_run_dir = load_scalar_series(args.td3_tb_log_dir)
    ppo_tb_series, ppo_run_dir = load_scalar_series(args.ppo_tb_log_dir)

    td3_train_series, td3_train_source = choose_training_reward_series(
        training_xlsx_path=args.td3_training_xlsx,
        tb_scalar_series=td3_tb_series,
    )
    ppo_train_series, ppo_train_source = choose_training_reward_series(
        training_xlsx_path=args.ppo_training_xlsx,
        tb_scalar_series=ppo_tb_series,
    )

    td3_eval_data, td3_eval_fallback_series, td3_eval_source = choose_eval_reward_source(
        eval_npz_path=args.td3_eval_npz,
        tb_scalar_series=td3_tb_series,
    )
    ppo_eval_data, ppo_eval_fallback_series, ppo_eval_source = choose_eval_reward_source(
        eval_npz_path=args.ppo_eval_npz,
        tb_scalar_series=ppo_tb_series,
    )

    output_path = args.output_dir / "td3_vs_ppo_reward_comparison.png"
    plot_reward_comparison(
        td3_train_series=td3_train_series,
        ppo_train_series=ppo_train_series,
        td3_train_source=td3_train_source,
        ppo_train_source=ppo_train_source,
        td3_eval_data=td3_eval_data,
        ppo_eval_data=ppo_eval_data,
        td3_eval_fallback_series=td3_eval_fallback_series,
        ppo_eval_fallback_series=ppo_eval_fallback_series,
        save_path=output_path,
    )

    print(f"TD3 vs PPO reward comparison plot saved to: {output_path}")
    if td3_run_dir is not None:
        print(f"TD3 TensorBoard event scope: {td3_run_dir}")
    if ppo_run_dir is not None:
        print(f"PPO TensorBoard event scope: {ppo_run_dir}")
    print(f"TD3 training reward source: {td3_train_source}")
    print(f"PPO training reward source: {ppo_train_source}")
    print(f"TD3 validation reward source: {td3_eval_source}")
    print(f"PPO validation reward source: {ppo_eval_source}")


if __name__ == "__main__":
    main()
