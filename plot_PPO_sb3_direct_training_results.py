from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
DEFAULT_TB_LOG_DIR = BASE_DIR / "tb" / "ppo_sb3_direct"
DEFAULT_EVAL_NPZ_PATH = BASE_DIR / "logs" / "ppo_sb3_direct" / "evaluations.npz"
DEFAULT_TRAINING_XLSX_PATH = (
    BASE_DIR / "results" / "ppo_sb3_direct_training" / "ppo_sb3_direct_training_export.xlsx"
)
DEFAULT_CONFIG_CSV_PATH = BASE_DIR / "results" / "ppo_sb3_direct_training" / "train_config.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "ppo_sb3_direct_training" / "plots"

TIMESTEP_SCALE = 1e4
TRAIN_REWARD_SMOOTH_WINDOW = 100
LOSS_SMOOTH_WINDOW = 25
FALLBACK_REWARD_SMOOTH_WINDOW = 5
EPISODE_SMOOTH_WINDOW = 100
DIAG_SMOOTH_WINDOW = 15

SCALAR_TAGS = {
    "sb3_reward_mean": ["rollout/ep_rew_mean"],
    "eval_reward": ["eval/mean_reward"],
    "episode_reward_scaled": ["custom/episode_reward_scaled"],
    "policy_gradient_loss": ["train/policy_gradient_loss"],
    "value_loss": ["train/value_loss"],
    "loss": ["train/loss"],
    "approx_kl": ["train/approx_kl"],
    "clip_fraction": ["train/clip_fraction"],
    "entropy_loss": ["train/entropy_loss"],
    "explained_variance": ["train/explained_variance"],
    "learning_rate": ["train/learning_rate"],
    "episode_system_cost": ["custom/episode_system_cost"],
    "episode_penalty_cost": ["custom/episode_penalty_cost"],
    "episode_reward_raw": ["custom/episode_reward_raw"],
    "episode_guide_reward": ["custom/episode_guide_reward"],
    "episode_penalty_unserved_e": ["custom/episode_penalty_unserved_e"],
    "episode_penalty_unserved_h": ["custom/episode_penalty_unserved_h"],
    "episode_penalty_unserved_c": ["custom/episode_penalty_unserved_c"],
    "episode_penalty_depart_energy": ["custom/episode_penalty_depart_energy"],
    "episode_penalty_depart_risk": ["custom/episode_penalty_depart_risk"],
    "episode_terminal_ees_shortage_kwh": ["custom/episode_terminal_ees_shortage_kwh"],
    "episode_penalty_terminal_ees_soc": ["custom/episode_penalty_terminal_ees_soc"],
    "episode_penalty_export_e": ["custom/episode_penalty_export_e"],
    "episode_penalty_ev_export_guard": ["custom/episode_penalty_ev_export_guard"],
    "episode_ev_peak_shaved_kwh": ["custom/episode_ev_peak_shaved_kwh"],
    "episode_ev_charge_rewarded_kwh": ["custom/episode_ev_charge_rewarded_kwh"],
    "episode_ev_flex_target_charge_kwh": ["custom/episode_ev_flex_target_charge_kwh"],
    "episode_ev_buffer_charge_kwh": ["custom/episode_ev_buffer_charge_kwh"],
    "episode_low_value_charge_kwh": ["custom/episode_low_value_charge_kwh"],
}

EPISODE_SUMMARY_KEYS = (
    "episode_reward_scaled",
    "episode_system_cost",
    "episode_penalty_cost",
    "episode_reward_raw",
    "episode_guide_reward",
    "episode_penalty_unserved_e",
    "episode_penalty_unserved_h",
    "episode_penalty_unserved_c",
    "episode_penalty_depart_energy",
    "episode_penalty_depart_risk",
    "episode_terminal_ees_shortage_kwh",
    "episode_penalty_terminal_ees_soc",
    "episode_penalty_export_e",
    "episode_penalty_ev_export_guard",
    "episode_ev_peak_shaved_kwh",
    "episode_ev_charge_rewarded_kwh",
    "episode_ev_flex_target_charge_kwh",
    "episode_ev_buffer_charge_kwh",
    "episode_low_value_charge_kwh",
)

PENALTY_PANEL_KEYS = [
    ("episode_penalty_unserved_e", "Unserved electric", "#1F77B4"),
    ("episode_penalty_unserved_h", "Unserved heat", "#17BECF"),
    ("episode_penalty_unserved_c", "Unserved cooling", "#2CA02C"),
    ("episode_penalty_depart_energy", "Departure energy", "#D62728"),
    ("episode_penalty_depart_risk", "Departure risk", "#FF7F0E"),
    ("episode_penalty_terminal_ees_soc", "EES terminal SOC", "#9467BD"),
    ("episode_penalty_export_e", "Export", "#8C564B"),
    ("episode_penalty_ev_export_guard", "EV export guard", "#E377C2"),
]

EV_DIAGNOSTIC_KEYS = [
    ("episode_ev_peak_shaved_kwh", "EV peak shaved", "#1F77B4"),
    ("episode_ev_charge_rewarded_kwh", "EV charge rewarded", "#2CA02C"),
    ("episode_ev_flex_target_charge_kwh", "EV flex target charge", "#17BECF"),
    ("episode_ev_buffer_charge_kwh", "EV buffer charge", "#BCBD22"),
    ("episode_low_value_charge_kwh", "Low-value charge", "#7F7F7F"),
]

PPO_CLIPPING_KEYS = [
    ("approx_kl", "Approx KL", "#1F77B4"),
    ("clip_fraction", "Clip fraction", "#2CA02C"),
]

PPO_OPTIMIZATION_KEYS = [
    ("entropy_loss", "Entropy loss", "#D62728"),
    ("loss", "Total loss", "#8C564B"),
    ("explained_variance", "Explained variance", "#17BECF"),
]


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
    step_to_value = {}
    for step, value in values:
        step_to_value[int(step)] = float(value)
    return sorted(step_to_value.items(), key=lambda item: item[0])


def load_scalar_series(tb_log_dir: Path) -> Tuple[Dict[str, List[Tuple[int, float]]], Path | None]:
    if EventAccumulator is None:
        print("TensorBoard is not available; TensorBoard curves will be skipped.")
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

    deduped: Dict[str, List[Tuple[int, float]]] = {}
    for key, values in series.items():
        deduped[key] = dedupe_series(values)
    return deduped, run_dir


def load_episode_summary_series(
    training_xlsx_path: Path,
) -> Tuple[Dict[str, List[Tuple[int, float]]], Path | None]:
    if not training_xlsx_path.exists():
        return {}, None
    if pd is None:
        print("pandas is not available; episode-level training curves from Excel will be skipped.")
        return {}, training_xlsx_path

    try:
        episode_df = pd.read_excel(training_xlsx_path, sheet_name="episode_summary")
    except Exception as exc:
        print(f"Failed to read episode_summary from {training_xlsx_path}: {exc}")
        return {}, training_xlsx_path

    if "global_step_end" not in episode_df.columns:
        print(f"Sheet 'episode_summary' in {training_xlsx_path} is missing 'global_step_end'.")
        return {}, training_xlsx_path

    step_values = pd.to_numeric(episode_df["global_step_end"], errors="coerce")
    series: Dict[str, List[Tuple[int, float]]] = {}
    for key in EPISODE_SUMMARY_KEYS:
        if key not in episode_df.columns:
            continue
        metric_values = pd.to_numeric(episode_df[key], errors="coerce")
        valid_mask = step_values.notna() & metric_values.notna()
        if not bool(valid_mask.any()):
            continue
        steps = step_values[valid_mask].to_numpy(dtype=np.int64)
        values = metric_values[valid_mask].to_numpy(dtype=np.float64)
        series[key] = dedupe_series(list(zip(steps.tolist(), values.tolist())))
    return series, training_xlsx_path


def merge_scalar_series(
    base_series: Dict[str, List[Tuple[int, float]]],
    preferred_series: Dict[str, List[Tuple[int, float]]],
) -> Dict[str, List[Tuple[int, float]]]:
    merged: Dict[str, List[Tuple[int, float]]] = {
        key: dedupe_series(values) for key, values in base_series.items()
    }
    for key, values in preferred_series.items():
        if values:
            merged[key] = dedupe_series(values)
    return merged


def load_eval_npz(eval_npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not eval_npz_path.exists():
        return None
    data = np.load(eval_npz_path)
    if "timesteps" not in data or "results" not in data:
        return None
    timesteps = np.asarray(data["timesteps"], dtype=np.int64)
    results = np.asarray(data["results"], dtype=np.float64)
    return timesteps, results.mean(axis=1), results.std(axis=1)


def load_config_row(config_csv_path: Path) -> dict[str, object]:
    if not config_csv_path.exists():
        return {}

    if pd is not None:
        try:
            df = pd.read_csv(config_csv_path)
        except Exception as exc:
            print(f"Failed to read train_config.csv from {config_csv_path}: {exc}")
            return {}
        if df.empty:
            return {}
        return df.iloc[0].to_dict()

    try:
        with config_csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            row = next(reader, None)
            return {} if row is None else dict(row)
    except Exception as exc:
        print(f"Failed to read train_config.csv from {config_csv_path}: {exc}")
        return {}


def series_to_xy(series: List[Tuple[int, float]]) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.array([step for step, _ in series], dtype=np.int64),
        np.array([value for _, value in series], dtype=np.float64),
    )


def draw_empty_panel(ax: plt.Axes, title: str, ylabel: str, message: str = "No data") -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Timesteps (x1e4)")
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#666666")
    ax.grid(True, alpha=0.15)


def plot_smoothed_series(
    ax: plt.Axes,
    series: List[Tuple[int, float]],
    *,
    title: str,
    ylabel: str,
    smooth_window: int,
    yscale: str | None = None,
    raw_color: str = "#C9D4E3",
    smooth_color: str = "#1F77B4",
    annotate_best: bool = False,
) -> bool:
    if not series:
        draw_empty_panel(ax, title, ylabel)
        return False

    x, y = series_to_xy(series)
    x_scaled = scaled_steps(x)
    ax.plot(x_scaled, y, color=raw_color, linewidth=1.0, alpha=0.85, label="raw")
    if smooth_window > 1 and len(y) >= smooth_window:
        y_smooth = moving_average(y, smooth_window)
        ax.plot(
            x_scaled,
            y_smooth,
            color=smooth_color,
            linewidth=2.0,
            label=f"moving avg ({smooth_window})",
        )
    if annotate_best and len(y) > 0:
        best_idx = int(np.argmax(y))
        ax.scatter(
            [x_scaled[best_idx]],
            [y[best_idx]],
            color="#D62728",
            s=42,
            zorder=5,
            label=f"best @ {int(x[best_idx])}",
        )
    ax.set_title(title)
    ax.set_xlabel("Timesteps (x1e4)")
    ax.set_ylabel(ylabel)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    return True


def plot_eval_panel(
    ax: plt.Axes,
    scalar_series: Dict[str, List[Tuple[int, float]]],
    eval_data: Tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> None:
    title = "Validation reward"
    ylabel = "Mean reward"

    if eval_data is not None:
        x, mean_reward, std_reward = eval_data
        x_scaled = scaled_steps(x)
        best_idx = int(np.argmax(mean_reward))
        ax.plot(x_scaled, mean_reward, color="#1F77B4", marker="o", linewidth=1.6, label="mean reward")
        ax.fill_between(
            x_scaled,
            mean_reward - std_reward,
            mean_reward + std_reward,
            color="#1F77B4",
            alpha=0.14,
            label="mean +/- std",
        )
        ax.scatter(
            [x_scaled[best_idx]],
            [mean_reward[best_idx]],
            color="#D62728",
            s=42,
            zorder=5,
            label=f"best @ {int(x[best_idx])}",
        )
        ax.set_title(title)
        ax.set_xlabel("Timesteps (x1e4)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        return

    plot_smoothed_series(
        ax,
        scalar_series.get("eval_reward", []),
        title=title,
        ylabel=ylabel,
        smooth_window=3,
        annotate_best=True,
    )


def plot_training_reward_panel(
    ax: plt.Axes,
    scalar_series: Dict[str, List[Tuple[int, float]]],
) -> str:
    preferred_series = scalar_series.get("episode_reward_scaled", [])
    if preferred_series:
        plot_smoothed_series(
            ax,
            preferred_series,
            title="Training reward (per episode)",
            ylabel="Episode reward (scaled)",
            smooth_window=TRAIN_REWARD_SMOOTH_WINDOW,
            raw_color="#D9E3F0",
            smooth_color="#1F77B4",
        )
        return "episode_reward_scaled"

    fallback_series = scalar_series.get("sb3_reward_mean", [])
    plot_smoothed_series(
        ax,
        fallback_series,
        title="Training reward (SB3 mean fallback)",
        ylabel="Episode reward mean",
        smooth_window=FALLBACK_REWARD_SMOOTH_WINDOW,
        raw_color="#D9E3F0",
        smooth_color="#1F77B4",
    )
    if fallback_series:
        ax.text(
            0.02,
            0.98,
            "Fallback: rollout/ep_rew_mean",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#555555",
        )
        return "sb3_reward_mean"
    return "missing"


def plot_multi_series_panel(
    ax: plt.Axes,
    scalar_series: Dict[str, List[Tuple[int, float]]],
    keys: Iterable[Tuple[str, str, str]],
    *,
    title: str,
    ylabel: str,
    smooth_window: int,
    yscale: str | None = None,
) -> None:
    plotted = False
    for key, label, color in keys:
        series = scalar_series.get(key, [])
        if not series:
            continue
        x, y = series_to_xy(series)
        if smooth_window > 1 and len(y) >= smooth_window:
            y = moving_average(y, smooth_window)
        ax.plot(scaled_steps(x), y, linewidth=1.8, color=color, label=label)
        plotted = True

    if not plotted:
        draw_empty_panel(ax, title, ylabel)
        return

    ax.set_title(title)
    ax.set_xlabel("Timesteps (x1e4)")
    ax.set_ylabel(ylabel)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


def format_config_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.6g}"
    return str(value)


def build_run_info_lines(
    config_row: dict[str, object],
    eval_data: Tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    scalar_series: Dict[str, List[Tuple[int, float]]],
) -> list[str]:
    lines: list[str] = ["SB3 direct PPO run"]
    for key in (
        "TOTAL_TIMESTEPS",
        "LEARNING_RATE",
        "N_STEPS",
        "BATCH_SIZE",
        "N_EPOCHS",
        "CLIP_RANGE",
        "GAMMA",
        "GAE_LAMBDA",
        "training_duration_hms",
        "device",
        "obs_dim",
        "action_dim",
    ):
        if key in config_row:
            lines.append(f"{key}: {format_config_value(config_row.get(key))}")

    if eval_data is not None and len(eval_data[1]) > 0:
        best_idx = int(np.argmax(eval_data[1]))
        lines.append(f"best_eval_step: {int(eval_data[0][best_idx])}")
        lines.append(f"best_eval_mean_reward: {float(eval_data[1][best_idx]):.6f}")
        lines.append(f"best_eval_std: {float(eval_data[2][best_idx]):.6f}")
    elif scalar_series.get("eval_reward"):
        x, y = series_to_xy(scalar_series["eval_reward"])
        best_idx = int(np.argmax(y))
        lines.append(f"best_eval_step: {int(x[best_idx])}")
        lines.append(f"best_eval_mean_reward: {float(y[best_idx]):.6f}")

    return lines


def plot_training_summary(
    scalar_series: Dict[str, List[Tuple[int, float]]],
    eval_data: Tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    save_path: Path,
) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))

    training_reward_source = plot_training_reward_panel(axes[0, 0], scalar_series)
    plot_eval_panel(axes[0, 1], scalar_series, eval_data)
    plot_smoothed_series(
        axes[1, 0],
        scalar_series.get("episode_system_cost", []),
        title="Episode system cost",
        ylabel="Cost",
        smooth_window=EPISODE_SMOOTH_WINDOW,
        smooth_color="#2CA02C",
    )
    plot_smoothed_series(
        axes[1, 1],
        scalar_series.get("episode_penalty_cost", []),
        title="Episode penalty cost",
        ylabel="Penalty",
        smooth_window=EPISODE_SMOOTH_WINDOW,
        yscale="symlog",
        smooth_color="#D62728",
    )

    fig.suptitle("SB3 direct PPO training summary", fontsize=15)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return training_reward_source


def plot_training_diagnostics(
    scalar_series: Dict[str, List[Tuple[int, float]]],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 13.0))

    plot_smoothed_series(
        axes[0, 0],
        scalar_series.get("policy_gradient_loss", []),
        title="Policy gradient loss",
        ylabel="Loss",
        smooth_window=LOSS_SMOOTH_WINDOW,
        yscale="symlog",
        smooth_color="#FF7F0E",
    )
    plot_smoothed_series(
        axes[0, 1],
        scalar_series.get("value_loss", []),
        title="Value loss",
        ylabel="Loss",
        smooth_window=LOSS_SMOOTH_WINDOW,
        yscale="symlog",
        smooth_color="#9467BD",
    )
    plot_multi_series_panel(
        axes[1, 0],
        scalar_series,
        PPO_CLIPPING_KEYS,
        title="PPO clipping diagnostics",
        ylabel="Value",
        smooth_window=DIAG_SMOOTH_WINDOW,
    )
    plot_multi_series_panel(
        axes[1, 1],
        scalar_series,
        PPO_OPTIMIZATION_KEYS,
        title="PPO optimization diagnostics",
        ylabel="Value",
        smooth_window=DIAG_SMOOTH_WINDOW,
    )
    plot_multi_series_panel(
        axes[2, 0],
        scalar_series,
        PENALTY_PANEL_KEYS,
        title="Key penalty terms",
        ylabel="Penalty",
        smooth_window=EPISODE_SMOOTH_WINDOW,
        yscale="symlog",
    )
    plot_multi_series_panel(
        axes[2, 1],
        scalar_series,
        EV_DIAGNOSTIC_KEYS,
        title="EV behavior diagnostics",
        ylabel="Energy (kWh)",
        smooth_window=EPISODE_SMOOTH_WINDOW,
    )

    fig.suptitle("SB3 direct PPO training diagnostics", fontsize=15)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.975])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot compact PPO training figures aligned with train_PPO_sb3_direct.py outputs."
    )
    parser.add_argument("--tb-log-dir", type=Path, default=DEFAULT_TB_LOG_DIR)
    parser.add_argument("--eval-npz", type=Path, default=DEFAULT_EVAL_NPZ_PATH)
    parser.add_argument("--training-xlsx", type=Path, default=DEFAULT_TRAINING_XLSX_PATH)
    parser.add_argument("--config-csv", type=Path, default=DEFAULT_CONFIG_CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tb_scalar_series, run_dir = load_scalar_series(args.tb_log_dir)
    episode_scalar_series, training_xlsx_path = load_episode_summary_series(args.training_xlsx)
    scalar_series = merge_scalar_series(tb_scalar_series, episode_scalar_series)
    eval_data = load_eval_npz(args.eval_npz)
    config_row = load_config_row(args.config_csv)

    summary_path = args.output_dir / "training_summary.png"
    diagnostics_path = args.output_dir / "training_diagnostics.png"

    training_reward_source = plot_training_summary(scalar_series, eval_data, summary_path)
    plot_training_diagnostics(scalar_series, diagnostics_path)

    print(f"SB3 direct PPO compact training plots saved to: {args.output_dir}")
    print(f"  - {summary_path.name}")
    print(f"  - {diagnostics_path.name}")
    if run_dir is not None:
        print(f"TensorBoard event scope: {run_dir}")
    if training_xlsx_path is not None:
        print(f"Episode summary source: {training_xlsx_path}")
    if args.config_csv.exists():
        print(f"Config source: {args.config_csv}")
    if training_reward_source == "episode_reward_scaled":
        print(
            "Training reward source: per-episode episode_reward_scaled "
            f"(moving average window={TRAIN_REWARD_SMOOTH_WINDOW})"
        )
    elif training_reward_source == "sb3_reward_mean":
        print(
            "Training reward source: fallback rollout/ep_rew_mean "
            f"(moving average window={FALLBACK_REWARD_SMOOTH_WINDOW})"
        )
    else:
        print("Training reward source: unavailable")
    if eval_data is not None and len(eval_data[1]) > 0:
        best_idx = int(np.argmax(eval_data[1]))
        print(
            "Best validation point: "
            f"step={int(eval_data[0][best_idx])}, "
            f"mean_reward={float(eval_data[1][best_idx]):.6f}, "
            f"std={float(eval_data[2][best_idx]):.6f}"
        )
    if config_row:
        print("Run info:")
        for line in build_run_info_lines(config_row, eval_data, scalar_series):
            print(f"  {line}")


if __name__ == "__main__":
    main()
