from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable


DEFAULT_TOTAL_TIMESTEPS = 96_000
EPISODE_LENGTH = 24
TARGET_POLICY_NOISE = 0.20
TARGET_NOISE_CLIP = 0.50
ACTION_NOISE_TYPE = None
ACTION_NOISE_SIGMA = 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one isolated CUDA TD3 seed using the original TD3 hyperparameters."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed to train.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="CUDA device passed to Stable-Baselines3. Use cuda or cuda:0.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help="Training timesteps. Defaults to 96000 for parity with the original TD3 experiment.",
    )
    return parser.parse_args(argv)


def validate_cuda_device(device: str) -> str:
    cleaned = str(device).strip()
    lowered = cleaned.lower()
    if lowered != "cuda" and not lowered.startswith("cuda:"):
        raise SystemExit(
            "This CUDA multiseed entrypoint requires --device cuda or --device cuda:<index>."
        )
    return cleaned


def total_timesteps_to_episodes(total_timesteps: int) -> int:
    total_timesteps = int(total_timesteps)
    if total_timesteps <= 0:
        raise SystemExit("--total-timesteps must be positive.")
    if total_timesteps % EPISODE_LENGTH != 0:
        raise SystemExit(
            f"--total-timesteps must be divisible by {EPISODE_LENGTH}; got {total_timesteps}."
        )
    return total_timesteps // EPISODE_LENGTH


def run_name_for_seed(seed: int) -> str:
    return f"td3_cuda_seed_{int(seed)}"


def run_suffix_for_seed(seed: int) -> str:
    return f"cuda_seed_{int(seed)}"


def paths_for_seed(seed: int) -> dict[str, Path]:
    suffix = run_suffix_for_seed(seed)
    return {
        "model_dir": Path(f"models/td3_yearly_single_{suffix}"),
        "best_model_path": Path(f"models/td3_yearly_single_{suffix}/best/best_model.zip"),
        "log_dir": Path(f"logs/td3_yearly_single_{suffix}"),
        "tb_dir": Path(f"tb/td3_yearly_single_{suffix}"),
        "result_dir": Path(f"results/td3_yearly_training_{suffix}"),
        "runtime_summary": Path(f"results/td3_yearly_training_{suffix}/training_runtime_summary.json"),
    }


def patch_cuda_run_suffix(train_module: Any) -> Callable[[str | None, int], str]:
    original = train_module.make_run_suffix

    def make_run_suffix(run_name: str | None, seed: int) -> str:
        if run_name is not None:
            cleaned = "".join(
                ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_name.strip()
            )
            if cleaned == run_name_for_seed(seed):
                return run_suffix_for_seed(seed)
        return original(run_name, seed)

    train_module.make_run_suffix = make_run_suffix
    return original


def update_training_runtime_summary(
    *,
    seed: int,
    device: str,
    total_timesteps: int,
    total_episodes: int,
) -> None:
    paths = paths_for_seed(seed)
    runtime_path = paths["runtime_summary"]
    data: dict[str, Any] = {}
    if runtime_path.exists():
        data = json.loads(runtime_path.read_text(encoding="utf-8"))

    data.update(
        {
            "method": "TD3-CUDA",
            "run_name": run_name_for_seed(seed),
            "seed": int(seed),
            "device": str(device),
            "model_dir": str(paths["model_dir"]),
            "best_model_path": str(paths["best_model_path"]),
            "log_dir": str(paths["log_dir"]),
            "result_dir": str(paths["result_dir"]),
            "total_episodes": int(total_episodes),
            "total_timesteps": int(total_timesteps),
            "action_noise_type": ACTION_NOISE_TYPE,
            "action_noise_sigma": ACTION_NOISE_SIGMA,
            "target_policy_noise": TARGET_POLICY_NOISE,
            "target_noise_clip": TARGET_NOISE_CLIP,
        }
    )
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed = int(args.seed)
    device = validate_cuda_device(args.device)
    total_timesteps = int(args.total_timesteps)
    total_episodes = total_timesteps_to_episodes(total_timesteps)
    run_name = run_name_for_seed(seed)

    import train_TD3 as base_train

    original_make_run_suffix = patch_cuda_run_suffix(base_train)
    old_argv = sys.argv[:]
    sys.argv = [
        "train_TD3.py",
        "--seed",
        str(seed),
        "--run-name",
        run_name,
        "--device",
        device,
        "--total-episodes",
        str(total_episodes),
    ]
    try:
        base_train.main()
    finally:
        sys.argv = old_argv
        base_train.make_run_suffix = original_make_run_suffix

    update_training_runtime_summary(
        seed=seed,
        device=device,
        total_timesteps=total_timesteps,
        total_episodes=total_episodes,
    )


if __name__ == "__main__":
    main()
