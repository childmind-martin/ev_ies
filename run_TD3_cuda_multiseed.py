from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_SEEDS = [42, 2024, 2025]
DEFAULT_TOTAL_TIMESTEPS = 96_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated CUDA TD3 training and deterministic testing for multiple seeds."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip completed training/testing stages. This is the default.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Rerun stages even if expected outputs already exist.",
    )
    return parser.parse_args()


def validate_cuda_device(device: str) -> str:
    cleaned = str(device).strip()
    lowered = cleaned.lower()
    if lowered != "cuda" and not lowered.startswith("cuda:"):
        raise SystemExit(
            "This CUDA multiseed runner requires --device cuda or --device cuda:<index>."
        )
    return cleaned


def run_suffix_for_seed(seed: int) -> str:
    return f"cuda_seed_{int(seed)}"


def paths_for_seed(seed: int) -> dict[str, Path]:
    suffix = run_suffix_for_seed(seed)
    return {
        "best_model": Path(f"models/td3_yearly_single_{suffix}/best/best_model.zip"),
        "training_summary": Path(f"results/td3_yearly_training_{suffix}/episode_summary.csv"),
        "training_runtime": Path(f"results/td3_yearly_training_{suffix}/training_runtime_summary.json"),
        "test_summary": Path(f"results/td3_yearly_test_{suffix}/daily_summary.csv"),
        "test_timeseries": Path(f"results/td3_yearly_test_{suffix}/timeseries_detail.csv"),
        "test_runtime": Path(f"results/td3_yearly_test_{suffix}/runtime_summary.json"),
        "test_export": Path(f"results/td3_yearly_test_{suffix}/td3_test_export.xlsx"),
    }


def training_complete(paths: dict[str, Path]) -> bool:
    return (
        paths["best_model"].exists()
        and paths["training_summary"].exists()
        and paths["training_runtime"].exists()
    )


def testing_complete(paths: dict[str, Path]) -> bool:
    return (
        paths["test_summary"].exists()
        and paths["test_timeseries"].exists()
        and paths["test_runtime"].exists()
        and paths["test_export"].exists()
    )


def run_command(command: list[str]) -> None:
    print(f"[run] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    seeds = [int(seed) for seed in args.seeds]
    device = validate_cuda_device(args.device)

    print(f"[td3-cuda-multiseed] seeds = {seeds}", flush=True)
    print(f"[td3-cuda-multiseed] device = {device}", flush=True)
    print(f"[td3-cuda-multiseed] total_timesteps = {args.total_timesteps}", flush=True)
    print(f"[td3-cuda-multiseed] skip_existing = {args.skip_existing}", flush=True)

    for seed in seeds:
        paths = paths_for_seed(seed)

        if args.skip_existing and training_complete(paths):
            print(f"[skip] seed={seed} training exists: {paths['best_model']}", flush=True)
        else:
            run_command(
                [
                    sys.executable,
                    "train_TD3_cuda_multiseed.py",
                    "--seed",
                    str(seed),
                    "--device",
                    device,
                    "--total-timesteps",
                    str(args.total_timesteps),
                ]
            )

        if args.skip_existing and testing_complete(paths):
            print(f"[skip] seed={seed} testing exists: {paths['test_summary']}", flush=True)
        else:
            run_command(
                [
                    sys.executable,
                    "test_TD3_cuda_multiseed.py",
                    "--seed",
                    str(seed),
                    "--device",
                    device,
                    "--total-timesteps",
                    str(args.total_timesteps),
                ]
            )

    print("[td3-cuda-multiseed] requested stages finished.", flush=True)


if __name__ == "__main__":
    main()
