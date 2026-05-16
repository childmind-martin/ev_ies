from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable


DEFAULT_TOTAL_TIMESTEPS = 96_000
TARGET_POLICY_NOISE = 0.20
TARGET_NOISE_CLIP = 0.50
ACTION_NOISE_TYPE = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test one isolated CUDA TD3 seed deterministically."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed to test.")
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
        help="Training timesteps recorded in runtime_summary.json.",
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


def run_name_for_seed(seed: int) -> str:
    return f"td3_cuda_seed_{int(seed)}"


def run_suffix_for_seed(seed: int) -> str:
    return f"cuda_seed_{int(seed)}"


def paths_for_seed(seed: int) -> dict[str, Path]:
    suffix = run_suffix_for_seed(seed)
    return {
        "model_path": Path(f"models/td3_yearly_single_{suffix}/best/best_model.zip"),
        "result_dir": Path(f"results/td3_yearly_test_{suffix}"),
        "runtime_summary": Path(f"results/td3_yearly_test_{suffix}/runtime_summary.json"),
    }


def patch_cuda_run_suffix(test_module: Any) -> Callable[[str | None, int], str]:
    original = test_module.make_run_suffix

    def make_run_suffix(run_name: str | None, seed: int) -> str:
        if run_name is not None:
            cleaned = "".join(
                ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_name.strip()
            )
            if cleaned == run_name_for_seed(seed):
                return run_suffix_for_seed(seed)
        return original(run_name, seed)

    test_module.make_run_suffix = make_run_suffix
    return original


def update_test_runtime_summary(*, seed: int, device: str, total_timesteps: int) -> None:
    paths = paths_for_seed(seed)
    runtime_path = paths["runtime_summary"]
    data: dict[str, Any] = {}
    if runtime_path.exists():
        data = json.loads(runtime_path.read_text(encoding="utf-8"))

    data.update(
        {
            "method": "TD3-CUDA",
            "seed": int(seed),
            "device": str(device),
            "run_name": run_name_for_seed(seed),
            "model_path": str(paths["model_path"]),
            "result_dir": str(paths["result_dir"]),
            "total_timesteps": int(total_timesteps),
            "deterministic": True,
            "action_noise_type": ACTION_NOISE_TYPE,
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
    run_name = run_name_for_seed(seed)

    import test_TD3 as base_test

    original_make_run_suffix = patch_cuda_run_suffix(base_test)
    old_argv = sys.argv[:]
    sys.argv = [
        "test_TD3.py",
        "--seed",
        str(seed),
        "--run-name",
        run_name,
        "--device",
        device,
    ]
    try:
        base_test.main()
    finally:
        sys.argv = old_argv
        base_test.make_run_suffix = original_make_run_suffix

    update_test_runtime_summary(
        seed=seed,
        device=device,
        total_timesteps=int(args.total_timesteps),
    )


if __name__ == "__main__":
    main()
