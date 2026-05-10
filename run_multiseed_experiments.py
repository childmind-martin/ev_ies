from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_METHODS = ["TD3", "DDPG", "PPO"]
DEFAULT_SEEDS = [42, 2024, 2025]

METHOD_SPECS = {
    "TD3": {
        "train_script": "train_TD3.py",
        "test_script": "test_TD3.py",
        "best_model": "models/td3_yearly_single_{run_suffix}/best/best_model.zip",
        "training_summary": "results/td3_yearly_training_{run_suffix}/episode_summary.csv",
        "test_summary": "results/td3_yearly_test_{run_suffix}/daily_summary.csv",
    },
    "DDPG": {
        "train_script": "train_DDPG.py",
        "test_script": "test_DDPG.py",
        "best_model": "models/ddpg_yearly_single_{run_suffix}/best/best_model.zip",
        "training_summary": "results/ddpg_yearly_training_{run_suffix}/episode_summary.csv",
        "test_summary": "results/ddpg_yearly_test_{run_suffix}/daily_summary.csv",
    },
    "PPO": {
        "train_script": "train_PPO_sb3_direct.py",
        "test_script": "test_PPO_sb3_direct.py",
        "best_model": "models/ppo_sb3_direct_{run_suffix}/best/best_model.zip",
        "training_summary": "results/ppo_sb3_direct_training_{run_suffix}/episode_summary.csv",
        "test_summary": "results/ppo_sb3_direct_test_{run_suffix}/daily_summary.csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed TD3/DDPG/PPO training and testing.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, help="Methods to run: TD3 DDPG PPO.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Random seeds to run.")
    parser.add_argument("--device", type=str, default="auto", help="SB3/PyTorch device passed to train/test scripts: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--total-episodes", type=int, default=None, help="Override training episodes for quick benchmark runs.")
    parser.add_argument("--run-prefix", type=str, default=None, help="Optional output prefix, for example cuda to create cuda_seed_42 outputs.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip train/test stages with existing outputs.")
    return parser.parse_args()


def normalize_methods(methods: list[str]) -> list[str]:
    normalized = []
    for method in methods:
        key = method.upper()
        if key not in METHOD_SPECS:
            raise ValueError(f"Unknown method {method!r}. Valid methods: {', '.join(METHOD_SPECS)}")
        normalized.append(key)
    return normalized


def run_command(command: list[str]) -> None:
    print(f"[run] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def build_run_name(seed: int, run_prefix: str | None) -> str:
    if run_prefix is None or not run_prefix.strip():
        return f"seed_{seed}"
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_prefix.strip())
    return f"{cleaned}_seed_{seed}" if cleaned else f"seed_{seed}"


def main() -> None:
    args = parse_args()
    methods = normalize_methods(args.methods)
    seeds = [int(seed) for seed in args.seeds]

    print(f"[multiseed] methods = {methods}", flush=True)
    print(f"[multiseed] seeds = {seeds}", flush=True)
    print(f"[multiseed] device = {args.device}", flush=True)
    print(f"[multiseed] total_episodes = {args.total_episodes}", flush=True)
    print(f"[multiseed] run_prefix = {args.run_prefix}", flush=True)
    print(f"[multiseed] skip_existing = {args.skip_existing}", flush=True)

    for method in methods:
        spec = METHOD_SPECS[method]
        for seed in seeds:
            run_name = build_run_name(seed, args.run_prefix)
            best_model = Path(spec["best_model"].format(run_suffix=run_name))
            training_summary = Path(spec["training_summary"].format(run_suffix=run_name))
            test_summary = Path(spec["test_summary"].format(run_suffix=run_name))

            if args.skip_existing and best_model.exists() and training_summary.exists():
                print(f"[skip] {method} seed={seed} training exists: {best_model} and {training_summary}", flush=True)
            else:
                train_command = [sys.executable, spec["train_script"], "--seed", str(seed), "--run-name", run_name, "--device", args.device]
                if args.total_episodes is not None:
                    train_command.extend(["--total-episodes", str(args.total_episodes)])
                run_command(train_command)

            if args.skip_existing and test_summary.exists():
                print(f"[skip] {method} seed={seed} test exists: {test_summary}", flush=True)
            else:
                run_command([sys.executable, spec["test_script"], "--seed", str(seed), "--run-name", run_name, "--device", args.device])

    print("[multiseed] all requested runs finished.", flush=True)


if __name__ == "__main__":
    main()
