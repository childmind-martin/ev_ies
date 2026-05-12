from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import test_TD3 as base
from ev_scenario_wrappers import (
    EVScenarioProvider,
    EV_SCENARIOS,
    SCENARIO_DESCRIPTIONS,
    normalize_ev_scenario,
    scenario_run_name,
    wrap_action,
    wrap_config,
)


EXTRA_CONFIG_COLUMNS = (
    "ev_scenario",
    "ev_scenario_description",
    "ev_scenario_ev_data_policy",
    "ev_scenario_action_policy",
    "ev_v2g_buffer_soc_margin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test TD3 under an EV scenario without changing the main V2G program."
    )
    parser.add_argument("--ev-scenario", choices=EV_SCENARIOS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=base.REQUESTED_DEVICE)
    return parser.parse_args()


def _scenario_ev_data_policy(ev_scenario: str) -> str:
    if ev_scenario == "v2g":
        return "EV data is unchanged."
    return "v2g_flag and EV discharge capability fields are set to 0."


def _scenario_action_policy(ev_scenario: str) -> str:
    if ev_scenario == "uncontrolled_charging":
        return "action[2] = 1.0 and action[3] = 0.0."
    if ev_scenario == "ordered_charging":
        return "TD3 action[2] is retained and action[3] = 0.0."
    return "TD3 raw action is retained."


def _preferred_v2g_run_name(seed: int) -> str | None:
    seed_model_path = Path(f"models/td3_yearly_single_seed_{int(seed)}/best/best_model.zip")
    if seed_model_path.exists():
        return f"seed_{int(seed)}"
    if int(seed) == int(base.DEFAULT_SEED):
        return None
    return f"seed_{int(seed)}"


def _extend_config_export(module: Any, ev_scenario: str) -> None:
    existing_columns = tuple(module.CONFIG_COLUMNS)
    module.CONFIG_COLUMNS = existing_columns + tuple(
        column for column in EXTRA_CONFIG_COLUMNS if column not in existing_columns
    )

    original_build_config_row = module.build_config_row

    def build_config_row(cfg: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        row = original_build_config_row(cfg, *args, **kwargs)
        row.update(
            {
                "ev_scenario": ev_scenario,
                "ev_scenario_description": SCENARIO_DESCRIPTIONS[ev_scenario],
                "ev_scenario_ev_data_policy": _scenario_ev_data_policy(ev_scenario),
                "ev_scenario_action_policy": _scenario_action_policy(ev_scenario),
                "ev_v2g_buffer_soc_margin": getattr(
                    cfg, "ev_v2g_buffer_soc_margin", None
                ),
            }
        )
        return row

    module.build_config_row = build_config_row


def _extend_runtime_summary(module: Any, ev_scenario: str) -> None:
    original_write_runtime = module.write_runtime_summary

    def write_runtime_summary(path: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
        row = original_write_runtime(path, *args, **kwargs)
        row.update(
            {
                "ev_scenario": ev_scenario,
                "ev_scenario_description": SCENARIO_DESCRIPTIONS[ev_scenario],
                "ev_scenario_ev_data_policy": _scenario_ev_data_policy(ev_scenario),
                "ev_scenario_action_policy": _scenario_action_policy(ev_scenario),
            }
        )
        Path(path).write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return row

    module.write_runtime_summary = write_runtime_summary


def _patch_runtime_modules(module: Any, ev_scenario: str) -> None:
    original_import_runtime_modules = module.import_runtime_modules

    def import_runtime_modules() -> dict[str, Any]:
        modules = original_import_runtime_modules()

        original_ies_config = modules["IESConfig"]
        original_ev_provider = modules["YearlyEVProvider"]

        def ScenarioIESConfig(*args: Any, **kwargs: Any) -> Any:
            return wrap_config(original_ies_config(*args, **kwargs), ev_scenario)

        class ScenarioYearlyEVProvider(EVScenarioProvider):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(original_ev_provider(*args, **kwargs), ev_scenario)

        modules["IESConfig"] = ScenarioIESConfig
        modules["YearlyEVProvider"] = ScenarioYearlyEVProvider
        return modules

    module.import_runtime_modules = import_runtime_modules


def _patch_rollout(module: Any, ev_scenario: str) -> None:
    def rollout_one_day(model: Any, env: Any) -> tuple[list[dict[str, Any]], float]:
        obs, _ = env.reset()
        done = False
        infos: list[dict[str, Any]] = []
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = wrap_action(action, ev_scenario)
            obs, reward, terminated, truncated, step_info = env.step(action)
            infos.append(step_info)
            total_reward += float(reward)
            done = bool(terminated or truncated)

        return infos, total_reward

    module.rollout_one_day = rollout_one_day


def main() -> None:
    args = parse_args()
    ev_scenario = normalize_ev_scenario(args.ev_scenario)
    seed = int(args.seed)

    if ev_scenario == "v2g":
        run_name = _preferred_v2g_run_name(seed)
    else:
        run_name = scenario_run_name(ev_scenario, seed)

    base_args = argparse.Namespace(seed=seed, run_name=run_name, device=str(args.device))

    _extend_config_export(base, ev_scenario)
    _extend_runtime_summary(base, ev_scenario)
    _patch_runtime_modules(base, ev_scenario)
    _patch_rollout(base, ev_scenario)
    base.parse_args = lambda argv=None: base_args
    base.main()


if __name__ == "__main__":
    main()
