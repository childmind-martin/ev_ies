from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

try:
    import gymnasium as gym
except ModuleNotFoundError:
    gym = None


EV_SCENARIOS = ("uncontrolled_charging", "ordered_charging", "v2g")
NO_V2G_SCENARIOS = ("uncontrolled_charging", "ordered_charging")

SCENARIO_DESCRIPTIONS = {
    "uncontrolled_charging": (
        "A uncontrolled_charging: EVs charge at the maximum feasible rate after arrival; "
        "EV discharge and V2G buffer charging are disabled."
    ),
    "ordered_charging": (
        "B ordered_charging: TD3 controls EV charging; EV discharge and V2G buffer "
        "charging are disabled."
    ),
    "v2g": (
        "C v2g: TD3 controls EV charging and discharging; the V2G buffer is retained."
    ),
}


def normalize_ev_scenario(ev_scenario: str) -> str:
    scenario = str(ev_scenario).strip().lower()
    if scenario not in EV_SCENARIOS:
        raise ValueError(
            f"Unknown EV scenario {ev_scenario!r}. "
            f"Valid scenarios: {', '.join(EV_SCENARIOS)}"
        )
    return scenario


def is_no_v2g_scenario(ev_scenario: str) -> bool:
    return normalize_ev_scenario(ev_scenario) in NO_V2G_SCENARIOS


def _copy_ev_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return deepcopy(value)


def _is_v2g_or_discharge_capability_field(field_name: str) -> bool:
    name = str(field_name).strip().lower()
    if name in {"v2g_flag", "v2g", "is_v2g", "can_v2g", "v2g_capable"}:
        return True
    if "v2g" in name:
        return True

    efficiency_tokens = ("eta", "eff", "efficiency")
    if any(token in name for token in efficiency_tokens):
        return False

    discharge_tokens = (
        "p_dis",
        "dis_max",
        "max_dis",
        "discharge_power",
        "max_discharge",
        "discharge_cap",
        "export_power",
    )
    return any(token in name for token in discharge_tokens)


def _zero_like(value: Any, *, bool_field: bool = False) -> Any:
    if isinstance(value, np.ndarray):
        if bool_field:
            return np.zeros_like(value, dtype=np.bool_)
        return np.zeros_like(value)
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, float, np.number)):
        return type(value)(0)
    return 0


def wrap_ev_data(ev_data: Mapping[str, Any], ev_scenario: str) -> dict[str, Any]:
    """Return scenario-specific EV input data without mutating the source mapping."""

    scenario = normalize_ev_scenario(ev_scenario)
    wrapped = {key: _copy_ev_value(value) for key, value in ev_data.items()}
    if scenario == "v2g":
        return wrapped

    for key, value in list(wrapped.items()):
        if _is_v2g_or_discharge_capability_field(key):
            wrapped[key] = _zero_like(value, bool_field=(str(key).lower() == "v2g_flag"))

    if "v2g_flag" in wrapped:
        wrapped["v2g_flag"] = _zero_like(wrapped["v2g_flag"], bool_field=True)
    if "p_dis_max_kw" in wrapped:
        wrapped["p_dis_max_kw"] = _zero_like(wrapped["p_dis_max_kw"])
    return wrapped


def wrap_ev_provider(
    ev_provider: Callable[[Any], Mapping[str, Any]],
    ev_scenario: str,
) -> Callable[[Any], dict[str, Any]]:
    scenario = normalize_ev_scenario(ev_scenario)

    def scenario_ev_provider(case: Any) -> dict[str, Any]:
        return wrap_ev_data(ev_provider(case), scenario)

    return scenario_ev_provider


def wrap_config(cfg: Any, ev_scenario: str, *, copy_config: bool = True) -> Any:
    """Return a config object with only scenario-level EV settings adjusted."""

    scenario = normalize_ev_scenario(ev_scenario)
    wrapped = deepcopy(cfg) if copy_config else cfg
    if scenario in NO_V2G_SCENARIOS:
        setattr(wrapped, "ev_v2g_buffer_soc_margin", 0.0)
    return wrapped


def wrap_action(action: Any, ev_scenario: str) -> np.ndarray:
    """Apply the scenario action policy.

    action[0] = EES, action[1] = GT, action[2] = EV charge,
    action[3] = EV discharge.
    """

    scenario = normalize_ev_scenario(ev_scenario)
    wrapped = np.asarray(action, dtype=np.float32).copy()
    if wrapped.shape[0] < 4:
        raise ValueError(f"TD3 action must have at least 4 entries, got shape={wrapped.shape}")

    if scenario == "uncontrolled_charging":
        wrapped[2] = 1.0
        wrapped[3] = 0.0
    elif scenario == "ordered_charging":
        wrapped[3] = 0.0
    return wrapped


if gym is not None:

    class EVScenarioActionWrapper(gym.ActionWrapper):
        def __init__(self, env: gym.Env, ev_scenario: str):
            super().__init__(env)
            self.ev_scenario = normalize_ev_scenario(ev_scenario)

        def action(self, action: Any) -> np.ndarray:
            return wrap_action(action, self.ev_scenario)

else:

    class EVScenarioActionWrapper:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "gymnasium is required when EVScenarioActionWrapper is used."
            )


@dataclass
class EVScenarioProvider:
    base_provider: Callable[[Any], Mapping[str, Any]]
    ev_scenario: str

    def __post_init__(self) -> None:
        self.ev_scenario = normalize_ev_scenario(self.ev_scenario)

    def __call__(self, case: Any) -> dict[str, Any]:
        return wrap_ev_data(self.base_provider(case), self.ev_scenario)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_provider, name)


def scenario_run_name(ev_scenario: str, seed: int) -> str:
    scenario = normalize_ev_scenario(ev_scenario)
    return f"{scenario}_seed_{int(seed)}"


def scenario_test_result_dir(ev_scenario: str, seed: int) -> str:
    scenario = normalize_ev_scenario(ev_scenario)
    if scenario == "v2g":
        return f"results/td3_yearly_test_seed_{int(seed)}"
    return f"results/td3_yearly_test_{scenario}_seed_{int(seed)}"


def scenario_training_result_dir(ev_scenario: str, seed: int) -> str:
    scenario = normalize_ev_scenario(ev_scenario)
    if scenario == "v2g":
        return f"results/td3_yearly_training_seed_{int(seed)}"
    return f"results/td3_yearly_training_{scenario}_seed_{int(seed)}"


def scenario_model_dir(ev_scenario: str, seed: int) -> str:
    scenario = normalize_ev_scenario(ev_scenario)
    if scenario == "v2g":
        return f"models/td3_yearly_single_seed_{int(seed)}"
    return f"models/td3_yearly_single_{scenario}_seed_{int(seed)}"
