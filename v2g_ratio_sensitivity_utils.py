from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


V2G_FIELD_CANDIDATES = ("v2g_flag", "is_v2g", "v2g_available")
P_DIS_FIELD_CANDIDATES = ("p_dis_max_kw", "discharge_power_max", "p_v2g_max")
VEHICLE_TYPE_FIELD_CANDIDATES = ("vehicle_type", "type")
V2G_PRIORITY_TYPES = (
    "contracted_v2g_commuter",
    "light_service_ev",
    "employee_commuter",
)
DEFAULT_V2G_DISCHARGE_POWER_KW = 14.0

_WARNED_MESSAGES: set[str] = set()


@dataclass
class V2GRatioMetadata:
    ratio: float
    target_v2g_ev_count: int
    actual_v2g_ev_count: int
    total_ev_count: int
    actual_v2g_ratio: float
    v2g_selection_rule: str
    v2g_field_name: str | None
    p_dis_field_name: str | None
    vehicle_type_field_name: str | None
    warnings: list[str] = field(default_factory=list)
    default_discharge_power_kw: float = DEFAULT_V2G_DISCHARGE_POWER_KW
    selected_default_power_count: int = 0

    def to_runtime_dict(self) -> dict[str, Any]:
        return {
            "ratio": float(self.ratio),
            "target_v2g_ev_count": int(self.target_v2g_ev_count),
            "actual_v2g_ev_count": int(self.actual_v2g_ev_count),
            "total_ev_count": int(self.total_ev_count),
            "actual_v2g_ratio": float(self.actual_v2g_ratio),
            "v2g_selection_rule": self.v2g_selection_rule,
            "v2g_field_name": self.v2g_field_name,
            "p_dis_field_name": self.p_dis_field_name,
            "vehicle_type_field_name": self.vehicle_type_field_name,
            "warnings": list(self.warnings),
            "default_discharge_power_kw": float(self.default_discharge_power_kw),
            "selected_default_power_count": int(self.selected_default_power_count),
        }


def apply_v2g_participation_ratio(ev_data: Any, ratio: float, seed: int = 42) -> Any:
    """Return an EV-data copy with only V2G capability fields changed."""

    modified, _ = apply_v2g_participation_ratio_with_metadata(
        ev_data, ratio, seed=seed, emit_warnings=True
    )
    return modified


def apply_v2g_participation_ratio_with_metadata(
    ev_data: Any,
    ratio: float,
    seed: int = 42,
    *,
    emit_warnings: bool = True,
) -> tuple[Any, V2GRatioMetadata]:
    copied = copy_ev_data(ev_data)
    fields = _detect_fields(copied)
    total_ev = _infer_total_ev_count(copied, fields)
    target_count = _target_v2g_count(total_ev, ratio)

    warnings: list[str] = []
    if fields["v2g"] is None:
        warnings.append(
            "No V2G flag field found; created v2g_flag for the sensitivity copy."
        )
        _set_field(copied, "v2g_flag", np.zeros(total_ev, dtype=np.int32))
        fields["v2g"] = "v2g_flag"

    if fields["p_dis"] is None:
        warnings.append(
            "No EV discharge power field found; created p_dis_max_kw for the sensitivity copy."
        )
        _set_field(
            copied,
            "p_dis_max_kw",
            np.full(total_ev, DEFAULT_V2G_DISCHARGE_POWER_KW, dtype=np.float32),
        )
        fields["p_dis"] = "p_dis_max_kw"

    selected = _select_v2g_indices(copied, target_count, seed, fields)
    v2g_flag = np.zeros(total_ev, dtype=np.int32)
    if selected.size:
        v2g_flag[selected] = 1

    original_p_dis = _field_as_array(ev_data, fields["p_dis"], total_ev)
    if original_p_dis is None:
        original_p_dis = np.full(
            total_ev, DEFAULT_V2G_DISCHARGE_POWER_KW, dtype=np.float32
        )
    p_dis = np.zeros(total_ev, dtype=np.float32)
    selected_default_count = 0
    if selected.size:
        selected_power = np.asarray(original_p_dis[selected], dtype=np.float32).copy()
        invalid_power = ~np.isfinite(selected_power) | (selected_power <= 0.0)
        selected_default_count = int(invalid_power.sum())
        if selected_default_count:
            selected_power[invalid_power] = DEFAULT_V2G_DISCHARGE_POWER_KW
            warnings.append(
                f"{selected_default_count} selected EV discharge power values were missing/non-positive; "
                f"defaulted to {DEFAULT_V2G_DISCHARGE_POWER_KW:g} kW."
            )
        p_dis[selected] = selected_power

    _set_field(copied, fields["v2g"], v2g_flag)
    _set_field(copied, fields["p_dis"], p_dis)

    if target_count <= 0:
        selection_rule = "all_ev_v2g_disabled"
    else:
        selection_rule = (
            "vehicle_type_priority"
            if fields["vehicle_type"] is not None
            else "seeded_random_by_ev_index"
        )

    if target_count > 0 and fields["vehicle_type"] is None:
        warnings.append(
            "No vehicle_type/type field found; selected V2G vehicles by EV index with the fixed seed."
        )

    actual_count = int(v2g_flag.sum())
    metadata = V2GRatioMetadata(
        ratio=float(ratio),
        target_v2g_ev_count=int(target_count),
        actual_v2g_ev_count=actual_count,
        total_ev_count=int(total_ev),
        actual_v2g_ratio=float(actual_count / total_ev) if total_ev else 0.0,
        v2g_selection_rule=selection_rule,
        v2g_field_name=fields["v2g"],
        p_dis_field_name=fields["p_dis"],
        vehicle_type_field_name=fields["vehicle_type"],
        warnings=warnings,
        selected_default_power_count=selected_default_count,
    )
    if emit_warnings:
        for message in warnings:
            _warn_once(message)
    return copied, metadata


def summarize_v2g_participation(ev_data: Any) -> V2GRatioMetadata:
    fields = _detect_fields(ev_data)
    total_ev = _infer_total_ev_count(ev_data, fields)
    warnings: list[str] = []
    v2g_values = _field_as_array(ev_data, fields["v2g"], total_ev)
    if v2g_values is None:
        actual_count = 0
        warnings.append("No V2G flag field found while summarizing EV data.")
    else:
        actual_count = int(np.asarray(v2g_values).astype(bool).sum())

    if fields["p_dis"] is None:
        warnings.append("No EV discharge power field found while summarizing EV data.")

    return V2GRatioMetadata(
        ratio=float(actual_count / total_ev) if total_ev else 0.0,
        target_v2g_ev_count=actual_count,
        actual_v2g_ev_count=actual_count,
        total_ev_count=int(total_ev),
        actual_v2g_ratio=float(actual_count / total_ev) if total_ev else 0.0,
        v2g_selection_rule="existing_v2g_flags_base_case",
        v2g_field_name=fields["v2g"],
        p_dis_field_name=fields["p_dis"],
        vehicle_type_field_name=fields["vehicle_type"],
        warnings=warnings,
    )


def copy_ev_data(ev_data: Any) -> Any:
    if isinstance(ev_data, Mapping):
        return {key: _copy_value(value) for key, value in ev_data.items()}
    if _is_pandas_dataframe(ev_data):
        return ev_data.copy(deep=True)
    if isinstance(ev_data, np.ndarray):
        return np.array(ev_data, copy=True)
    return deepcopy(ev_data)


def ratio_dir_name(ratio: float) -> str:
    ratio = float(ratio)
    percent = ratio * 100.0
    if abs(percent - round(percent)) <= 1e-9:
        return f"ratio_{int(round(percent)):03d}"
    token = f"{percent:05.1f}".replace(".", "")
    return f"ratio_{token}"


def is_base_ratio(ratio: float) -> bool:
    return abs(float(ratio) - 0.279) <= 5e-4


def _copy_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return deepcopy(value)


def _detect_fields(ev_data: Any) -> dict[str, str | None]:
    names = _field_names(ev_data)
    return {
        "v2g": _first_existing_name(names, V2G_FIELD_CANDIDATES),
        "p_dis": _first_existing_name(names, P_DIS_FIELD_CANDIDATES),
        "vehicle_type": _first_existing_name(names, VEHICLE_TYPE_FIELD_CANDIDATES),
    }


def _field_names(ev_data: Any) -> list[str]:
    if isinstance(ev_data, Mapping):
        return [str(key) for key in ev_data.keys()]
    if _is_pandas_dataframe(ev_data):
        return [str(key) for key in ev_data.columns]
    if isinstance(ev_data, np.ndarray) and ev_data.dtype.names:
        return [str(key) for key in ev_data.dtype.names]
    return []


def _first_existing_name(names: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {name.lower(): name for name in names}
    for candidate in candidates:
        found = lookup.get(candidate.lower())
        if found is not None:
            return found
    return None


def _infer_total_ev_count(ev_data: Any, fields: Mapping[str, str | None]) -> int:
    for key in (fields.get("v2g"), fields.get("p_dis"), fields.get("vehicle_type")):
        values = _get_field(ev_data, key)
        if values is not None:
            return int(len(values))

    if isinstance(ev_data, Mapping):
        for values in ev_data.values():
            try:
                return int(len(values))
            except TypeError:
                continue
    if _is_pandas_dataframe(ev_data):
        return int(len(ev_data))
    if isinstance(ev_data, np.ndarray):
        if ev_data.ndim == 0:
            raise ValueError("Cannot infer EV count from scalar ndarray.")
        return int(ev_data.shape[0])
    raise ValueError("Cannot infer total EV count from ev_data.")


def _target_v2g_count(total_ev: int, ratio: float) -> int:
    if total_ev < 0:
        raise ValueError("total_ev must be non-negative.")
    ratio = float(ratio)
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"V2G participation ratio must be within [0, 1], got {ratio}.")
    return int(np.clip(round(total_ev * ratio), 0, total_ev))


def _select_v2g_indices(
    ev_data: Any,
    target_count: int,
    seed: int,
    fields: Mapping[str, str | None],
) -> np.ndarray:
    total_ev = _infer_total_ev_count(ev_data, fields)
    target_count = int(np.clip(target_count, 0, total_ev))
    if target_count <= 0:
        return np.array([], dtype=np.int64)

    type_field = fields.get("vehicle_type")
    type_values = _get_field(ev_data, type_field)
    if type_values is None:
        rng = np.random.default_rng(int(seed))
        return np.sort(rng.choice(total_ev, size=target_count, replace=False)).astype(
            np.int64
        )

    normalized_types = np.asarray(type_values).astype(str)
    selected: list[int] = []
    used = np.zeros(total_ev, dtype=bool)
    for vehicle_type in V2G_PRIORITY_TYPES:
        indices = np.flatnonzero(normalized_types == vehicle_type)
        for idx in indices:
            if len(selected) >= target_count:
                break
            selected.append(int(idx))
            used[int(idx)] = True
        if len(selected) >= target_count:
            break

    if len(selected) < target_count:
        for idx in range(total_ev):
            if used[idx]:
                continue
            selected.append(idx)
            if len(selected) >= target_count:
                break

    return np.asarray(selected, dtype=np.int64)


def _get_field(ev_data: Any, field_name: str | None) -> Any | None:
    if field_name is None:
        return None
    if isinstance(ev_data, Mapping):
        return ev_data.get(field_name)
    if _is_pandas_dataframe(ev_data):
        if field_name in ev_data.columns:
            return ev_data[field_name].to_numpy()
        return None
    if isinstance(ev_data, np.ndarray) and ev_data.dtype.names:
        if field_name in ev_data.dtype.names:
            return ev_data[field_name]
    return None


def _field_as_array(ev_data: Any, field_name: str | None, total_ev: int) -> np.ndarray | None:
    values = _get_field(ev_data, field_name)
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.shape[0] != total_ev:
        raise ValueError(
            f"EV field {field_name!r} length {arr.shape[0]} does not match total_ev={total_ev}."
        )
    return arr


def _set_field(ev_data: Any, field_name: str, values: np.ndarray) -> None:
    if isinstance(ev_data, Mapping):
        ev_data[field_name] = values
        return
    if _is_pandas_dataframe(ev_data):
        ev_data[field_name] = values
        return
    if isinstance(ev_data, np.ndarray) and ev_data.dtype.names:
        if field_name not in ev_data.dtype.names:
            raise KeyError(
                f"Cannot create new field {field_name!r} in a structured ndarray copy."
            )
        ev_data[field_name] = values
        return
    raise TypeError(
        "Unsupported ev_data type for V2G field updates. Use a mapping, DataFrame, or structured ndarray."
    )


def _is_pandas_dataframe(value: Any) -> bool:
    return value.__class__.__module__.startswith("pandas.") and value.__class__.__name__ == "DataFrame"


def _warn_once(message: str) -> None:
    if message in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(message)
    print(f"WARNING: {message}", file=sys.stderr)
