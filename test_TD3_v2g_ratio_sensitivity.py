from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

import test_TD3 as base
from v2g_ratio_sensitivity_utils import (
    V2GRatioMetadata,
    apply_v2g_participation_ratio_with_metadata,
    copy_ev_data,
    is_base_ratio,
    ratio_dir_name,
    summarize_v2g_participation,
)


DEFAULT_RATIOS = [0.0, 0.1, 0.2, 0.279, 0.4, 0.5]
OUTPUT_ROOT = Path("results/v2g_ratio_sensitivity")
DEFAULT_MODEL_CANDIDATES = (
    Path("models/td3_yearly_single_seed_{seed}/best/best_model.zip"),
    Path("models/td3_yearly_single_seed_42/best/best_model.zip"),
    Path("models/td3_yearly_single/best/best_model.zip"),
)

EXTRA_DAILY_COLUMNS = (
    "v2g_ratio_target",
    "v2g_ratio_actual",
    "n_v2g_ev",
    "n_total_ev",
    "total_cost_plus_penalty",
    "total_ev_charge_kwh",
    "total_ev_discharge_kwh",
    "total_ev_export_overlap_kwh",
)

EXTRA_CONFIG_COLUMNS = (
    "v2g_ratio_target",
    "v2g_ratio_actual",
    "target_v2g_ev_count",
    "actual_v2g_ev_count",
    "total_ev_count",
    "v2g_selection_rule",
    "v2g_field_name",
    "p_dis_field_name",
    "vehicle_type_field_name",
    "ev_v2g_buffer_soc_margin",
    "deterministic",
    "use_existing_v2g_for_base",
    "ratio_output_dir",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated TD3 V2G participation-ratio sensitivity tests."
    )
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--use-existing-v2g-for-base",
        type=parse_bool,
        default=True,
        help="For ratio=0.279, keep the original EV V2G flags instead of reselecting vehicles.",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def clean_run_name(run_name: str | None, seed: int) -> str:
    if run_name is None:
        return f"seed_{int(seed)}"
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(run_name).strip()
    )
    return cleaned or f"seed_{int(seed)}"


def resolve_model_path(seed: int, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Explicit model path not found: {path}")
        return path

    seen: set[Path] = set()
    candidates: list[Path] = []
    for template in DEFAULT_MODEL_CANDIDATES:
        candidate = Path(str(template).format(seed=int(seed)))
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No TD3 model found. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def extend_base_export_columns() -> None:
    daily_columns = list(base.DAILY_SUMMARY_COLUMNS)
    for column in EXTRA_DAILY_COLUMNS:
        if column not in daily_columns:
            daily_columns.append(column)
    base.DAILY_SUMMARY_COLUMNS = tuple(daily_columns)

    config_columns = list(base.CONFIG_COLUMNS)
    for column in EXTRA_CONFIG_COLUMNS:
        if column not in config_columns:
            config_columns.append(column)
    base.CONFIG_COLUMNS = tuple(config_columns)


def numeric_sum(infos: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(item.get(key, 0.0) or 0.0) for item in infos))


def numeric_avg(infos: list[dict[str, Any]], key: str) -> float:
    return numeric_sum(infos, key) / max(len(infos), 1)


def rollout_one_day(
    model: Any,
    env: Any,
    *,
    force_zero_ev_discharge: bool,
) -> tuple[list[dict[str, Any]], float]:
    obs, _ = env.reset()
    done = False
    infos: list[dict[str, Any]] = []
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).copy()
        if force_zero_ev_discharge:
            if action.shape[0] < 4:
                raise ValueError(f"TD3 action must have at least 4 entries, got {action.shape}")
            action[3] = 0.0
        obs, reward, terminated, truncated, step_info = env.step(action)
        infos.append(step_info)
        total_reward += float(reward)
        done = bool(terminated or truncated)

    return infos, total_reward


def make_base_existing_metadata(ev_data: Any, target_ratio: float) -> V2GRatioMetadata:
    metadata = summarize_v2g_participation(ev_data)
    metadata.ratio = float(target_ratio)
    metadata.target_v2g_ev_count = int(round(metadata.total_ev_count * float(target_ratio)))
    metadata.v2g_selection_rule = "existing_v2g_flags_base_case"
    return metadata


def make_ratio_ev_provider(
    ev_provider: Any,
    *,
    ratio: float,
    seed: int,
    use_existing_base: bool,
) -> Any:
    use_existing = bool(use_existing_base and is_base_ratio(ratio))

    def provider(case: Any) -> dict[str, Any]:
        ev_data = ev_provider(case)
        if use_existing:
            return copy_ev_data(ev_data)
        modified, _ = apply_v2g_participation_ratio_with_metadata(
            ev_data,
            ratio,
            seed=seed,
            emit_warnings=False,
        )
        return modified

    return provider


def build_summary_row(
    *,
    case_idx: int,
    case: Any,
    infos: list[dict[str, Any]],
    total_reward: float,
    cfg: Any,
    metadata: V2GRatioMetadata,
) -> dict[str, Any]:
    total_system_cost = numeric_sum(infos, "system_cost")
    total_penalties = numeric_sum(infos, "penalty_cost")
    total_ev_charge_kwh = numeric_sum(infos, "p_ev_ch") * float(cfg.dt)
    total_ev_discharge_kwh = numeric_sum(infos, "p_ev_dis") * float(cfg.dt)
    total_ev_export_overlap_kwh = numeric_sum(infos, "ev_export_overlap_kwh")

    final_info = infos[-1] if infos else {}
    missing_terminal_info = [
        key
        for key in (
            "final_ees_soc",
            "ees_soc_init",
            "terminal_ees_required_soc",
            "episode_terminal_ees_shortage_kwh",
            "episode_penalty_terminal_ees_soc",
            "ees_terminal_soc_feasible",
        )
        if key not in final_info
    ]
    if missing_terminal_info:
        print(
            "WARNING: EES terminal SOC info missing: "
            f"case_index={case_idx}, missing={missing_terminal_info}; fallback values used."
        )

    final_ees_soc = float(final_info.get("final_ees_soc", final_info.get("ees_soc", 0.0)))
    ees_soc_init = float(final_info.get("ees_soc_init", final_info.get("ees_soc_episode_init", 0.0)))
    terminal_ees_required_soc = float(final_info.get("terminal_ees_required_soc", 0.0))
    terminal_ees_shortage_kwh = float(final_info.get("episode_terminal_ees_shortage_kwh", 0.0))
    total_penalty_terminal_ees_soc = float(final_info.get("episode_penalty_terminal_ees_soc", 0.0))
    ees_terminal_soc_feasible = bool(final_info.get("ees_terminal_soc_feasible", True))

    return {
        "case_index": int(case_idx),
        "month": case.month,
        "day_of_year": case.day_of_year,
        "season": case.season,
        "split": case.set_name,
        "total_reward": float(total_reward),
        "total_reward_raw": numeric_sum(infos, "reward_raw"),
        "total_system_cost": total_system_cost,
        "total_penalties": total_penalties,
        "total_cost_plus_penalty": total_system_cost + total_penalties,
        "total_guide_reward": numeric_sum(infos, "guide_reward"),
        "total_cost_grid": numeric_sum(infos, "cost_grid"),
        "total_cost_gas": numeric_sum(infos, "cost_gas"),
        "total_cost_deg": numeric_sum(infos, "cost_deg"),
        "total_cost_om": numeric_sum(infos, "cost_om"),
        "total_unmet_e": numeric_sum(infos, "unmet_e"),
        "total_unmet_h": numeric_sum(infos, "unmet_h"),
        "total_unmet_c": numeric_sum(infos, "unmet_c"),
        "total_surplus_e": numeric_sum(infos, "surplus_e"),
        "total_surplus_h": numeric_sum(infos, "surplus_h"),
        "total_surplus_c": numeric_sum(infos, "surplus_c"),
        "total_grid_overflow": numeric_sum(infos, "grid_overflow"),
        "total_penalty_unserved_e": numeric_sum(infos, "penalty_unserved_e"),
        "total_penalty_unserved_h": numeric_sum(infos, "penalty_unserved_h"),
        "total_penalty_unserved_c": numeric_sum(infos, "penalty_unserved_c"),
        "total_penalty_depart_energy": numeric_sum(infos, "penalty_depart_energy"),
        "total_penalty_depart_energy_soft": numeric_sum(infos, "penalty_depart_energy_soft"),
        "total_penalty_depart_energy_mid": numeric_sum(infos, "penalty_depart_energy_mid"),
        "total_penalty_depart_energy_hard": numeric_sum(infos, "penalty_depart_energy_hard"),
        "total_penalty_depart_risk": numeric_sum(infos, "penalty_depart_risk"),
        "total_penalty_surplus_e": numeric_sum(infos, "penalty_surplus_e"),
        "total_penalty_surplus_h": numeric_sum(infos, "penalty_surplus_h"),
        "total_penalty_surplus_c": numeric_sum(infos, "penalty_surplus_c"),
        "total_penalty_export_e": numeric_sum(infos, "penalty_export_e"),
        "total_penalty_ev_export_guard": numeric_sum(infos, "penalty_ev_export_guard"),
        "total_penalty_terminal_ees_soc": total_penalty_terminal_ees_soc,
        "total_depart_energy_shortage_kwh": numeric_sum(infos, "depart_energy_shortage_kwh"),
        "total_depart_shortage_soft_kwh": numeric_sum(infos, "depart_shortage_soft_kwh"),
        "total_depart_shortage_mid_kwh": numeric_sum(infos, "depart_shortage_mid_kwh"),
        "total_depart_shortage_hard_kwh": numeric_sum(infos, "depart_shortage_hard_kwh"),
        "total_depart_risk_energy_kwh": numeric_sum(infos, "depart_risk_energy_kwh"),
        "total_reward_storage_discharge_bonus": numeric_sum(infos, "reward_storage_discharge_bonus"),
        "total_reward_storage_charge_bonus": numeric_sum(infos, "reward_storage_charge_bonus"),
        "total_reward_ev_target_timing_bonus": numeric_sum(infos, "reward_ev_target_timing_bonus"),
        "total_storage_peak_shaved_kwh": numeric_sum(infos, "storage_peak_shaved_kwh"),
        "total_storage_charge_rewarded_kwh": numeric_sum(infos, "storage_charge_rewarded_kwh"),
        "total_ees_charge_rewarded_kwh": numeric_sum(infos, "ees_charge_rewarded_kwh"),
        "total_ev_flex_target_charge_kwh": numeric_sum(infos, "ev_flex_target_charge_kwh"),
        "total_ev_buffer_charge_kwh": numeric_sum(infos, "ev_buffer_charge_kwh"),
        "total_low_value_charge_kwh": numeric_sum(infos, "low_value_charge_kwh"),
        "total_gt_export_clip": numeric_sum(infos, "p_gt_export_clip"),
        "total_gt_export_clip_steps": int(sum(1 for item in infos if float(item.get("p_gt_export_clip", 0.0)) > base.ALERT_TOL)),
        "total_gt_safe_infeasible_steps": int(sum(1 for item in infos if not bool(item.get("gt_safe_feasible", True)))),
        "total_grid_buy_kwh": numeric_sum(infos, "p_grid_buy") * float(cfg.dt),
        "total_grid_sell_kwh": numeric_sum(infos, "p_grid_sell") * float(cfg.dt),
        "total_ev_charge_kwh": total_ev_charge_kwh,
        "total_ev_discharge_kwh": total_ev_discharge_kwh,
        "total_ev_export_overlap_kwh": total_ev_export_overlap_kwh,
        "avg_p_gt_kw": numeric_avg(infos, "p_gt"),
        "avg_p_ev_ch_kw": numeric_avg(infos, "p_ev_ch"),
        "avg_p_ev_dis_kw": numeric_avg(infos, "p_ev_dis"),
        "avg_p_ees_ch_kw": numeric_avg(infos, "p_ees_ch"),
        "avg_p_ees_dis_kw": numeric_avg(infos, "p_ees_dis"),
        "ees_soc_episode_init": float(infos[0].get("ees_soc_episode_init", 0.0)) if infos else 0.0,
        "final_ees_soc": final_ees_soc,
        "ees_soc_init": ees_soc_init,
        "terminal_ees_required_soc": terminal_ees_required_soc,
        "terminal_ees_shortage_kwh": terminal_ees_shortage_kwh,
        "ees_terminal_soc_feasible": ees_terminal_soc_feasible,
        "final_gt_power": float(infos[-1].get("p_gt", 0.0)) if infos else 0.0,
        "v2g_ratio_target": float(metadata.ratio),
        "v2g_ratio_actual": float(metadata.actual_v2g_ratio),
        "n_v2g_ev": int(metadata.actual_v2g_ev_count),
        "n_total_ev": int(metadata.total_ev_count),
    }


def write_runtime_summary(
    path: Path,
    *,
    metadata: V2GRatioMetadata,
    cfg: Any,
    model_path: Path,
    seed: int,
    run_name: str,
    ratio_label: str,
    device: str,
    n_days: int,
    n_steps: int,
    test_duration_seconds: float,
    use_existing_base: bool,
    force_zero_ev_discharge: bool,
) -> dict[str, Any]:
    time_per_day = test_duration_seconds / n_days if n_days > 0 else 0.0
    time_per_step = test_duration_seconds / n_steps if n_steps > 0 else 0.0
    row = {
        "method": "TD3",
        "run_name": run_name,
        "ratio_label": ratio_label,
        "n_days": int(n_days),
        "n_steps": int(n_steps),
        "test_duration_seconds": round(float(test_duration_seconds), 6),
        "time_per_day_seconds": round(float(time_per_day), 6),
        "time_per_step_seconds": round(float(time_per_step), 6),
        **metadata.to_runtime_dict(),
        "ev_v2g_buffer_soc_margin": float(getattr(cfg, "ev_v2g_buffer_soc_margin", 0.0)),
        "model_path": str(model_path),
        "seed": int(seed),
        "deterministic": True,
        "device": str(device),
        "use_existing_v2g_for_base": bool(use_existing_base),
        "force_zero_ev_discharge_action": bool(force_zero_ev_discharge),
        "output_dir": str(path.parent),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return row


def run_one_ratio(
    *,
    ratio: float,
    seed: int,
    run_name: str,
    output_root: Path,
    use_existing_base: bool,
    modules: dict[str, Any],
    model: Any,
    model_path: Path,
    loader: Any,
    train_cases: list[Any],
    val_cases: list[Any],
    test_cases: list[Any],
    ev_provider: Any,
    excel_engine: str,
) -> dict[str, Any]:
    pd = modules["pd"]
    IESConfig = modules["IESConfig"]
    ParkIESEnv = modules["ParkIESEnv"]

    ratio = float(ratio)
    label = ratio_dir_name(ratio)
    result_dir = Path(output_root) / run_name / label
    summary_csv = result_dir / "daily_summary.csv"
    timeseries_csv = result_dir / "timeseries_detail.csv"
    export_xlsx = result_dir / "td3_v2g_ratio_test_export.xlsx"
    runtime_json = result_dir / "runtime_summary.json"
    result_dir.mkdir(parents=True, exist_ok=True)

    base.RESULT_DIR = result_dir
    base.SUMMARY_CSV = summary_csv
    base.TIMESERIES_CSV = timeseries_csv
    base.TEST_EXPORT_XLSX = export_xlsx
    base.RUNTIME_SUMMARY_JSON = runtime_json
    base.MODEL_PATH = model_path
    base.SEED = int(seed)
    base.RUN_NAME = run_name
    extend_base_export_columns()

    cfg = IESConfig(episode_length=24, dt=1.0, reward_scale=base.REWARD_SCALE)
    force_zero_ev_discharge = False
    if abs(ratio) <= 1e-12:
        setattr(cfg, "ev_v2g_buffer_soc_margin", 0.0)
        force_zero_ev_discharge = True

    probe_case = test_cases[0]
    probe_raw_ev_data = ev_provider(probe_case)
    if use_existing_base and is_base_ratio(ratio):
        probe_ev_data = copy_ev_data(probe_raw_ev_data)
        metadata = make_base_existing_metadata(probe_raw_ev_data, ratio)
    else:
        probe_ev_data, metadata = apply_v2g_participation_ratio_with_metadata(
            probe_raw_ev_data,
            ratio,
            seed=seed,
            emit_warnings=True,
        )

    probe_env = ParkIESEnv(cfg=cfg, ts_data=probe_case.ts_data, ev_data=probe_ev_data)
    print(
        f"[ratio] {label}: target={metadata.target_v2g_ev_count}, "
        f"actual={metadata.actual_v2g_ev_count}/{metadata.total_ev_count} "
        f"({metadata.actual_v2g_ratio:.4f}), obs_dim={probe_env.obs_dim}, "
        f"buffer_margin={getattr(cfg, 'ev_v2g_buffer_soc_margin', None)}"
    )
    probe_env.close()

    ratio_ev_provider = make_ratio_ev_provider(
        ev_provider,
        ratio=ratio,
        seed=seed,
        use_existing_base=use_existing_base,
    )

    run_started_at = datetime.now().astimezone()
    run_id = run_started_at.strftime("%Y%m%d_%H%M%S")
    timestamp_start = run_started_at.isoformat(timespec="seconds")
    summary_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    test_started_perf = perf_counter()
    for case_idx, case in enumerate(test_cases):
        ev_data = ratio_ev_provider(case)
        env = ParkIESEnv(cfg=cfg, ts_data=case.ts_data, ev_data=ev_data)
        infos, total_reward = rollout_one_day(
            model,
            env,
            force_zero_ev_discharge=force_zero_ev_discharge,
        )
        summary_row = build_summary_row(
            case_idx=case_idx,
            case=case,
            infos=infos,
            total_reward=total_reward,
            cfg=cfg,
            metadata=metadata,
        )
        summary_rows.append(summary_row)
        for step_info in infos:
            step_rows.append(base.build_step_row(case_idx, case, step_info))
        env.close()

        print(
            f"[{label} {case_idx + 1:02d}/{len(test_cases)}] "
            f"day={case.day_of_year}, cost+penalty={summary_row['total_cost_plus_penalty']:.4f}, "
            f"ev_dis={summary_row['total_ev_discharge_kwh']:.4f}, "
            f"grid_buy={summary_row['total_grid_buy_kwh']:.4f}, "
            f"terminal_shortage={summary_row['terminal_ees_shortage_kwh']:.6f}"
        )

    test_duration_seconds = max(perf_counter() - test_started_perf, 0.0)
    alert_rows = base.build_alert_rows(step_rows)

    base.save_csv(summary_rows, summary_csv, base.DAILY_SUMMARY_COLUMNS)
    base.save_csv(step_rows, timeseries_csv, base.STEP_DETAIL_COLUMNS)

    runtime_summary = write_runtime_summary(
        runtime_json,
        metadata=metadata,
        cfg=cfg,
        model_path=model_path,
        seed=seed,
        run_name=run_name,
        ratio_label=label,
        device=str(model.device),
        n_days=len(summary_rows),
        n_steps=len(step_rows),
        test_duration_seconds=test_duration_seconds,
        use_existing_base=use_existing_base,
        force_zero_ev_discharge=force_zero_ev_discharge,
    )

    config_row = base.build_config_row(
        cfg,
        run_id=run_id,
        timestamp_start=timestamp_start,
        n_train_cases=len(train_cases),
        n_val_cases=len(val_cases),
        n_test_cases=len(test_cases),
        model=model,
    )
    config_row.update(
        {
            "v2g_ratio_target": float(metadata.ratio),
            "v2g_ratio_actual": float(metadata.actual_v2g_ratio),
            "target_v2g_ev_count": int(metadata.target_v2g_ev_count),
            "actual_v2g_ev_count": int(metadata.actual_v2g_ev_count),
            "total_ev_count": int(metadata.total_ev_count),
            "v2g_selection_rule": metadata.v2g_selection_rule,
            "v2g_field_name": metadata.v2g_field_name,
            "p_dis_field_name": metadata.p_dis_field_name,
            "vehicle_type_field_name": metadata.vehicle_type_field_name,
            "ev_v2g_buffer_soc_margin": float(getattr(cfg, "ev_v2g_buffer_soc_margin", 0.0)),
            "deterministic": True,
            "use_existing_v2g_for_base": bool(use_existing_base),
            "ratio_output_dir": str(result_dir),
        }
    )
    base.export_test_workbook(
        pd,
        output_path=export_xlsx,
        excel_engine=excel_engine,
        step_rows=step_rows,
        summary_rows=summary_rows,
        config_row=config_row,
        alert_rows=alert_rows,
    )

    mean_cost_plus_penalty = float(
        np.mean([row["total_cost_plus_penalty"] for row in summary_rows])
    )
    mean_ev_discharge = float(
        np.mean([row["total_ev_discharge_kwh"] for row in summary_rows])
    )
    print(
        f"[done] {label}: daily_summary={summary_csv}, "
        f"mean_cost+penalty={mean_cost_plus_penalty:.4f}, "
        f"mean_ev_discharge={mean_ev_discharge:.4f}, runtime={runtime_summary['test_duration_seconds']:.3f}s"
    )
    return runtime_summary


def main() -> None:
    args = parse_args()
    base.configure_stdio()

    seed = int(args.seed)
    run_name = clean_run_name(args.run_name, seed)
    ratios = [float(ratio) for ratio in args.ratios]
    model_path = resolve_model_path(seed, args.model_path)

    base.SEED = seed
    base.REQUESTED_DEVICE = str(args.device).strip() or "auto"
    base.MODEL_PATH = model_path

    excel_engine = base.preflight_check()
    modules = base.import_runtime_modules()
    np_module = modules["np"]
    th = modules["th"]
    TD3 = modules["TD3"]
    YearlyCSVDataLoader = modules["YearlyCSVDataLoader"]
    YearlyEVProvider = modules["YearlyEVProvider"]

    base.set_random_seed(seed, np_module, th)
    resolved_device = base.resolve_sb3_device(th, base.REQUESTED_DEVICE)
    base.print_torch_runtime_summary(
        th,
        requested_device=base.REQUESTED_DEVICE,
        resolved_device=resolved_device,
    )

    print(f"[startup] run_name={run_name}")
    print(f"[startup] model_path={model_path}")
    print(f"[startup] ratios={ratios}")
    print(f"[startup] output_root={Path(args.output_root)}")
    print("[startup] Loading yearly CSV cases...")
    loader = YearlyCSVDataLoader(base.YEARLY_CSV_PATH, val_days_per_month=base.VAL_DAYS_PER_MONTH)
    train_cases, val_cases, test_cases = loader.load()
    loader.summary()
    if not test_cases:
        raise RuntimeError("Test set is empty. Please check the split configuration.")

    print("[startup] Loading EV provider...")
    ev_provider = YearlyEVProvider(base.YEARLY_EV_PATH)

    print("[startup] Loading TD3 model without retraining...")
    model = TD3.load(str(model_path), device=resolved_device)

    runtime_rows = []
    for ratio in ratios:
        runtime_rows.append(
            run_one_ratio(
                ratio=ratio,
                seed=seed,
                run_name=run_name,
                output_root=Path(args.output_root),
                use_existing_base=bool(args.use_existing_v2g_for_base),
                modules=modules,
                model=model,
                model_path=model_path,
                loader=loader,
                train_cases=train_cases,
                val_cases=val_cases,
                test_cases=test_cases,
                ev_provider=ev_provider,
                excel_engine=excel_engine,
            )
        )

    print("\n========== V2G Ratio Sensitivity Completed ==========")
    for row in runtime_rows:
        print(
            f"{row['ratio_label']}: target_ratio={row['ratio']:.4f}, "
            f"actual={row['actual_v2g_ev_count']}/{row['total_ev_count']} "
            f"({row['actual_v2g_ratio']:.4f}), output={row['output_dir']}"
        )


if __name__ == "__main__":
    main()
