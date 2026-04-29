from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_YEARLY_CSV_PATH = BASE_DIR / "yearly_data_sci.csv"
DEFAULT_SUMMARY_CSV_PATH = BASE_DIR / "results" / "td3_yearly_test" / "daily_summary.csv"
DEFAULT_TIMESERIES_CSV_PATH = BASE_DIR / "results" / "td3_yearly_test" / "timeseries_detail.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "td3_yearly_test" / "plots"
DEFAULT_TYPICAL_SUMMARY_PATH = BASE_DIR / "results" / "td3_yearly_test" / "typical_days_summary.csv"
DEFAULT_CASE_ANALYSIS_PATH = BASE_DIR / "results" / "td3_yearly_test" / "abnormal_case_summary.csv"
DEFAULT_ANALYSIS_REPORT_PATH = BASE_DIR / "results" / "td3_yearly_test" / "test_analysis_report.md"

SEASON_ORDER = ["spring", "summer", "autumn", "winter"]
AC_COP = 1.20
BAR_WIDTH = 0.86
PNG_DPI = 300
EPS = 1e-9
DISPLAY_EPS = 1e-6

COLOR_MAP = {
    "PV": "#D6D85C",
    "WT": "#AAB6E8",
    "GT": "#F4B548",
    "Grid buy": "#99A6B6",
    "Grid sell": "#6E87A8",
    "EES discharge": "#F29A92",
    "EES charge": "#4B88BF",
    "EV discharge": "#CF8DD9",
    "EV charge": "#78B9E7",
    "EC electric input": "#83D7D0",
    "WHB heat": "#A89CC6",
    "GB heat": "#F6B55E",
    "AC heat input": "#B59889",
    "AC cooling": "#37B8A8",
    "EC cooling": "#92DFD8",
    "Load": "#222222",
    "Penalty": "#D62728",
    "Cost": "#4B88BF",
    "SOC": "#111111",
}

COST_COMPONENT_SPECS = [
    ("total_cost_grid", "Grid purchase", "#7BA6D8"),
    ("total_cost_gas", "Gas", "#F4B548"),
    ("total_cost_deg", "Battery degradation", "#CF8DD9"),
    ("total_cost_om", "O&M", "#8E7C6E"),
]

PENALTY_COMPONENT_SPECS = [
    ("total_penalty_unserved_e", "Unserved electricity", "#D62728"),
    ("total_penalty_unserved_h", "Unserved heat", "#FF7F0E"),
    ("total_penalty_unserved_c", "Unserved cooling", "#9467BD"),
    ("total_penalty_depart_energy", "Depart energy shortage", "#E15759"),
    ("total_penalty_depart_risk", "Depart risk", "#B279A2"),
    ("total_penalty_surplus_e", "Electric surplus", "#F28E2B"),
    ("total_penalty_surplus_h", "Heat surplus", "#EDC948"),
    ("total_penalty_surplus_c", "Cooling surplus", "#59A14F"),
    ("total_penalty_export_e", "Grid export", "#4E79A7"),
    ("total_penalty_ev_export_guard", "EV export guard", "#76B7B2"),
    ("total_penalty_terminal_ees_soc", "EES terminal SOC penalty", "#9467BD"),
]

ALWAYS_SHOW_COMPONENT_COLUMNS = {"total_penalty_terminal_ees_soc"}


def configure_matplotlib() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_daily_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"daily_summary.csv not found: {path}")
    df = pd.read_csv(path)
    required = [
        "case_index",
        "month",
        "day_of_year",
        "season",
        "split",
        "total_system_cost",
        "total_penalties",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"daily_summary.csv is missing columns: {missing}")
    return df.sort_values("case_index").reset_index(drop=True)


def load_timeseries(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"timeseries_detail.csv not found: {path}")
    df = pd.read_csv(path)
    required = [
        "case_index",
        "month",
        "day_of_year",
        "season",
        "time_step",
        "p_grid_buy",
        "p_grid_sell",
        "p_gt",
        "p_whb_heat",
        "p_gb_heat",
        "p_ac_cool",
        "p_ec_cool",
        "p_ec_elec_in",
        "p_ev_ch",
        "p_ev_dis",
        "p_ees_ch",
        "p_ees_dis",
        "ees_soc",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"timeseries_detail.csv is missing columns: {missing}")
    return df.sort_values(["case_index", "time_step"]).reset_index(drop=True)


def load_source_day(yearly_csv_path: Path, day_of_year: int) -> pd.DataFrame:
    if not yearly_csv_path.exists():
        raise FileNotFoundError(f"yearly_data_sci.csv not found: {yearly_csv_path}")
    df = pd.read_csv(yearly_csv_path)
    required = ["Day", "Hour", "Electric Load", "Heat Load", "Cold Load", "PV", "Wind"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"yearly_data_sci.csv is missing columns: {missing}")

    day_df = df[df["Day"] == day_of_year].copy().sort_values("Hour").reset_index(drop=True)
    if len(day_df) != 24:
        raise ValueError(f"Expected 24 rows for Day={day_of_year}, got {len(day_df)}")
    return day_df


def filter_test_rows(summary_df: pd.DataFrame) -> pd.DataFrame:
    if "split" not in summary_df.columns:
        return summary_df.copy()
    test_df = summary_df[summary_df["split"].astype(str).str.lower() == "test"].copy()
    return test_df if not test_df.empty else summary_df.copy()


def get_existing_component_specs(
    df: pd.DataFrame,
    specs: list[tuple[str, str, str]],
    *,
    min_total: float = DISPLAY_EPS,
) -> list[tuple[str, str, str]]:
    existing: list[tuple[str, str, str]] = []
    for col, label, color in specs:
        if col not in df.columns:
            continue
        if float(df[col].abs().sum()) <= min_total and col not in ALWAYS_SHOW_COMPONENT_COLUMNS:
            continue
        existing.append((col, label, color))
    return existing


def dominant_component_from_row(
    row: pd.Series,
    specs: list[tuple[str, str, str]],
    total_col: str,
) -> tuple[str, float, float]:
    total_value = float(row.get(total_col, 0.0) or 0.0)
    best_label = "None"
    best_value = 0.0
    for col, label, _ in specs:
        value = float(row.get(col, 0.0) or 0.0)
        if value > best_value:
            best_label = label
            best_value = value
    share_pct = 0.0 if abs(total_value) <= EPS else best_value / total_value * 100.0
    return best_label, best_value, share_pct


def detect_abnormal_cases(
    test_df: pd.DataFrame,
    metric_col: str,
    *,
    fallback_top_k: int = 5,
) -> dict[str, object]:
    metric = test_df[metric_col].astype(float).fillna(0.0)
    q1 = float(metric.quantile(0.25))
    q3 = float(metric.quantile(0.75))
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr

    if float(metric.max()) <= DISPLAY_EPS:
        flagged_df = test_df.iloc[0:0].copy()
        method = "all_zero"
    else:
        flagged_df = test_df[metric > upper + DISPLAY_EPS].copy().sort_values(metric_col, ascending=False)
        method = "iqr"
        if flagged_df.empty:
            top_k = min(max(int(fallback_top_k), 1), len(test_df))
            flagged_df = test_df.nlargest(top_k, metric_col).copy()
            upper = float(flagged_df[metric_col].min())
            method = f"top_{top_k}"

    return {
        "metric_col": metric_col,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "upper": upper,
        "method": method,
        "flagged_df": flagged_df.reset_index(drop=True),
    }


def describe_abnormal_method(result: dict[str, object], metric_label: str) -> str:
    method = str(result.get("method", ""))
    upper = float(result.get("upper", 0.0) or 0.0)
    if method == "all_zero":
        return f"{metric_label} 全部接近 0，未识别到正异常。"
    if method.startswith("top_"):
        top_k = method.split("_", 1)[1]
        return f"{metric_label} 按 IQR 未出现离群点，改为展示最高 {top_k} 个 case。"
    return f"{metric_label} 按 IQR 规则识别异常，上阈值为 {upper:.2f}。"


def build_case_axis_ticks(case_indices: np.ndarray) -> np.ndarray:
    if len(case_indices) == 0:
        return case_indices
    step = max(int(np.ceil(len(case_indices) / 15.0)), 1)
    ticks = case_indices[::step]
    if ticks[-1] != case_indices[-1]:
        ticks = np.append(ticks, case_indices[-1])
    return ticks


def style_case_axis(ax: plt.Axes, case_indices: np.ndarray) -> None:
    if len(case_indices) == 0:
        return
    ax.set_xlim(float(case_indices.min()) - 0.8, float(case_indices.max()) + 0.8)
    ax.set_xticks(build_case_axis_ticks(case_indices))
    ax.grid(True, alpha=0.22, axis="y")


def annotate_flagged_cases(
    ax: plt.Axes,
    flagged_df: pd.DataFrame,
    metric_col: str,
    *,
    color: str = "#7A1E1E",
) -> None:
    if flagged_df.empty:
        return
    for _, row in flagged_df.iterrows():
        x = float(row["case_index"])
        y = float(row[metric_col])
        ax.annotate(
            f"{int(row['case_index'])}",
            xy=(x, y),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=color,
            rotation=0,
        )


def component_share_text(
    df: pd.DataFrame,
    total_col: str,
    specs: list[tuple[str, str, str]],
) -> str:
    total_value = float(df[total_col].sum()) if total_col in df.columns else 0.0
    if abs(total_value) <= EPS:
        return "No material total."

    parts: list[str] = []
    for col, label, _ in specs:
        if col not in df.columns:
            continue
        value = float(df[col].sum())
        if abs(value) <= DISPLAY_EPS:
            continue
        parts.append(f"{label}: {value:,.1f} ({value / total_value * 100.0:.1f}%)")
    return " | ".join(parts) if parts else "No material components."


def build_abnormal_case_summary(
    summary_df: pd.DataFrame,
    cost_result: dict[str, object],
    penalty_result: dict[str, object],
) -> pd.DataFrame:
    test_df = filter_test_rows(summary_df).sort_values("case_index").reset_index(drop=True)
    cost_cases = set(pd.to_numeric(cost_result["flagged_df"]["case_index"], errors="coerce").dropna().astype(int).tolist())
    penalty_cases = set(pd.to_numeric(penalty_result["flagged_df"]["case_index"], errors="coerce").dropna().astype(int).tolist())
    target_cases = sorted(cost_cases | penalty_cases)

    rows: list[dict[str, object]] = []
    for case_index in target_cases:
        matched = test_df[test_df["case_index"].astype(int) == int(case_index)]
        if matched.empty:
            continue
        row = matched.iloc[0]
        cost_label, cost_value, cost_share = dominant_component_from_row(row, COST_COMPONENT_SPECS, "total_system_cost")
        penalty_label, penalty_value, penalty_share = dominant_component_from_row(
            row,
            PENALTY_COMPONENT_SPECS,
            "total_penalties",
        )
        rows.append(
            {
                "case_index": int(row["case_index"]),
                "month": int(row["month"]),
                "day_of_year": int(row["day_of_year"]),
                "season": str(row["season"]),
                "total_system_cost": float(row.get("total_system_cost", 0.0)),
                "total_penalties": float(row.get("total_penalties", 0.0)),
                "cost_abnormal": int(int(row["case_index"]) in cost_cases),
                "penalty_abnormal": int(int(row["case_index"]) in penalty_cases),
                "dominant_cost_component": cost_label,
                "dominant_cost_value": cost_value,
                "dominant_cost_share_pct": cost_share,
                "dominant_penalty_component": penalty_label,
                "dominant_penalty_value": penalty_value,
                "dominant_penalty_share_pct": penalty_share,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "case_index",
                "month",
                "day_of_year",
                "season",
                "total_system_cost",
                "total_penalties",
                "cost_abnormal",
                "penalty_abnormal",
                "dominant_cost_component",
                "dominant_cost_value",
                "dominant_cost_share_pct",
                "dominant_penalty_component",
                "dominant_penalty_value",
                "dominant_penalty_share_pct",
            ]
        )

    abnormal_df = pd.DataFrame(rows)
    return abnormal_df.sort_values(
        ["cost_abnormal", "penalty_abnormal", "total_system_cost", "total_penalties"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def get_case_timeseries(
    timeseries_df: pd.DataFrame,
    *,
    case_index: int | None = None,
    day_of_year: int | None = None,
) -> pd.DataFrame:
    if case_index is not None:
        case_df = timeseries_df[timeseries_df["case_index"] == case_index].copy()
    elif day_of_year is not None:
        case_df = timeseries_df[timeseries_df["day_of_year"] == day_of_year].copy()
    else:
        raise ValueError("Provide case_index or day_of_year.")

    if case_df.empty:
        raise ValueError(f"No matching timeseries rows for case_index={case_index}, day_of_year={day_of_year}")
    return case_df.sort_values("time_step").reset_index(drop=True)


def get_case_summary(summary_df: pd.DataFrame, case_index: int) -> pd.Series | None:
    matched = summary_df[summary_df["case_index"] == case_index]
    if matched.empty:
        return None
    return matched.iloc[0]


def select_typical_days(summary_df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    test_df = filter_test_rows(summary_df)
    if test_df.empty:
        raise ValueError("daily_summary.csv contains no test rows.")

    rows = []
    for season in SEASON_ORDER:
        season_df = test_df[test_df["season"].astype(str).str.lower() == season].copy()
        if season_df.empty:
            raise ValueError(f"No test rows for season={season}")
        season_mean_cost = float(season_df["total_system_cost"].mean())
        season_df["distance_to_mean_cost"] = (season_df["total_system_cost"] - season_mean_cost).abs()
        season_df["season_mean_cost"] = season_mean_cost
        rows.append(
            season_df.sort_values(["distance_to_mean_cost", "day_of_year", "case_index"]).iloc[0]
        )

    typical_df = pd.DataFrame(rows).reset_index(drop=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    typical_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return typical_df


def stacked_bar(ax: plt.Axes, hours: np.ndarray, components: list[tuple[str, np.ndarray]], *, positive: bool) -> None:
    base = np.zeros_like(hours, dtype=float)
    sign = 1.0 if positive else -1.0
    for label, values in components:
        heights = sign * values
        ax.bar(
            hours,
            heights,
            width=BAR_WIDTH,
            bottom=base,
            label=label,
            color=COLOR_MAP.get(label),
            edgecolor="white",
            linewidth=0.3,
        )
        base = base + heights


def style_balance_axis(ax: plt.Axes, ylabel: str, *, show_xlabel: bool) -> None:
    if show_xlabel:
        ax.set_xlabel("Time (hour)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(1, 25))
    ax.set_xlim(0.5, 24.5)
    ax.grid(True, alpha=0.25, axis="y")
    ax.axhline(0.0, color="black", linewidth=1.0)


def draw_empty_panel(ax: plt.Axes, title: str, message: str = "No data") -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#666666")
    ax.grid(True, alpha=0.15)


def format_metric(value: object, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.{digits}f}"
    return str(value)


def format_bool_metric(value: object) -> str:
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes", "y"}:
        return "True"
    if text in {"false", "0", "0.0", "no", "n"}:
        return "False"
    return str(value)


def terminal_ees_stats(test_df: pd.DataFrame) -> dict[str, float | int]:
    final_soc = (
        pd.to_numeric(test_df["final_ees_soc"], errors="coerce")
        if "final_ees_soc" in test_df.columns
        else pd.Series(np.nan, index=test_df.index)
    )
    shortage = (
        pd.to_numeric(test_df["terminal_ees_shortage_kwh"], errors="coerce")
        if "terminal_ees_shortage_kwh" in test_df.columns
        else pd.Series(0.0, index=test_df.index)
    ).fillna(0.0)
    penalty = (
        pd.to_numeric(test_df["total_penalty_terminal_ees_soc"], errors="coerce")
        if "total_penalty_terminal_ees_soc" in test_df.columns
        else pd.Series(0.0, index=test_df.index)
    ).fillna(0.0)
    if "ees_terminal_soc_feasible" in test_df.columns:
        feasible_text = test_df["ees_terminal_soc_feasible"].astype(str).str.strip().str.lower()
        feasible_num = pd.to_numeric(test_df["ees_terminal_soc_feasible"], errors="coerce")
        infeasible = feasible_text.isin({"false", "f", "no", "n"}) | (feasible_num == 0)
    else:
        infeasible = pd.Series(False, index=test_df.index)
    violation = infeasible | (shortage > 1e-6)
    return {
        "n_terminal_ees_violation_days": int(violation.sum()),
        "min_final_ees_soc": float(final_soc.min()) if final_soc.notna().any() else float("nan"),
        "sum_terminal_ees_shortage_kwh": float(shortage.sum()),
        "sum_penalty_terminal_ees_soc": float(penalty.sum()),
    }


def terminal_ees_stats_text(test_df: pd.DataFrame) -> str:
    stats = terminal_ees_stats(test_df)
    return (
        "EES terminal SOC\n"
        f"violations: {stats['n_terminal_ees_violation_days']}\n"
        f"min final SOC: {stats['min_final_ees_soc']:.4f}\n"
        f"shortage: {stats['sum_terminal_ees_shortage_kwh']:.2f} kWh\n"
        f"penalty: {stats['sum_penalty_terminal_ees_soc']:.2f}"
    )


def summary_text_box(summary_row: pd.Series | None) -> str:
    if summary_row is None:
        return "No daily summary row found."
    lines = [
        f"Total cost: {format_metric(summary_row.get('total_system_cost'))}",
        f"Penalties: {format_metric(summary_row.get('total_penalties'))}",
        f"Guide reward: {format_metric(summary_row.get('total_guide_reward'))}",
        f"Grid buy (kWh): {format_metric(summary_row.get('total_grid_buy_kwh'))}",
        f"Grid sell (kWh): {format_metric(summary_row.get('total_grid_sell_kwh'))}",
        f"Final EES SOC: {format_metric(summary_row.get('final_ees_soc'), 4)}",
        f"Required EES SOC: {format_metric(summary_row.get('terminal_ees_required_soc'), 4)}",
        f"EES terminal shortage: {format_metric(summary_row.get('terminal_ees_shortage_kwh'), 2)} kWh",
        f"EES terminal penalty: {format_metric(summary_row.get('total_penalty_terminal_ees_soc'), 2)}",
        f"EES terminal feasible: {format_bool_metric(summary_row.get('ees_terminal_soc_feasible'))}",
    ]
    return "\n".join(lines)


def plot_daily_cost_penalty_anomalies(
    summary_df: pd.DataFrame,
    output_dir: Path,
    cost_result: dict[str, object],
    penalty_result: dict[str, object],
) -> None:
    test_df = filter_test_rows(summary_df).sort_values("case_index").reset_index(drop=True)
    case_indices = test_df["case_index"].to_numpy(dtype=float)
    terminal_stats = terminal_ees_stats(test_df)
    print(
        "EES terminal SOC anomaly check: "
        f"n_terminal_ees_violation_days={terminal_stats['n_terminal_ees_violation_days']}, "
        f"min_final_ees_soc={terminal_stats['min_final_ees_soc']:.6f}, "
        f"sum_terminal_ees_shortage_kwh={terminal_stats['sum_terminal_ees_shortage_kwh']:.6f}, "
        f"sum_penalty_terminal_ees_soc={terminal_stats['sum_penalty_terminal_ees_soc']:.6f}"
    )

    fig, axes = plt.subplots(2, 1, figsize=(15.5, 10.0), sharex=True)

    configs = [
        (
            axes[0],
            "total_system_cost",
            "测试日总成本异常定位",
            "总成本",
            COLOR_MAP["Cost"],
            cost_result,
        ),
        (
            axes[1],
            "total_penalties",
            "测试日总惩罚异常定位",
            "总惩罚",
            COLOR_MAP["Penalty"],
            penalty_result,
        ),
    ]

    for ax, metric_col, title, ylabel, color, result in configs:
        values = test_df[metric_col].to_numpy(dtype=float)
        ax.bar(case_indices, values, color=color, alpha=0.84, width=0.86, label=ylabel)

        flagged_df = result["flagged_df"]
        method = str(result["method"])
        if method != "all_zero":
            threshold = float(result["upper"])
            threshold_label = "IQR threshold" if method == "iqr" else "Top-case cutoff"
            ax.axhline(
                threshold,
                color="#444444",
                linestyle="--",
                linewidth=1.4,
                label=f"{threshold_label}: {threshold:.2f}",
            )

        if not flagged_df.empty:
            ax.scatter(
                flagged_df["case_index"].to_numpy(dtype=float),
                flagged_df[metric_col].to_numpy(dtype=float),
                color="#7A1E1E",
                s=34,
                zorder=4,
                label="Flagged case",
            )
            annotate_flagged_cases(ax, flagged_df, metric_col)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        style_case_axis(ax, case_indices)
        ax.legend(loc="upper left")
        ax.text(
            0.01,
            0.96,
            describe_abnormal_method(result, ylabel),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            bbox={"boxstyle": "round,pad=0.30", "facecolor": "#F8F9FA", "edgecolor": "#D5D8DC"},
        )

    axes[1].set_xlabel("测试 case_index")
    fig.suptitle("TD3 测试日异常总览", fontsize=16)
    axes[1].text(
        0.99,
        0.96,
        terminal_ees_stats_text(test_df),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "#FFF8E8", "edgecolor": "#D8C690"},
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(output_dir / "test_daily_cost_penalty_anomalies.png", dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_daily_cost_breakdown(
    summary_df: pd.DataFrame,
    output_dir: Path,
    cost_result: dict[str, object],
) -> None:
    test_df = filter_test_rows(summary_df).sort_values("case_index").reset_index(drop=True)
    case_indices = test_df["case_index"].to_numpy(dtype=float)
    specs = get_existing_component_specs(test_df, COST_COMPONENT_SPECS)

    fig, ax = plt.subplots(figsize=(16.0, 7.4))
    if specs:
        bottom = np.zeros(len(test_df), dtype=float)
        for col, label, color in specs:
            values = test_df[col].to_numpy(dtype=float)
            ax.bar(case_indices, values, bottom=bottom, width=0.86, color=color, label=label)
            bottom = bottom + values
        ax.plot(
            case_indices,
            test_df["total_system_cost"].to_numpy(dtype=float),
            color="#111111",
            linewidth=1.8,
            marker="o",
            markersize=3.2,
            label="Total system cost",
        )
        flagged_df = cost_result["flagged_df"]
        if not flagged_df.empty:
            ax.scatter(
                flagged_df["case_index"].to_numpy(dtype=float),
                flagged_df["total_system_cost"].to_numpy(dtype=float),
                color="#7A1E1E",
                s=38,
                zorder=4,
                label="Flagged case",
            )
            annotate_flagged_cases(ax, flagged_df, "total_system_cost")

        ax.set_title("测试日成本分解")
        ax.set_xlabel("测试 case_index")
        ax.set_ylabel("成本")
        style_case_axis(ax, case_indices)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.00), fontsize=9.0, frameon=True)
        ax.text(
            0.01,
            0.98,
            component_share_text(test_df, "total_system_cost", specs),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F8F9FA", "edgecolor": "#D5D8DC"},
        )
    else:
        draw_empty_panel(ax, "测试日成本分解", "Missing cost component columns")

    fig.tight_layout()
    fig.savefig(output_dir / "test_daily_cost_breakdown.png", dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_daily_penalty_breakdown(
    summary_df: pd.DataFrame,
    output_dir: Path,
    penalty_result: dict[str, object],
) -> None:
    test_df = filter_test_rows(summary_df).sort_values("case_index").reset_index(drop=True)
    case_indices = test_df["case_index"].to_numpy(dtype=float)
    specs = get_existing_component_specs(test_df, PENALTY_COMPONENT_SPECS)

    fig, ax = plt.subplots(figsize=(16.0, 7.4))
    if specs:
        bottom = np.zeros(len(test_df), dtype=float)
        for col, label, color in specs:
            values = test_df[col].to_numpy(dtype=float)
            ax.bar(case_indices, values, bottom=bottom, width=0.86, color=color, label=label)
            bottom = bottom + values
        ax.plot(
            case_indices,
            test_df["total_penalties"].to_numpy(dtype=float),
            color="#111111",
            linewidth=1.8,
            marker="o",
            markersize=3.2,
            label="Total penalties",
        )
        flagged_df = penalty_result["flagged_df"]
        if not flagged_df.empty:
            ax.scatter(
                flagged_df["case_index"].to_numpy(dtype=float),
                flagged_df["total_penalties"].to_numpy(dtype=float),
                color="#7A1E1E",
                s=38,
                zorder=4,
                label="Flagged case",
            )
            annotate_flagged_cases(ax, flagged_df, "total_penalties")

        ax.set_title("测试日惩罚项分解")
        ax.set_xlabel("测试 case_index")
        ax.set_ylabel("惩罚")
        style_case_axis(ax, case_indices)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.00), fontsize=9.0, frameon=True)
        ax.text(
            0.01,
            0.98,
            component_share_text(test_df, "total_penalties", specs),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F8F9FA", "edgecolor": "#D5D8DC"},
        )
    else:
        draw_empty_panel(ax, "测试日惩罚项分解", "No non-zero penalty columns found")

    fig.tight_layout()
    fig.savefig(output_dir / "test_daily_penalty_breakdown.png", dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def build_case_report_lines(
    df: pd.DataFrame,
    metric_col: str,
    specs: list[tuple[str, str, str]],
    total_col: str,
    *,
    top_k: int = 5,
) -> list[str]:
    if df.empty:
        return ["- 无"]

    lines: list[str] = []
    for _, row in df.head(top_k).iterrows():
        label, value, share = dominant_component_from_row(row, specs, total_col)
        lines.append(
            "- "
            f"case {int(row['case_index'])} | day {int(row['day_of_year'])} | {str(row['season'])}: "
            f"{metric_col}={format_metric(row.get(metric_col))}, "
            f"主导项={label} {format_metric(value)} ({share:.1f}%)"
        )
    return lines


def write_test_analysis_report(
    summary_df: pd.DataFrame,
    output_path: Path,
    abnormal_case_path: Path,
    cost_result: dict[str, object],
    penalty_result: dict[str, object],
    abnormal_case_df: pd.DataFrame,
    *,
    typical_df: pd.DataFrame | None = None,
) -> None:
    test_df = filter_test_rows(summary_df).sort_values("case_index").reset_index(drop=True)
    cost_specs = get_existing_component_specs(test_df, COST_COMPONENT_SPECS)
    penalty_specs = get_existing_component_specs(test_df, PENALTY_COMPONENT_SPECS)

    season_stats = (
        test_df.groupby("season", sort=False)[["total_system_cost", "total_penalties"]]
        .mean(numeric_only=True)
        .reindex([season for season in SEASON_ORDER if season in test_df["season"].astype(str).unique()])
    )

    lines = [
        "# TD3 测试结果自动分析",
        "",
        f"- 测试日数量: {len(test_df)}",
        f"- 季节分布: {', '.join(f'{name}:{count}' for name, count in test_df['season'].value_counts(sort=False).to_dict().items())}",
        f"- 成本异常规则: {describe_abnormal_method(cost_result, '总成本')}",
        f"- 惩罚异常规则: {describe_abnormal_method(penalty_result, '总惩罚')}",
        f"- 异常 case 汇总表: `{abnormal_case_path}`",
        "",
        "## 成本整体",
        "",
        f"- 累计系统成本: {format_metric(test_df['total_system_cost'].sum())}",
        f"- 成本构成: {component_share_text(test_df, 'total_system_cost', cost_specs)}",
    ]

    if not season_stats.empty:
        lines.append("- 分季节平均成本 / 惩罚:")
        for season, row in season_stats.iterrows():
            lines.append(
                f"  - {season}: cost={format_metric(row['total_system_cost'])}, "
                f"penalty={format_metric(row['total_penalties'])}"
            )

    lines.extend(
        [
            "",
            "### 高成本 case",
            *build_case_report_lines(
                pd.DataFrame(cost_result["flagged_df"]),
                "total_system_cost",
                COST_COMPONENT_SPECS,
                "total_system_cost",
            ),
            "",
            "## 惩罚整体",
            "",
            f"- 累计总惩罚: {format_metric(test_df['total_penalties'].sum())}",
            f"- 惩罚构成: {component_share_text(test_df, 'total_penalties', penalty_specs)}",
            "",
            "### 高惩罚 case",
            *build_case_report_lines(
                pd.DataFrame(penalty_result["flagged_df"]),
                "total_penalties",
                PENALTY_COMPONENT_SPECS,
                "total_penalties",
            ),
        ]
    )

    if typical_df is not None and not typical_df.empty:
        lines.extend(["", "## 典型日"])
        for _, row in typical_df.iterrows():
            lines.append(
                "- "
                f"{str(row['season'])}: case {int(row['case_index'])}, day {int(row['day_of_year'])}, "
                f"total_cost={format_metric(row.get('total_system_cost'))}, "
                f"total_penalty={format_metric(row.get('total_penalties'))}"
            )

    if abnormal_case_df.empty:
        lines.extend(["", "## 异常 case 说明", "", "- 未生成异常 case 汇总表。"])
    else:
        lines.extend(["", "## 异常 case 说明", ""])
        for _, row in abnormal_case_df.iterrows():
            tags: list[str] = []
            if int(row["cost_abnormal"]) == 1:
                tags.append("高成本")
            if int(row["penalty_abnormal"]) == 1:
                tags.append("高惩罚")
            lines.append(
                "- "
                f"case {int(row['case_index'])} ({str(row['season'])}, day {int(row['day_of_year'])}) "
                f"[{'/'.join(tags)}]: 成本主导项={row['dominant_cost_component']} "
                f"{format_metric(row['dominant_cost_value'])} ({float(row['dominant_cost_share_pct']):.1f}%), "
                f"惩罚主导项={row['dominant_penalty_component']} "
                f"{format_metric(row['dominant_penalty_value'])} ({float(row['dominant_penalty_share_pct']):.1f}%)"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary_overview(summary_df: pd.DataFrame, output_dir: Path) -> None:
    test_df = filter_test_rows(summary_df).sort_values("case_index").reset_index(drop=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = test_df["case_index"].to_numpy(dtype=float)

    ax = axes[0, 0]
    ax.bar(
        x,
        test_df["total_system_cost"].to_numpy(dtype=float),
        color=COLOR_MAP["Cost"],
        alpha=0.85,
        label="System cost",
    )
    ax.set_title("Daily system cost and penalties")
    ax.set_xlabel("Test case index")
    ax.set_ylabel("System cost")
    ax.grid(True, alpha=0.25, axis="y")
    ax2 = ax.twinx()
    ax2.plot(
        x,
        test_df["total_penalties"].to_numpy(dtype=float),
        color=COLOR_MAP["Penalty"],
        marker="o",
        linewidth=1.5,
        label="Penalties",
    )
    ax2.set_ylabel("Penalties")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    ax = axes[0, 1]
    if {"total_grid_buy_kwh", "total_grid_sell_kwh"}.issubset(test_df.columns):
        ax.bar(x, test_df["total_grid_buy_kwh"].to_numpy(dtype=float), color="#7BA6D8", label="Grid buy")
        ax.plot(
            x,
            test_df["total_grid_sell_kwh"].to_numpy(dtype=float),
            color="#355C7D",
            marker="s",
            linewidth=1.5,
            label="Grid sell",
        )
        ax.set_title("Daily grid exchange")
        ax.set_xlabel("Test case index")
        ax.set_ylabel("Energy (kWh)")
        ax.grid(True, alpha=0.25)
        ax.legend()
    else:
        draw_empty_panel(ax, "Daily grid exchange", "Missing total_grid_buy_kwh / total_grid_sell_kwh")

    grouped = test_df.groupby("season", sort=False).mean(numeric_only=True)
    grouped = grouped.reindex([season for season in SEASON_ORDER if season in grouped.index])

    ax = axes[1, 0]
    if not grouped.empty:
        ax.bar(grouped.index, grouped["total_system_cost"], color=COLOR_MAP["Cost"])
        ax.set_title("Seasonal mean system cost")
        ax.set_xlabel("Season")
        ax.set_ylabel("System cost")
        ax.grid(True, alpha=0.25, axis="y")
    else:
        draw_empty_panel(ax, "Seasonal mean system cost")

    ax = axes[1, 1]
    if not grouped.empty and {"total_storage_peak_shaved_kwh", "total_storage_charge_rewarded_kwh"}.issubset(grouped.columns):
        width = 0.35
        pos = np.arange(len(grouped.index))
        ax.bar(
            pos - width / 2,
            grouped["total_storage_peak_shaved_kwh"].to_numpy(dtype=float),
            width=width,
            color="#F29A92",
            label="Storage peak shaved",
        )
        ax.bar(
            pos + width / 2,
            grouped["total_storage_charge_rewarded_kwh"].to_numpy(dtype=float),
            width=width,
            color="#78B9E7",
            label="Storage charge rewarded",
        )
        ax.set_xticks(pos)
        ax.set_xticklabels(grouped.index)
        ax.set_title("Seasonal storage support metrics")
        ax.set_xlabel("Season")
        ax.set_ylabel("Energy (kWh)")
        ax.grid(True, alpha=0.25, axis="y")
        ax.legend()
    elif not grouped.empty:
        ax.bar(grouped.index, grouped["total_penalties"], color=COLOR_MAP["Penalty"])
        ax.set_title("Seasonal mean penalties")
        ax.set_xlabel("Season")
        ax.set_ylabel("Penalties")
        ax.grid(True, alpha=0.25, axis="y")
    else:
        draw_empty_panel(ax, "Seasonal EV support metrics")

    fig.suptitle("TD3 test summary overview", fontsize=15)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(output_dir / "test_summary_overview.png", dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_test_diagnostics(summary_df: pd.DataFrame, output_dir: Path) -> None:
    test_df = filter_test_rows(summary_df).sort_values("case_index").reset_index(drop=True)
    x = test_df["case_index"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    penalty_cols = [col for col in test_df.columns if col.startswith("total_penalty_")]
    totals = (
        test_df[penalty_cols].sum(numeric_only=True).sort_values(ascending=False)
        if penalty_cols
        else pd.Series(dtype=float)
    )
    totals = totals[totals.abs() > 1e-9].head(8)
    if not totals.empty:
        labels = [label.replace("total_penalty_", "") for label in totals.index]
        ax.barh(np.arange(len(totals)), totals.to_numpy(dtype=float), color=COLOR_MAP["Penalty"])
        ax.set_yticks(np.arange(len(totals)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title("Penalty breakdown")
        ax.set_xlabel("Penalty total")
        ax.grid(True, alpha=0.25, axis="x")
    else:
        draw_empty_panel(ax, "Penalty breakdown", "No penalty columns found")

    ax = axes[0, 1]
    if {"ees_soc_episode_init", "final_ees_soc"}.issubset(test_df.columns):
        init_soc = test_df["ees_soc_episode_init"].to_numpy(dtype=float)
        final_soc = test_df["final_ees_soc"].to_numpy(dtype=float)
        ax.plot(x, init_soc, color="#7F7F7F", linewidth=1.4, label="Initial SOC")
        ax.plot(x, final_soc, color=COLOR_MAP["Penalty"], marker="o", linewidth=1.4, label="Final SOC")
        ax.set_title("EES initial vs final SOC")
        ax.set_xlabel("Test case index")
        ax.set_ylabel("SOC")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend()
    else:
        draw_empty_panel(ax, "EES SOC", "Missing ees_soc_episode_init / final_ees_soc")

    ax = axes[1, 0]
    gt_diag_cols = [
        ("total_gt_export_clip_steps", "Export clip steps", "#355C7D"),
        ("total_gt_safe_infeasible_steps", "GT safe infeasible", "#D62728"),
    ]
    plotted = False
    for col, label, color in gt_diag_cols:
        if col not in test_df.columns:
            continue
        ax.plot(x, test_df[col].to_numpy(dtype=float), marker="o", linewidth=1.4, color=color, label=label)
        plotted = True
    if plotted:
        ax.set_title("GT feasibility diagnostics")
        ax.set_xlabel("Test case index")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)
        ax.legend()
    else:
        draw_empty_panel(ax, "GT feasibility diagnostics", "Missing GT diagnostic columns")

    ax = axes[1, 1]
    support_cols = [
        ("total_storage_peak_shaved_kwh", "Storage peak shaved", "#F29A92"),
        ("total_storage_charge_rewarded_kwh", "Storage charge rewarded", "#78B9E7"),
        ("total_ees_charge_rewarded_kwh", "EES charge rewarded", "#9467BD"),
        ("total_low_value_charge_kwh", "Low-value charge", "#2CA02C"),
    ]
    plotted = False
    for col, label, color in support_cols:
        if col not in test_df.columns:
            continue
        ax.plot(x, test_df[col].to_numpy(dtype=float), marker="o", linewidth=1.4, color=color, label=label)
        plotted = True
    if plotted:
        ax.set_title("Storage and EV diagnostics")
        ax.set_xlabel("Test case index")
        ax.set_ylabel("Energy (kWh)")
        ax.grid(True, alpha=0.25)
        ax.legend()
    else:
        draw_empty_panel(ax, "Storage and EV diagnostics", "Missing storage / EV columns")

    fig.suptitle("TD3 test diagnostics", fontsize=15)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(output_dir / "test_diagnostics.png", dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def build_dispatch_title(case_index: int, day_of_year: int, month: int, season: str | None) -> tuple[str, str]:
    season_text = str(season).lower() if season is not None else ""
    prefix_season = f"{season_text}_" if season_text else ""
    prefix = f"{prefix_season}case_{case_index:02d}_day_{day_of_year:03d}_month_{month:02d}"
    if season_text:
        title = f"TD3 {season_text.capitalize()} case {case_index:02d} | day {day_of_year} | month {month:02d}"
    else:
        title = f"TD3 case {case_index:02d} | day {day_of_year} | month {month:02d}"
    return prefix, title


def plot_dispatch_dashboard_png(
    case_df: pd.DataFrame,
    source_df: pd.DataFrame,
    save_path: Path,
    title: str,
    summary_row: pd.Series | None,
) -> None:
    hours = np.arange(1, 25)
    fig, axes = plt.subplots(4, 1, figsize=(17.5, 18.5), sharex=True)

    pv = source_df["PV"].to_numpy(dtype=float)
    wt = source_df["Wind"].to_numpy(dtype=float)
    electric_load = source_df["Electric Load"].to_numpy(dtype=float)
    heat_load = source_df["Heat Load"].to_numpy(dtype=float)
    cold_load = source_df["Cold Load"].to_numpy(dtype=float)

    p_gt = case_df["p_gt"].to_numpy(dtype=float)
    p_grid_buy = case_df["p_grid_buy"].to_numpy(dtype=float)
    p_grid_sell = case_df["p_grid_sell"].to_numpy(dtype=float)
    p_ees_ch = case_df["p_ees_ch"].to_numpy(dtype=float)
    p_ees_dis = case_df["p_ees_dis"].to_numpy(dtype=float)
    p_ev_ch = case_df["p_ev_ch"].to_numpy(dtype=float)
    p_ev_dis = case_df["p_ev_dis"].to_numpy(dtype=float)
    p_ec_elec_in = case_df["p_ec_elec_in"].to_numpy(dtype=float)
    p_whb_heat = case_df["p_whb_heat"].to_numpy(dtype=float)
    p_gb_heat = case_df["p_gb_heat"].to_numpy(dtype=float)
    p_ac_cool = case_df["p_ac_cool"].to_numpy(dtype=float)
    p_ec_cool = case_df["p_ec_cool"].to_numpy(dtype=float)
    ees_soc = case_df["ees_soc"].to_numpy(dtype=float)
    ac_heat_input = p_ac_cool / max(AC_COP, 1e-6)

    ax = axes[0]
    stacked_bar(
        ax,
        hours,
        [
            ("PV", pv),
            ("WT", wt),
            ("GT", p_gt),
            ("Grid buy", p_grid_buy),
            ("EES discharge", p_ees_dis),
            ("EV discharge", p_ev_dis),
        ],
        positive=True,
    )
    stacked_bar(
        ax,
        hours,
        [
            ("Grid sell", p_grid_sell),
            ("EC electric input", p_ec_elec_in),
            ("EES charge", p_ees_ch),
            ("EV charge", p_ev_ch),
        ],
        positive=False,
    )
    ax.plot(hours, electric_load, color=COLOR_MAP["Load"], linewidth=2.0, marker="o", markersize=3.5, label="Electric load")
    ax.set_title("Electric dispatch balance")
    style_balance_axis(ax, "Power (kW)", show_xlabel=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.00), fontsize=8.5, frameon=True)

    ax = axes[1]
    stacked_bar(ax, hours, [("WHB heat", p_whb_heat), ("GB heat", p_gb_heat)], positive=True)
    stacked_bar(ax, hours, [("AC heat input", ac_heat_input)], positive=False)
    ax.plot(hours, heat_load, color=COLOR_MAP["Load"], linewidth=2.0, marker="o", markersize=3.5, label="Heat load")
    ax.set_title("Heat dispatch balance")
    style_balance_axis(ax, "Power (kW)", show_xlabel=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.00), fontsize=8.5, frameon=True)

    ax = axes[2]
    stacked_bar(ax, hours, [("AC cooling", p_ac_cool), ("EC cooling", p_ec_cool)], positive=True)
    ax.plot(hours, cold_load, color=COLOR_MAP["Load"], linewidth=2.0, marker="o", markersize=3.5, label="Cooling load")
    ax.set_title("Cooling dispatch balance")
    style_balance_axis(ax, "Power (kW)", show_xlabel=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.00), fontsize=8.5, frameon=True)

    ax = axes[3]
    ax.bar(hours, p_ees_dis, width=BAR_WIDTH, color=COLOR_MAP["EES discharge"], label="EES discharge")
    ax.bar(hours, -p_ees_ch, width=BAR_WIDTH, color=COLOR_MAP["EES charge"], label="EES charge")
    ax.bar(hours, p_ev_dis, width=BAR_WIDTH * 0.62, color=COLOR_MAP["EV discharge"], label="EV discharge")
    ax.bar(hours, -p_ev_ch, width=BAR_WIDTH * 0.62, color=COLOR_MAP["EV charge"], label="EV charge")
    ax.set_title("Storage and EV schedule")
    style_balance_axis(ax, "Power (kW)", show_xlabel=True)
    ax2 = ax.twinx()
    ax2.plot(hours, ees_soc, color=COLOR_MAP["SOC"], linewidth=2.0, linestyle="--", marker="s", markersize=3.5, label="EES SOC")
    ax2.set_ylabel("SOC")
    ax2.set_ylim(0.0, 1.0)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", bbox_to_anchor=(1.01, 1.00), fontsize=8.5, frameon=True)

    fig.suptitle(title, fontsize=16, y=0.995)
    fig.text(
        0.995,
        0.988,
        summary_text_box(summary_row),
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F8F9FA", "edgecolor": "#D5D8DC"},
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.84, 0.97])
    fig.savefig(save_path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def build_hourly_table(case_df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Hour": np.arange(1, 25, dtype=int),
            "Electric Load (kW)": source_df["Electric Load"].to_numpy(dtype=float),
            "Heat Load (kW)": source_df["Heat Load"].to_numpy(dtype=float),
            "Cold Load (kW)": source_df["Cold Load"].to_numpy(dtype=float),
            "PV (kW)": source_df["PV"].to_numpy(dtype=float),
            "Wind (kW)": source_df["Wind"].to_numpy(dtype=float),
            "GT (kW)": case_df["p_gt"].to_numpy(dtype=float),
            "Grid buy (kW)": case_df["p_grid_buy"].to_numpy(dtype=float),
            "Grid sell (kW)": case_df["p_grid_sell"].to_numpy(dtype=float),
            "EES charge (kW)": case_df["p_ees_ch"].to_numpy(dtype=float),
            "EES discharge (kW)": case_df["p_ees_dis"].to_numpy(dtype=float),
            "EV charge (kW)": case_df["p_ev_ch"].to_numpy(dtype=float),
            "EV discharge (kW)": case_df["p_ev_dis"].to_numpy(dtype=float),
            "WHB heat (kW)": case_df["p_whb_heat"].to_numpy(dtype=float),
            "GB heat (kW)": case_df["p_gb_heat"].to_numpy(dtype=float),
            "AC cooling (kW)": case_df["p_ac_cool"].to_numpy(dtype=float),
            "EC cooling (kW)": case_df["p_ec_cool"].to_numpy(dtype=float),
            "EC electric input (kW)": case_df["p_ec_elec_in"].to_numpy(dtype=float),
            "EES SOC": case_df["ees_soc"].to_numpy(dtype=float),
        }
    )


def plotly_bar_trace(name: str, hours: list[int], values: np.ndarray, color: str, *, negative: bool = False) -> dict:
    values = np.asarray(values, dtype=float)
    signed_values = -values if negative else values
    return {
        "type": "bar",
        "name": name,
        "x": hours,
        "y": signed_values.tolist(),
        "marker": {"color": color},
        "customdata": values.tolist(),
        "hovertemplate": f"Hour %{{x}}<br>{name}: %{{customdata:.2f}} kW<extra></extra>",
    }


def plotly_line_trace(
    name: str,
    hours: list[int],
    values: np.ndarray,
    color: str,
    *,
    yaxis: str = "y",
    unit: str = "kW",
) -> dict:
    values = np.asarray(values, dtype=float)
    unit_text = f" {unit}" if unit else ""
    return {
        "type": "scatter",
        "mode": "lines+markers",
        "name": name,
        "x": hours,
        "y": values.tolist(),
        "line": {"color": color, "width": 2.0},
        "marker": {"size": 6},
        "yaxis": yaxis,
        "hovertemplate": f"Hour %{{x}}<br>{name}: %{{y:.2f}}{unit_text}<extra></extra>",
    }


def build_metric_cards(summary_row: pd.Series | None) -> str:
    if summary_row is None:
        return (
            "<div class='metric-card'>"
            "<div class='metric-label'>Summary</div>"
            "<div class='metric-value'>No summary row found</div>"
            "</div>"
        )

    items = [
        ("Total system cost", format_metric(summary_row.get("total_system_cost"))),
        ("Total penalties", format_metric(summary_row.get("total_penalties"))),
        ("Guide reward", format_metric(summary_row.get("total_guide_reward"))),
        ("Grid buy (kWh)", format_metric(summary_row.get("total_grid_buy_kwh"))),
        ("Grid sell (kWh)", format_metric(summary_row.get("total_grid_sell_kwh"))),
        ("Final EES SOC", format_metric(summary_row.get("final_ees_soc"), 4)),
        ("Required EES SOC", format_metric(summary_row.get("terminal_ees_required_soc"), 4)),
        ("EES terminal shortage (kWh)", format_metric(summary_row.get("terminal_ees_shortage_kwh"), 2)),
        ("EES terminal penalty", format_metric(summary_row.get("total_penalty_terminal_ees_soc"), 2)),
        ("EES terminal feasible", format_bool_metric(summary_row.get("ees_terminal_soc_feasible"))),
    ]
    parts: list[str] = []
    for label, value in items:
        parts.append(
            "<div class='metric-card'>"
            f"<div class='metric-label'>{html.escape(label)}</div>"
            f"<div class='metric-value'>{html.escape(value)}</div>"
            "</div>"
        )
    return "".join(parts)


def render_interactive_dispatch_html(
    case_df: pd.DataFrame,
    source_df: pd.DataFrame,
    save_path: Path,
    title: str,
    summary_row: pd.Series | None,
) -> None:
    hours = list(range(1, 25))
    ac_heat_input = case_df["p_ac_cool"].to_numpy(dtype=float) / max(AC_COP, 1e-6)

    electric_traces = [
        plotly_bar_trace("PV", hours, source_df["PV"].to_numpy(dtype=float), COLOR_MAP["PV"]),
        plotly_bar_trace("Wind", hours, source_df["Wind"].to_numpy(dtype=float), COLOR_MAP["WT"]),
        plotly_bar_trace("GT", hours, case_df["p_gt"].to_numpy(dtype=float), COLOR_MAP["GT"]),
        plotly_bar_trace("Grid buy", hours, case_df["p_grid_buy"].to_numpy(dtype=float), COLOR_MAP["Grid buy"]),
        plotly_bar_trace("EES discharge", hours, case_df["p_ees_dis"].to_numpy(dtype=float), COLOR_MAP["EES discharge"]),
        plotly_bar_trace("EV discharge", hours, case_df["p_ev_dis"].to_numpy(dtype=float), COLOR_MAP["EV discharge"]),
        plotly_bar_trace("Grid sell", hours, case_df["p_grid_sell"].to_numpy(dtype=float), COLOR_MAP["Grid sell"], negative=True),
        plotly_bar_trace("EC electric input", hours, case_df["p_ec_elec_in"].to_numpy(dtype=float), COLOR_MAP["EC electric input"], negative=True),
        plotly_bar_trace("EES charge", hours, case_df["p_ees_ch"].to_numpy(dtype=float), COLOR_MAP["EES charge"], negative=True),
        plotly_bar_trace("EV charge", hours, case_df["p_ev_ch"].to_numpy(dtype=float), COLOR_MAP["EV charge"], negative=True),
        plotly_line_trace("Electric load", hours, source_df["Electric Load"].to_numpy(dtype=float), COLOR_MAP["Load"]),
    ]
    electric_layout = {
        "title": "Electric dispatch balance",
        "barmode": "relative",
        "hovermode": "x unified",
        "xaxis": {"title": "Hour", "tickmode": "array", "tickvals": hours},
        "yaxis": {"title": "Power (kW)", "zeroline": True},
        "legend": {"orientation": "h", "y": 1.14},
        "margin": {"l": 70, "r": 30, "t": 70, "b": 55},
    }

    heat_traces = [
        plotly_bar_trace("WHB heat", hours, case_df["p_whb_heat"].to_numpy(dtype=float), COLOR_MAP["WHB heat"]),
        plotly_bar_trace("GB heat", hours, case_df["p_gb_heat"].to_numpy(dtype=float), COLOR_MAP["GB heat"]),
        plotly_bar_trace("AC heat input", hours, ac_heat_input, COLOR_MAP["AC heat input"], negative=True),
        plotly_line_trace("Heat load", hours, source_df["Heat Load"].to_numpy(dtype=float), COLOR_MAP["Load"]),
    ]
    heat_layout = {
        "title": "Heat dispatch balance",
        "barmode": "relative",
        "hovermode": "x unified",
        "xaxis": {"title": "Hour", "tickmode": "array", "tickvals": hours},
        "yaxis": {"title": "Power (kW)", "zeroline": True},
        "legend": {"orientation": "h", "y": 1.14},
        "margin": {"l": 70, "r": 30, "t": 70, "b": 55},
    }

    cooling_traces = [
        plotly_bar_trace("AC cooling", hours, case_df["p_ac_cool"].to_numpy(dtype=float), COLOR_MAP["AC cooling"]),
        plotly_bar_trace("EC cooling", hours, case_df["p_ec_cool"].to_numpy(dtype=float), COLOR_MAP["EC cooling"]),
        plotly_line_trace("Cooling load", hours, source_df["Cold Load"].to_numpy(dtype=float), COLOR_MAP["Load"]),
    ]
    cooling_layout = {
        "title": "Cooling dispatch balance",
        "barmode": "relative",
        "hovermode": "x unified",
        "xaxis": {"title": "Hour", "tickmode": "array", "tickvals": hours},
        "yaxis": {"title": "Power (kW)", "zeroline": True},
        "legend": {"orientation": "h", "y": 1.14},
        "margin": {"l": 70, "r": 30, "t": 70, "b": 55},
    }

    storage_traces = [
        plotly_bar_trace("EES discharge", hours, case_df["p_ees_dis"].to_numpy(dtype=float), COLOR_MAP["EES discharge"]),
        plotly_bar_trace("EES charge", hours, case_df["p_ees_ch"].to_numpy(dtype=float), COLOR_MAP["EES charge"], negative=True),
        plotly_bar_trace("EV discharge", hours, case_df["p_ev_dis"].to_numpy(dtype=float), COLOR_MAP["EV discharge"]),
        plotly_bar_trace("EV charge", hours, case_df["p_ev_ch"].to_numpy(dtype=float), COLOR_MAP["EV charge"], negative=True),
        plotly_line_trace("EES SOC", hours, case_df["ees_soc"].to_numpy(dtype=float), COLOR_MAP["SOC"], yaxis="y2", unit=""),
    ]
    storage_layout = {
        "title": "Storage and EV schedule",
        "barmode": "relative",
        "hovermode": "x unified",
        "xaxis": {"title": "Hour", "tickmode": "array", "tickvals": hours},
        "yaxis": {"title": "Power (kW)", "zeroline": True},
        "yaxis2": {"title": "SOC", "overlaying": "y", "side": "right", "range": [0.0, 1.0]},
        "legend": {"orientation": "h", "y": 1.14},
        "margin": {"l": 70, "r": 70, "t": 70, "b": 55},
    }

    hourly_table = build_hourly_table(case_df, source_df)
    table_html = hourly_table.to_html(
        index=False,
        justify="center",
        border=0,
        classes=["dispatch-table"],
        float_format=lambda value: f"{float(value):.2f}",
    )

    template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>__TITLE__</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {
      margin: 0;
      font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
      background: #F4F6F8;
      color: #20262E;
    }
    .page {
      max-width: 1480px;
      margin: 0 auto;
      padding: 24px 28px 40px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.25;
    }
    .subtitle {
      margin: 0 0 18px;
      color: #5F6B7A;
      font-size: 14px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .metric-card {
      background: #FFFFFF;
      border-radius: 12px;
      padding: 14px 16px;
      box-shadow: 0 4px 16px rgba(26, 39, 52, 0.08);
    }
    .metric-label {
      font-size: 12px;
      color: #718096;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .metric-value {
      font-size: 20px;
      font-weight: 700;
      color: #17212B;
    }
    .note {
      background: #FFF7E6;
      border: 1px solid #F4D19B;
      border-radius: 10px;
      padding: 12px 14px;
      margin-bottom: 20px;
      font-size: 13px;
      color: #7A5A00;
    }
    .chart-card {
      background: #FFFFFF;
      border-radius: 14px;
      padding: 12px 14px 4px;
      margin-bottom: 18px;
      box-shadow: 0 4px 16px rgba(26, 39, 52, 0.08);
    }
    .table-card {
      background: #FFFFFF;
      border-radius: 14px;
      padding: 16px 18px;
      box-shadow: 0 4px 16px rgba(26, 39, 52, 0.08);
    }
    .table-title {
      margin: 0 0 12px;
      font-size: 18px;
    }
    table.dispatch-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    table.dispatch-table thead th {
      position: sticky;
      top: 0;
      background: #EDF2F7;
      z-index: 1;
    }
    table.dispatch-table th,
    table.dispatch-table td {
      border: 1px solid #E2E8F0;
      padding: 6px 8px;
      text-align: center;
      white-space: nowrap;
    }
    table.dispatch-table tbody tr:nth-child(odd) {
      background: #FAFBFC;
    }
  </style>
</head>
<body>
  <div class="page">
    <h1>__TITLE__</h1>
    <p class="subtitle">Interactive dispatch dashboard. Hover on bars or lines to inspect exact values by hour.</p>
    <div class="metrics">__METRIC_CARDS__</div>
    <div class="note">
      This HTML uses Plotly via CDN for hover interaction. If the charts do not render in your browser,
      rerun this script with <code>--dispatch-format png</code> to export a static figure instead.
    </div>
    <div class="chart-card"><div id="electric-chart" style="height: 430px;"></div></div>
    <div class="chart-card"><div id="heat-chart" style="height: 390px;"></div></div>
    <div class="chart-card"><div id="cooling-chart" style="height: 390px;"></div></div>
    <div class="chart-card"><div id="storage-chart" style="height: 420px;"></div></div>
    <div class="table-card">
      <h2 class="table-title">Hourly dispatch table</h2>
      __TABLE_HTML__
    </div>
  </div>
  <script>
    const config = {responsive: true, displaylogo: false};
    Plotly.newPlot("electric-chart", __ELECTRIC_TRACES__, __ELECTRIC_LAYOUT__, config);
    Plotly.newPlot("heat-chart", __HEAT_TRACES__, __HEAT_LAYOUT__, config);
    Plotly.newPlot("cooling-chart", __COOLING_TRACES__, __COOLING_LAYOUT__, config);
    Plotly.newPlot("storage-chart", __STORAGE_TRACES__, __STORAGE_LAYOUT__, config);
  </script>
</body>
</html>
"""

    html_text = (
        template.replace("__TITLE__", html.escape(title))
        .replace("__METRIC_CARDS__", build_metric_cards(summary_row))
        .replace("__TABLE_HTML__", table_html)
        .replace("__ELECTRIC_TRACES__", json.dumps(electric_traces, ensure_ascii=False))
        .replace("__ELECTRIC_LAYOUT__", json.dumps(electric_layout, ensure_ascii=False))
        .replace("__HEAT_TRACES__", json.dumps(heat_traces, ensure_ascii=False))
        .replace("__HEAT_LAYOUT__", json.dumps(heat_layout, ensure_ascii=False))
        .replace("__COOLING_TRACES__", json.dumps(cooling_traces, ensure_ascii=False))
        .replace("__COOLING_LAYOUT__", json.dumps(cooling_layout, ensure_ascii=False))
        .replace("__STORAGE_TRACES__", json.dumps(storage_traces, ensure_ascii=False))
        .replace("__STORAGE_LAYOUT__", json.dumps(storage_layout, ensure_ascii=False))
    )
    save_path.write_text(html_text, encoding="utf-8")


def plot_one_case(
    summary_df: pd.DataFrame,
    timeseries_df: pd.DataFrame,
    yearly_csv_path: Path,
    output_dir: Path,
    dispatch_format: str,
    *,
    case_index: int,
    day_of_year: int,
    month: int,
    season: str | None = None,
) -> None:
    case_df = get_case_timeseries(timeseries_df, case_index=case_index)
    source_df = load_source_day(yearly_csv_path, day_of_year)
    summary_row = get_case_summary(summary_df, case_index)
    actual_season = season or str(case_df["season"].iloc[0])
    prefix, title = build_dispatch_title(case_index, day_of_year, month, actual_season)

    if dispatch_format == "png":
        plot_dispatch_dashboard_png(
            case_df,
            source_df,
            output_dir / f"{prefix}_dispatch_dashboard.png",
            title,
            summary_row,
        )
    else:
        render_interactive_dispatch_html(
            case_df,
            source_df,
            output_dir / f"{prefix}_dispatch_dashboard.html",
            title,
            summary_row,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot compact TD3 test figures aligned with test_TD3.py outputs."
    )
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV_PATH)
    parser.add_argument("--timeseries-csv", type=Path, default=DEFAULT_TIMESERIES_CSV_PATH)
    parser.add_argument("--yearly-csv", type=Path, default=DEFAULT_YEARLY_CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--typical-summary", type=Path, default=DEFAULT_TYPICAL_SUMMARY_PATH)
    parser.add_argument("--case-analysis-csv", type=Path, default=DEFAULT_CASE_ANALYSIS_PATH)
    parser.add_argument("--analysis-report", type=Path, default=DEFAULT_ANALYSIS_REPORT_PATH)
    parser.add_argument("--case-index", type=int, default=None)
    parser.add_argument("--day-of-year", type=int, default=None)
    parser.add_argument("--dispatch-format", choices=["html", "png"], default="png")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--skip-balance", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.case_index is not None and args.day_of_year is not None:
        raise ValueError("--case-index and --day-of-year cannot be used together.")

    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.case_analysis_csv.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_report.parent.mkdir(parents=True, exist_ok=True)

    summary_df = load_daily_summary(args.summary_csv)
    timeseries_df = load_timeseries(args.timeseries_csv)
    typical_df: pd.DataFrame | None = None

    if not args.skip_balance and args.case_index is None and args.day_of_year is None:
        typical_df = select_typical_days(summary_df, args.typical_summary)

    if not args.skip_summary:
        test_df = filter_test_rows(summary_df).sort_values("case_index").reset_index(drop=True)
        cost_result = detect_abnormal_cases(test_df, "total_system_cost", fallback_top_k=5)
        penalty_result = detect_abnormal_cases(test_df, "total_penalties", fallback_top_k=5)
        abnormal_case_df = build_abnormal_case_summary(summary_df, cost_result, penalty_result)
        abnormal_case_df.to_csv(args.case_analysis_csv, index=False, encoding="utf-8-sig")

        plot_daily_cost_penalty_anomalies(summary_df, args.output_dir, cost_result, penalty_result)
        plot_daily_cost_breakdown(summary_df, args.output_dir, cost_result)
        plot_daily_penalty_breakdown(summary_df, args.output_dir, penalty_result)
        write_test_analysis_report(
            summary_df,
            args.analysis_report,
            args.case_analysis_csv,
            cost_result,
            penalty_result,
            abnormal_case_df,
            typical_df=typical_df,
        )

    if not args.skip_balance:
        if args.case_index is not None:
            case_df = get_case_timeseries(timeseries_df, case_index=args.case_index)
            plot_one_case(
                summary_df,
                timeseries_df,
                args.yearly_csv,
                args.output_dir,
                args.dispatch_format,
                case_index=int(case_df["case_index"].iloc[0]),
                day_of_year=int(case_df["day_of_year"].iloc[0]),
                month=int(case_df["month"].iloc[0]),
                season=str(case_df["season"].iloc[0]),
            )
        elif args.day_of_year is not None:
            case_df = get_case_timeseries(timeseries_df, day_of_year=args.day_of_year)
            plot_one_case(
                summary_df,
                timeseries_df,
                args.yearly_csv,
                args.output_dir,
                args.dispatch_format,
                case_index=int(case_df["case_index"].iloc[0]),
                day_of_year=int(case_df["day_of_year"].iloc[0]),
                month=int(case_df["month"].iloc[0]),
                season=str(case_df["season"].iloc[0]),
            )
        else:
            if typical_df is None:
                typical_df = select_typical_days(summary_df, args.typical_summary)
            for _, row in typical_df.iterrows():
                plot_one_case(
                    summary_df,
                    timeseries_df,
                    args.yearly_csv,
                    args.output_dir,
                    args.dispatch_format,
                    case_index=int(row["case_index"]),
                    day_of_year=int(row["day_of_year"]),
                    month=int(row["month"]),
                    season=str(row["season"]).lower(),
                )
            print(f"Typical-day summary saved to: {args.typical_summary}")

    print(f"TD3 test plots saved to: {args.output_dir}")
    if not args.skip_summary:
        print(f"Abnormal-case summary saved to: {args.case_analysis_csv}")
        print(f"Analysis report saved to: {args.analysis_report}")
    print(f"Dispatch format: {args.dispatch_format}")


if __name__ == "__main__":
    main()
