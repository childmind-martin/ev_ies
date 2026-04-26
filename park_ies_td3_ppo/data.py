"""年度源荷数据生成脚本（论文近似强度版）。

设计目标：
1. 仅基于春、夏、秋、冬四个典型日生成全年数据。
2. 数据波动强度接近论文中的“中等不确定性”风格，而不是重尾极端扰动。
3. 保留足够场景多样性以训练泛化能力较强的 RL 智能体，同时避免过强扰动导致训练不收敛。
4. 输出 CSV、NPY、测试集掩码，以及全年/分季节统计图。

相对原始版本的主要修改：
- 删除 rare shock 式极端天气：不再对 PV/风电/负荷加入大幅度小概率突变。
- 负荷改为“公共日因子 + 通道个体因子 + 小时相关噪声”。
- PV 改为“天气状态驱动”（clear / normal / cloudy），更接近日内太阳辐照波动机理。
- 风电改为“风况状态驱动”（weak / normal / windy），但强度保持温和。
- 保留按月留出最后 5 天作为测试集，便于与论文式训练/验证组织方式对齐。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta


# ============================================================
# 0) 四季典型日数据录入：每个季节包含 24 个小时点的源荷曲线
# ============================================================

def make_typical_df(elec, heat, cold, pv, wind):
    """将五类 24 小时序列整理为统一的典型日 DataFrame。"""
    if not (len(elec) == len(heat) == len(cold) == len(pv) == len(wind) == 24):
        raise ValueError("典型日每个序列必须是 24 个点。")
    return pd.DataFrame({
        "Hour": list(range(0, 24)),
        "Electric Load": np.array(elec, dtype=float),
        "Heat Load": np.array(heat, dtype=float),
        "Cold Load": np.array(cold, dtype=float),
        "PV": np.array(pv, dtype=float),
        "Wind": np.array(wind, dtype=float),
    })


spring_data = make_typical_df(
    elec=[14396,14159,13760,13603,13650,14011,14819,14912,15050,15200,15189,15082,15133,14903,14877,14689,14463,14597,14891,15441,15444,14823,14507,14274],
    heat=[7213,7339,7585,7852,8131,8501,8913,8821,8386,7820,7235,6969,6889,6890,6888,6889,6886,6882,6960,7124,7250,7273,7213,7198],
    cold=[9595,9091,8698,8637,8639,8985,9826,10544,11900,13283,14434,15402,16194,16910,17538,17837,17840,17476,16697,15495,14217,12953,11582,10692],
    pv=[0,0,0,0,0,0,189,854,3184,6623,9658,11615,12692,12282,10137,7902,5242,2349,677,0,0,0,0,0],
    wind=[6180,6172,6108,6220,6287,6059,6099,5870,5421,5054,5452,6060,6615,6843,7013,7168,6809,6760,6480,6194,6276,6299,6365,6332]
)

summer_data = make_typical_df(
    elec=[17394,16947,16636,16494,16574,16940,17134,17137,17429,17664,17809,17752,17853,17855,17682,17641,17847,17861,17847,17835,17673,17131,17043,17240],
    heat=[5449,5452,5443,5529,5545,5598,5626,5511,5474,5422,5395,5397,5398,5399,5398,5399,5396,5400,5396,5392,5392,5393,5394,5424],
    cold=[18479,18303,18323,18319,18586,20860,23609,25080,26315,27419,28199,28773,29576,30183,30576,30880,30958,30042,28685,27118,24917,23399,21666,19786],
    pv=[0,0,0,0,0,180,1598,4294,7405,10221,12525,14159,15373,14768,13672,12159,9503,6823,4061,1486,0,0,0,0],
    wind=[4238,3849,4009,4223,3929,3834,3499,2826,2736,2894,3384,3944,4415,4540,4499,4647,4694,4369,4099,3900,3463,4230,3712,4110]
)

autumn_data = make_typical_df(
    elec=[12024,11754,11449,11219,11131,11535,12557,13044,13309,13325,13332,13337,13336,13080,13085,12874,12702,12637,12704,13027,13332,13134,12716,12157],
    heat=[6964,7064,7148,7313,7428,7842,8043,8173,7975,7458,6891,6645,6611,6613,6612,6613,6610,6607,6624,6684,6785,6851,6843,6932],
    cold=[12172,11782,11463,11402,11422,12453,13476,14004,15238,17149,18918,20240,21326,22161,22817,23216,23001,21882,19960,18398,17033,15721,14241,12916],
    pv=[0,0,0,0,0,0,387,1768,4514,7370,9678,11173,11856,11869,10208,8444,6288,3715,1483,0,0,0,0,0],
    wind=[5058,4866,4915,5084,5009,4835,4670,4135,3885,3837,4284,4837,5333,5489,5554,5708,5541,5387,5105,4905,4733,5109,4908,5066]
)

winter_data = make_typical_df(
    elec=[14773,14483,14137,14008,14152,14775,15606,15831,15913,15917,15929,15937,15944,15713,15753,15758,15930,15946,15939,15934,15936,15248,14967,14832],
    heat=[11518,11739,12123,12532,13196,14079,14735,15171,14904,13898,12548,11128,10146,9764,9763,9764,9760,9765,9999,10415,10849,11237,11310,11340],
    cold=[5039,4826,4820,4818,4818,5325,5732,5735,5979,6510,7467,8552,9600,10513,11204,11529,11533,10851,9704,8839,8174,7481,6541,5680],
    pv=[0,0,0,0,0,0,0,334,1719,4621,7534,9751,11152,10244,7679,5301,2869,743,0,0,0,0,0,0],
    wind=[7165,7420,7452,7594,7607,7565,7962,8003,7498,6850,7202,7771,8318,8520,8966,9076,8559,8525,8403,7729,7886,7166,7757,7305]
)

typicals = {
    "spring": spring_data,
    "summer": summer_data,
    "autumn": autumn_data,
    "winter": winter_data,
}

PROFILE_COLUMNS = ["Electric Load", "Heat Load", "Cold Load", "PV", "Wind"]
LOAD_COLUMNS = ["Electric Load", "Heat Load", "Cold Load"]

# ============================================================
# 1) 全局配置：论文近似强度
# ============================================================
DATA_YEAR = 2026
RNG_SEED = 42

# 与旧版相比，换季平滑窗口略放宽，但仍不大，避免人为产生大量跨季节混合样本。
TRANSITION_DAYS = 7

# 负荷：小到中等波动，保持可学习性
LOAD_COMMON_SIGMA = 0.020
LOAD_COMMON_BOUNDS = (0.95, 1.05)
LOAD_CHANNEL_SIGMA = {
    "Electric Load": 0.010,
    "Heat Load": 0.010,
    "Cold Load": 0.015,
}
LOAD_CHANNEL_BOUNDS = {
    "Electric Load": (0.97, 1.03),
    "Heat Load": (0.97, 1.03),
    "Cold Load": (0.96, 1.04),
}
LOAD_AR_PARAMS = {
    "Electric Load": {"rho": 0.65, "sigma": 0.012},
    "Heat Load": {"rho": 0.65, "sigma": 0.012},
    "Cold Load": {"rho": 0.65, "sigma": 0.015},
}
LOAD_REL_BOUNDS = {
    "Electric Load": (0.88, 1.12),
    "Heat Load": (0.88, 1.12),
    "Cold Load": (0.85, 1.15),
}

# PV：日级波动中等，并由天气状态驱动，不再用 rare shock 暴击式塌陷
PV_DAY_SIGMA = 0.070
PV_DAY_BOUNDS = (0.82, 1.18)
PV_AR_RHO = 0.55
PV_AR_SIGMA = 0.022
PV_REL_BOUNDS = (0.60, 1.30)
PV_REGIME_PROBS = {
    "spring": {"clear": 0.28, "normal": 0.57, "cloudy": 0.15},
    "summer": {"clear": 0.40, "normal": 0.48, "cloudy": 0.12},
    "autumn": {"clear": 0.25, "normal": 0.58, "cloudy": 0.17},
    "winter": {"clear": 0.18, "normal": 0.57, "cloudy": 0.25},
}

# 风电：略强于负荷、弱于旧版的极端设定
WIND_DAY_SIGMA = 0.070
WIND_DAY_BOUNDS = (0.85, 1.18)
WIND_AR_RHO = 0.65
WIND_AR_SIGMA = 0.028
WIND_REL_BOUNDS = (0.75, 1.25)
WIND_REGIME_PROBS = {
    "spring": {"weak": 0.20, "normal": 0.55, "windy": 0.25},
    "summer": {"weak": 0.25, "normal": 0.60, "windy": 0.15},
    "autumn": {"weak": 0.20, "normal": 0.60, "windy": 0.20},
    "winter": {"weak": 0.18, "normal": 0.57, "windy": 0.25},
}

SEASON_TRANSITIONS = (
    (3, 1, "winter", "spring"),
    (6, 1, "spring", "summer"),
    (9, 1, "summer", "autumn"),
    (12, 1, "autumn", "winter"),
)


# ============================================================
# 2) 季节选择、过渡期平滑与随机扰动基础函数
# ============================================================

def month_to_season(month: int) -> str:
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    if month in [9, 10, 11]:
        return "autumn"
    return "winter"



def _blend_profiles(from_season: str, to_season: str, alpha: float) -> pd.DataFrame:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = typicals[from_season].copy()
    for col in PROFILE_COLUMNS:
        blended[col] = (
            (1.0 - alpha) * typicals[from_season][col].to_numpy(dtype=float)
            + alpha * typicals[to_season][col].to_numpy(dtype=float)
        )
    return blended



def get_base_profile_blended(date, transition_days: int = 7):
    """获取某一天的基础曲线；靠近换季日时做平滑混合。"""
    date = pd.Timestamp(date).normalize()
    season = month_to_season(date.month)

    if transition_days <= 0:
        return typicals[season].copy(), season

    for month, day, from_season, to_season in SEASON_TRANSITIONS:
        boundary = pd.Timestamp(year=date.year, month=month, day=day)
        window_start = boundary - timedelta(days=transition_days)
        window_end = boundary + timedelta(days=transition_days)

        if window_start <= date <= window_end:
            span = max((window_end - window_start).days, 1)
            alpha = (date - window_start).days / span
            return _blend_profiles(from_season, to_season, alpha), season

    return typicals[season].copy(), season



def truncated_normal(rng, mu: float, sigma: float, low: float, high: float) -> float:
    return float(np.clip(rng.normal(mu, sigma), low, high))



def ar1_noise(rng, n: int = 24, rho: float = 0.6, sigma: float = 0.01) -> np.ndarray:
    """生成 AR(1) 自相关噪声，使相邻小时扰动连续。"""
    eps = rng.normal(0.0, sigma, size=n)
    x = np.zeros(n, dtype=float)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    return x



def clip_relative_to_base(values, base_values, bounds):
    """按相对典型日的上下界裁剪，抑制不合理极端样本。"""
    base_values = np.asarray(base_values, dtype=float)
    clipped = np.asarray(values, dtype=float).copy()
    low, high = bounds
    positive_mask = base_values > 0
    clipped[positive_mask] = np.clip(
        clipped[positive_mask],
        base_values[positive_mask] * low,
        base_values[positive_mask] * high,
    )
    clipped[~positive_mask] = 0.0
    return clipped



def sample_from_prob_dict(rng, prob_dict: dict) -> str:
    keys = list(prob_dict.keys())
    probs = np.array(list(prob_dict.values()), dtype=float)
    probs = probs / probs.sum()
    idx = rng.choice(len(keys), p=probs)
    return keys[idx]



def pv_regime_curve(base_df: pd.DataFrame, regime: str, rng) -> np.ndarray:
    """构造 24 小时 PV 天气状态曲线。"""
    curve = np.ones(24, dtype=float)
    daylight = base_df["PV"].to_numpy(dtype=float) > 0
    hours = np.arange(24)
    noon_shape = np.exp(-0.5 * ((hours - 13.0) / 3.0) ** 2)

    if regime == "clear":
        base_scale = rng.uniform(0.98, 1.06)
        curve[daylight] *= base_scale
    elif regime == "normal":
        base_scale = rng.uniform(0.92, 1.02)
        noon_dip = rng.uniform(0.02, 0.08)
        curve[daylight] *= base_scale * (1.0 - noon_dip * noon_shape[daylight])
    elif regime == "cloudy":
        base_scale = rng.uniform(0.80, 0.92)
        noon_dip = rng.uniform(0.08, 0.18)
        curve[daylight] *= base_scale * (1.0 - noon_dip * noon_shape[daylight])
    else:
        raise ValueError(f"Unknown PV regime: {regime}")

    return curve



def wind_regime_scale(regime: str, rng) -> float:
    if regime == "weak":
        return rng.uniform(0.92, 0.99)
    if regime == "normal":
        return rng.uniform(0.97, 1.03)
    if regime == "windy":
        return rng.uniform(1.03, 1.12)
    raise ValueError(f"Unknown wind regime: {regime}")


# ============================================================
# 3) 核心扰动：论文近似强度，不混入灾变日
# ============================================================

def apply_paper_like_perturbations(base_df: pd.DataFrame, rng, season: str):
    """对某一天的基础曲线施加中等强度扰动。"""
    df = base_df.copy()

    # ---------- 1) 负荷：公共日因子 + 个体因子 + 小时相关噪声 ----------
    load_common = truncated_normal(
        rng,
        mu=1.0,
        sigma=LOAD_COMMON_SIGMA,
        low=LOAD_COMMON_BOUNDS[0],
        high=LOAD_COMMON_BOUNDS[1],
    )

    for col in LOAD_COLUMNS:
        ch_factor = truncated_normal(
            rng,
            mu=1.0,
            sigma=LOAD_CHANNEL_SIGMA[col],
            low=LOAD_CHANNEL_BOUNDS[col][0],
            high=LOAD_CHANNEL_BOUNDS[col][1],
        )
        noise = ar1_noise(
            rng,
            24,
            rho=LOAD_AR_PARAMS[col]["rho"],
            sigma=LOAD_AR_PARAMS[col]["sigma"],
        )
        values = base_df[col].to_numpy(dtype=float) * load_common * ch_factor * (1.0 + noise)
        df[col] = clip_relative_to_base(values, base_df[col].to_numpy(dtype=float), LOAD_REL_BOUNDS[col])

    # ---------- 2) PV：日级因子 + 天气状态 + 小时 AR 噪声 ----------
    pv_day = truncated_normal(rng, 1.0, PV_DAY_SIGMA, *PV_DAY_BOUNDS)
    pv_regime = sample_from_prob_dict(rng, PV_REGIME_PROBS[season])
    pv_curve = pv_regime_curve(base_df, pv_regime, rng)
    pv_noise = ar1_noise(rng, 24, rho=PV_AR_RHO, sigma=PV_AR_SIGMA)
    pv_values = base_df["PV"].to_numpy(dtype=float) * pv_day * pv_curve * (1.0 + pv_noise)
    df["PV"] = clip_relative_to_base(pv_values, base_df["PV"].to_numpy(dtype=float), PV_REL_BOUNDS)

    # ---------- 3) 风电：日级因子 + 风况状态 + 小时 AR 噪声 ----------
    wind_day = truncated_normal(rng, 1.0, WIND_DAY_SIGMA, *WIND_DAY_BOUNDS)
    wind_regime = sample_from_prob_dict(rng, WIND_REGIME_PROBS[season])
    wind_scale = wind_regime_scale(wind_regime, rng)
    wind_noise = ar1_noise(rng, 24, rho=WIND_AR_RHO, sigma=WIND_AR_SIGMA)
    wind_values = base_df["Wind"].to_numpy(dtype=float) * wind_day * wind_scale * (1.0 + wind_noise)
    df["Wind"] = clip_relative_to_base(wind_values, base_df["Wind"].to_numpy(dtype=float), WIND_REL_BOUNDS)

    # ---------- 4) 物理约束 ----------
    df[PROFILE_COLUMNS] = df[PROFILE_COLUMNS].clip(lower=0.0)
    mask_night = base_df["PV"].to_numpy(dtype=float) == 0.0
    df.loc[mask_night, "PV"] = 0.0

    return df, pv_regime, wind_regime


# ============================================================
# 4) 生成主循环：按天生成全年 8760 条逐小时记录
# ============================================================

def generate_full_year_data(seed: int = RNG_SEED, year: int = DATA_YEAR) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    date_range = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
    all_records = []

    for date in date_range:
        base_profile, season = get_base_profile_blended(date, transition_days=TRANSITION_DAYS)
        daily_profile, pv_regime, wind_regime = apply_paper_like_perturbations(base_profile, rng, season)

        for h in range(24):
            all_records.append({
                "Month": int(date.month),
                "Day": int(date.day_of_year),
                "Hour": int(h),
                "Season": season,
                "PV Regime": pv_regime,
                "Wind Regime": wind_regime,
                "Electric Load": float(daily_profile.iloc[h]["Electric Load"]),
                "Heat Load": float(daily_profile.iloc[h]["Heat Load"]),
                "Cold Load": float(daily_profile.iloc[h]["Cold Load"]),
                "PV": float(daily_profile.iloc[h]["PV"]),
                "Wind": float(daily_profile.iloc[h]["Wind"]),
            })

    return pd.DataFrame(all_records)


# ============================================================
# 5) 划分与保存：输出训练/测试标记和模型可直接读取的数据文件
# ============================================================

def split_and_save(df: pd.DataFrame, prefix: str = "yearly_data_sci") -> pd.DataFrame:
    df = df.copy()
    df["Set"] = "train"

    # 与论文式组织方式一致：每月最后 5 天作为 holdout。
    for m in range(1, 13):
        days_in_month = sorted(df[df["Month"] == m]["Day"].unique())
        test_days = days_in_month[-5:]
        df.loc[(df["Month"] == m) & (df["Day"].isin(test_days)), "Set"] = "test"

    csv_path = f"{prefix}.csv"
    npy_path = f"{prefix}.npy"
    mask_path = f"{prefix}_set_mask.npy"

    df.to_csv(csv_path, index=False)
    print(f">>> CSV 已生成: {csv_path}")

    data_matrix = df[["Electric Load", "Heat Load", "Cold Load", "PV", "Wind"]].values
    num_days = len(df["Day"].unique())
    data_reshaped = data_matrix.reshape(num_days, 24, 5)
    np.save(npy_path, data_reshaped)
    print(f">>> NPY 已生成: {npy_path}")

    day_set = df[df["Hour"] == 0].sort_values("Day")["Set"].values
    set_mask = np.array([1 if s == "test" else 0 for s in day_set], dtype=np.int8)
    np.save(mask_path, set_mask)
    print(f">>> Set Mask 已生成: {mask_path}")

    return df


# ============================================================
# 6) 绘图：检查生成数据的日内均值、标准差和季节差异
# ============================================================

def plot_annual_stats(df: pd.DataFrame, out_png: str = "yearly_data_vis.png"):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Generated Annual Profiles (Mean ± Std)", fontsize=16)

    cols = ["Electric Load", "Heat Load", "Cold Load", "PV", "Wind"]
    colors = ["#1f77b4", "#d62728", "#9467bd", "#ff7f0e", "#2ca02c"]

    for i, col in enumerate(cols):
        ax = axes[i // 3, i % 3]
        stats = df.groupby("Hour")[col].agg(["mean", "std"]).reset_index()
        hour = stats["Hour"].to_numpy(dtype=int)
        mean = stats["mean"].to_numpy(dtype=float)
        std = np.nan_to_num(stats["std"].to_numpy(dtype=float), nan=0.0)

        ax.plot(hour, mean, color=colors[i], lw=2)
        ax.fill_between(hour, mean - std, mean + std, color=colors[i], alpha=0.25)
        ax.set_title(col, fontweight="bold")
        ax.set_xlim(0, 23)

    fig.delaxes(axes[1, 2])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f">>> 全年统计图已生成: {out_png}")



def plot_seasonal_stats(df: pd.DataFrame, out_png: str = "seasonal_profiles.png"):
    sns.set_style("whitegrid")
    seasons = ["spring", "summer", "autumn", "winter"]
    cols = ["Electric Load", "Heat Load", "Cold Load", "PV", "Wind"]
    colors = ["#1f77b4", "#d62728", "#9467bd", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(5, 4, figsize=(22, 14), sharex=True, sharey="row")
    fig.suptitle("Seasonal Source-Load Profiles (Mean ± Std)", fontsize=18, y=0.995)

    hours_full = np.arange(24)

    for i, col_name in enumerate(cols):
        y_max = float(df[col_name].max()) * 1.08

        for j, season in enumerate(seasons):
            ax = axes[i, j]
            df_season = df[df["Season"] == season]
            if df_season.empty:
                continue

            stats = df_season.groupby("Hour")[col_name].agg(["mean", "std"]).reindex(hours_full)
            mean = np.nan_to_num(stats["mean"].to_numpy(dtype=float), nan=0.0)
            std = np.nan_to_num(stats["std"].to_numpy(dtype=float), nan=0.0)

            ax.plot(hours_full, mean, color=colors[i], lw=2)
            ax.fill_between(hours_full, mean - std, mean + std, color=colors[i], alpha=0.25)

            if i == 0:
                ax.set_title(season.capitalize(), fontsize=13, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{col_name}\n(kW)", fontsize=11, fontweight="bold")
            if i == 4:
                ax.set_xlabel("Hour", fontsize=11)

            ax.set_xlim(0, 23)
            ax.set_ylim(0, y_max)
            ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f">>> 分季节统计图已生成: {out_png}")


# ============================================================
# 7) 诊断输出：便于快速检查波动强度是否合适
# ============================================================

def print_quick_stats(df: pd.DataFrame):
    print("\n>>> 全年按小时统计（均值 / 标准差 / 平均变异系数）")
    for col in PROFILE_COLUMNS:
        stats = df.groupby("Hour")[col].agg(["mean", "std"])
        avg_mean = float(stats["mean"].mean())
        avg_std = float(stats["std"].mean())
        avg_cv = avg_std / (avg_mean + 1e-8)
        print(f"{col:<15s} avg_mean={avg_mean:10.3f}, avg_std={avg_std:10.3f}, avg_cv={avg_cv:8.4f}")

    print("\n>>> 天气状态分布（按天统计）")
    day_df = df[df["Hour"] == 0].copy()
    print("PV Regime:")
    print(day_df["PV Regime"].value_counts(normalize=True).sort_index())
    print("\nWind Regime:")
    print(day_df["Wind Regime"].value_counts(normalize=True).sort_index())


# ============================================================
# 8) 主程序
# ============================================================

if __name__ == "__main__":
    df_final = generate_full_year_data(seed=RNG_SEED, year=DATA_YEAR)
    df_final = split_and_save(df_final, prefix="yearly_data_sci")
    plot_annual_stats(df_final, out_png="yearly_data_vis.png")
    plot_seasonal_stats(df_final, out_png="seasonal_profiles.png")
    print_quick_stats(df_final)
