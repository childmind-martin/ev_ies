"""EV 年度数据集生成脚本。

生成年度训练/测试环境需要的 EV 输入文件。生成逻辑按车辆角色抽样，
最后保存为一个全年 `.npy` 数组，并保持 `YearlyEVProvider` 所期望的
历史 10 特征字段格式。

重要约定：
- 三类 EV 采用同一套物理模型：
  cap_kwh ~ Uniform(50, 60), p_ch_max_kw = 14, p_dis_max_kw = 14, eff = 0.95。
- 是否允许 V2G 只由 v2g_flag 决定。
- 到站/离站时间直接输出为整数时段 arr_step / dep_step，避免环境中
  dep_step == t + 1 的离站判断因浮点数不相等而失效。
"""

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EVRoleSpec:
    """一类 EV 用户群体的行为参数。"""

    name: str
    count: int
    arr_mean: float
    arr_std: float
    arr_low: float
    arr_high: float
    dep_mean: float
    dep_std: float
    dep_low: float
    dep_high: float
    min_stay: float
    init_soc_low: float
    init_soc_high: float
    target_soc_low: float
    target_soc_high: float
    v2g_ratio: float


ROLE_SPECS = (
    # 采用“1240 辆总车 + 约 346 辆 V2G 车”的默认规模，
    # 使 14 kW 单车下的聚合充电/放电能力更接近前述文献比例：
    # - 最大聚合充电功率 ≈ 17.36 MW
    # - 最大聚合 V2G 放电功率 ≈ 4.84 MW
    # 员工普通通勤车：园区 EV 主体，早到晚走，主要承担工作地补能，不参与 V2G。
    EVRoleSpec(
        name="employee_commuter",
        count=870,
        arr_mean=8.4,
        arr_std=0.7,
        arr_low=7.0,
        arr_high=10.0,
        dep_mean=18.1,
        dep_std=0.8,
        dep_low=17.0,
        dep_high=19.5,
        min_stay=6.0,
        init_soc_low=0.30,
        init_soc_high=0.60,
        target_soc_low=0.90,
        target_soc_high=0.95,
        v2g_ratio=0.00,
    ),
    # 员工签约互动型通勤车：愿意参与 V2G，目标 SOC 稍低，给协调放电留出空间。
    EVRoleSpec(
        name="contracted_v2g_commuter",
        count=290,
        arr_mean=8.3,
        arr_std=0.7,
        arr_low=7.0,
        arr_high=9.8,
        # 晚高峰覆盖稳妥版：签约 V2G 通勤车适度延迟离站，
        # 覆盖 19-22 点峰段，同时避免大量员工车滞留到午夜。
        dep_mean=21.0,
        dep_std=1.0,
        dep_low=19.0,
        dep_high=23.5,
        min_stay=6.0,
        init_soc_low=0.45,
        init_soc_high=0.75,
        target_soc_low=0.80,
        target_soc_high=0.88,
        v2g_ratio=1.00,
    ),
    # 园区轻型服务车：时间窗口更规律、可控性更强，其中一部分具备 V2G 资格。
    EVRoleSpec(
        name="light_service_ev",
        count=80,
        arr_mean=8.2,
        arr_std=0.5,
        arr_low=7.2,
        arr_high=9.2,
        # 晚高峰覆盖稳妥版：轻型服务车作为园区可控车队，
        # 停留窗口略长于普通通勤车，增强晚峰 V2G 可用性。
        dep_mean=21.5,
        dep_std=1.0,
        dep_low=19.5,
        dep_high=24.0,
        min_stay=6.0,
        init_soc_low=0.35,
        init_soc_high=0.70,
        target_soc_low=0.82,
        target_soc_high=0.90,
        v2g_ratio=0.70,
    ),
)

DEFAULT_NUM_EVS = sum(spec.count for spec in ROLE_SPECS)
CAP_KWH_LOW = 50.0
CAP_KWH_HIGH = 60.0
P_CH_MAX_KW = 14.0
P_DIS_MAX_KW = 14.0
EFF = 0.95
ENERGY_PER_KM = 0.16


def _sample_clipped_normal(rng, mean, std, low, high, size):
    values = rng.normal(mean, std, size=size)
    return np.clip(values, low, high)


def _role_counts_for_num_evs(num_evs: int) -> np.ndarray:
    """根据默认 870/290/80 配置，为给定总车数生成每类固定数量。"""
    base_counts = np.array([spec.count for spec in ROLE_SPECS], dtype=np.int64)
    base_total = int(base_counts.sum())
    if num_evs == base_total:
        return base_counts.copy()

    raw_counts = base_counts.astype(np.float64) * float(num_evs) / max(base_total, 1)
    counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(num_evs - counts.sum())
    if remainder > 0:
        fractional_order = np.argsort(-(raw_counts - counts))
        counts[fractional_order[:remainder]] += 1
    return counts


def _sample_v2g_mask(rng, v2g_ratio: float, size: int) -> np.ndarray:
    """按给定比例生成 V2G 标志；当前默认配置下每天约得到 346 辆 V2G 车。"""
    n_v2g = int(round(size * v2g_ratio))
    n_v2g = int(np.clip(n_v2g, 0, size))
    v2g_mask = np.zeros(size, dtype=np.float32)
    if n_v2g > 0:
        v2g_mask[:n_v2g] = 1.0
        rng.shuffle(v2g_mask)
    return v2g_mask


def _sample_role_group(rng, spec: EVRoleSpec, size: int):
    """为某一类 EV 批量抽样到站、离站、SOC 和 V2G 能力。"""
    arr_raw = _sample_clipped_normal(
        rng, spec.arr_mean, spec.arr_std, spec.arr_low, spec.arr_high, size
    )
    dep_raw = _sample_clipped_normal(
        rng, spec.dep_mean, spec.dep_std, spec.dep_low, spec.dep_high, size
    )
    dep_raw = np.maximum(dep_raw, arr_raw + spec.min_stay)

    # 直接转换为环境使用的整数时段。
    # 到达向下取整：车辆从该小时开始可接入。
    # 离站向上取整：车辆在该整点离开，环境可用 dep_step == t + 1 判断离站。
    arr_step = np.floor(arr_raw).astype(np.int32)
    dep_step = np.ceil(dep_raw).astype(np.int32)

    arr_step = np.clip(arr_step, 0, 23)
    min_stay_steps = int(np.ceil(spec.min_stay))
    dep_step = np.maximum(dep_step, arr_step + min_stay_steps)
    dep_step = np.clip(dep_step, 1, 24)

    init_soc = rng.uniform(spec.init_soc_low, spec.init_soc_high, size=size)
    target_soc = rng.uniform(spec.target_soc_low, spec.target_soc_high, size=size)
    v2g_mask = _sample_v2g_mask(rng, spec.v2g_ratio, size)

    return arr_step, dep_step, init_soc, target_soc, v2g_mask


def generate_ev_dataset_role_based(
    num_evs=DEFAULT_NUM_EVS,
    days=365,
    save_path="ev_data_s123_bilevel.npy",
    seed=42,
):
    """
    生成基于三类角色的 EV 年度数据集。

    输出形状：
        (days, num_evs, 10), float32

    字段索引：
        0: arr_step，到站时段，整数小时，0-23
        1: dep_step，离站时段，整数小时，1-24
        2: 初始 SOC
        3: 目标 SOC
        4: V2G 标志，0/1
        5: 电池容量，kWh
        6: 最大充电功率，kW
        7: 最大放电功率，kW；是否允许使用由第 4 列 V2G 标志决定
        8: 日行驶里程，km，由初始 SOC 近似反推
        9: 充放电效率
    """
    rng = np.random.default_rng(seed)
    role_counts = _role_counts_for_num_evs(num_evs)

    ev_data = np.zeros((days, num_evs, 10), dtype=np.float32)
    role_names = [spec.name for spec in ROLE_SPECS]
    role_counts_total = {name: 0 for name in role_names}
    v2g_total = 0

    for d in range(days):
        role_ids = np.repeat(np.arange(len(ROLE_SPECS), dtype=np.int32), role_counts)
        rng.shuffle(role_ids)

        arr_step = np.zeros(num_evs, dtype=np.float32)
        dep_step = np.zeros(num_evs, dtype=np.float32)
        init_soc = np.zeros(num_evs, dtype=np.float32)
        target_soc = np.zeros(num_evs, dtype=np.float32)
        v2g_mask = np.zeros(num_evs, dtype=np.float32)
        cap_kwh = rng.uniform(CAP_KWH_LOW, CAP_KWH_HIGH, size=num_evs).astype(np.float32)

        for ridx, spec in enumerate(ROLE_SPECS):
            mask = role_ids == ridx
            count = int(mask.sum())
            if count == 0:
                continue

            role_counts_total[spec.name] += count

            (
                arr_step_role,
                dep_step_role,
                init_soc_role,
                target_soc_role,
                v2g_mask_role,
            ) = _sample_role_group(rng, spec, count)

            arr_step[mask] = arr_step_role.astype(np.float32)
            dep_step[mask] = dep_step_role.astype(np.float32)
            init_soc[mask] = init_soc_role.astype(np.float32)
            target_soc[mask] = target_soc_role.astype(np.float32)
            v2g_mask[mask] = v2g_mask_role

        mileage = np.clip((1.0 - init_soc) * cap_kwh / ENERGY_PER_KM, 5.0, 300.0)
        v2g_total += int(v2g_mask.sum())

        ev_data[d, :, 0] = arr_step
        ev_data[d, :, 1] = dep_step
        ev_data[d, :, 2] = init_soc
        ev_data[d, :, 3] = target_soc
        ev_data[d, :, 4] = v2g_mask
        ev_data[d, :, 5] = cap_kwh
        ev_data[d, :, 6] = P_CH_MAX_KW
        ev_data[d, :, 7] = P_DIS_MAX_KW
        ev_data[d, :, 8] = mileage.astype(np.float32)
        ev_data[d, :, 9] = EFF

    np.save(save_path, ev_data)

    total_evs = days * num_evs
    realized_v2g_ratio = v2g_total / max(total_evs, 1)
    avg_v2g_per_day = v2g_total / max(days, 1)

    print("EV dataset generated successfully.")
    print(f"Saved to: {os.path.abspath(save_path)}")
    print(f"Shape: {days} days x {num_evs} EVs x 10 features")
    print(f"Max aggregate charging power: {num_evs * P_CH_MAX_KW / 1000.0:.3f} MW")
    print(f"Max aggregate V2G discharging power: {avg_v2g_per_day * P_DIS_MAX_KW / 1000.0:.3f} MW")
    print("Role shares:")
    for ridx, spec in enumerate(ROLE_SPECS):
        realized_share = role_counts_total[spec.name] / max(total_evs, 1)
        configured_count = int(role_counts[ridx])
        configured_share = configured_count / max(num_evs, 1)
        print(
            f"  {spec.name}: count={configured_count}, configured={configured_share:.2%}, "
            f"realized={realized_share:.2%}, v2g_ratio={spec.v2g_ratio:.2%}"
        )
    print(f"Average V2G vehicles per day: {avg_v2g_per_day:.1f}")
    print(f"Realized aggregate V2G ratio: {realized_v2g_ratio:.2%}")

    return ev_data



def print_peak_hour_coverage(ev_data: np.ndarray, peak_hours=(12, 13, 14, 19, 20, 21, 22)):
    """打印 EV 在峰值电价时段的接入覆盖情况。

    注意：这里统计的是“接入且具备 V2G 资格”的理论最大功率；
    实际放电还会受到 target_soc、当前 SOC、剩余停留时间和 EMS 动作限制。
    """
    arr = ev_data[:, :, 0].astype(np.int32)
    dep = ev_data[:, :, 1].astype(np.int32)
    v2g = ev_data[:, :, 4] > 0.5
    p_ch = ev_data[:, :, 6]
    p_dis = ev_data[:, :, 7]

    print("\nPeak-hour EV availability check:")
    print("hour | active_EV | active_V2G | charge_cap_MW | v2g_cap_MW")
    for h in peak_hours:
        active = (arr <= h) & (h < dep)
        active_v2g = active & v2g
        active_ev_avg = float(active.sum(axis=1).mean())
        active_v2g_avg = float(active_v2g.sum(axis=1).mean())
        charge_cap_mw = float((active * p_ch).sum(axis=1).mean() / 1000.0)
        v2g_cap_mw = float((active_v2g * p_dis).sum(axis=1).mean() / 1000.0)
        print(
            f"{h:>4d} | {active_ev_avg:>9.1f} | {active_v2g_avg:>10.1f} | "
            f"{charge_cap_mw:>13.3f} | {v2g_cap_mw:>10.3f}"
        )


def generate_ev_dataset_50_percent_v2g(
    num_evs=DEFAULT_NUM_EVS,
    days=365,
    save_path="ev_data_s123_bilevel.npy",
    seed=42,
):
    """向后兼容旧函数名。"""
    return generate_ev_dataset_role_based(
        num_evs=num_evs,
        days=days,
        save_path=save_path,
        seed=seed,
    )


if __name__ == "__main__":
    ev_data = generate_ev_dataset_role_based()
    print_peak_hour_coverage(ev_data)
