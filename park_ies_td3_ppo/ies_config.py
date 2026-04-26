from dataclasses import dataclass, field

import numpy as np

__all__ = ["IESConfig"]


def _default_grid_buy_price_24() -> np.ndarray:
    """
    按照表格生成一天24小时购电价格
    谷段：00:00-06:00、22:00-23:00 -> 0.32
    峰段：10:00-12:00、14:00-15:00 -> 1.05
    平段：其余时间 -> 0.82
    """
    buy = np.array(
        [
            0.32, 0.32, 0.32, 0.32, 0.32, 0.32,
            0.82, 0.82, 0.82, 0.82,
            1.05, 1.05,
            0.82, 0.82,
            1.05,
            0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82,
            0.32, 0.32,
        ],
        dtype=np.float32,
    )
    return buy


def _default_grid_sell_price_24() -> np.ndarray:
    return np.full(24, 0.05, dtype=np.float32)


def _default_gas_price_24() -> np.ndarray:
    return np.full(24, 0.2732, dtype=np.float32)


@dataclass
class IESConfig:
    # ====================
    # 一、仿真时间与预测窗口
    # ====================
    # episode_length: 单个回合长度，这里默认 24 个时段（即 24h 典型日）
    # dt: 单步时长，单位 h
    # future_horizon: EV 未来边界预测步数
    # exogenous_future_horizon: 外生量（负荷、PV、WT）未来预测步数
    episode_length: int = 24
    dt: float = 1.0
    future_horizon: int = 7
    exogenous_future_horizon: int = 3

    # ====================
    # 二、电网与基础购售电约束
    # ====================
    # 最大购电功率。
    # 最大上网功率；超过部分记为弃电/溢出。
    grid_import_max: float = 20000.0
    grid_export_max: float = 3000.0
    grid_buy_price_24: np.ndarray = field(default_factory=_default_grid_buy_price_24)
    grid_sell_price_24: np.ndarray = field(default_factory=_default_grid_sell_price_24)
    gas_price_24: np.ndarray = field(default_factory=_default_gas_price_24)

    # ====================
    # 三、燃气轮机 GT 与余热锅炉 WHB
    # ====================
    # GT 先发电，再由余热回收得到热功率。
    gt_p_max: float = 15000.0
    gt_eta_e: float = 0.35
    gt_ramp: float = 3000.0

    whb_h_max: float = 17000.0
    whb_eta_h: float = 0.83

    # ====================
    # 四、燃气锅炉 GB
    # ====================
    gb_h_max: float = 18000.0
    gb_eta_h: float = 0.85
    gb_ramp: float = 3600.0

    # ====================
    # 五、制冷设备：AC / EC
    # ====================
    # AC: 吸收式制冷机，耗热制冷
    # EC: 电制冷机，耗电制冷
    ac_c_max: float = 25000.0
    ac_cop: float = 1.20

    ec_c_max: float = 23000.0
    ec_cop: float = 3.50

    # ====================
    # 六、电储能 EES
    # ====================
    # 重点关注：SOC 边界、充放电效率、固定初始 SOC。
    ees_p_max: float = 3500.0
    ees_e_cap: float = 10500.0
    # EES 额定功率为 3500 kW、额定容量为 10500 kWh，对应典型额定时长为 3 h，更接近参考文献的相对配置。
    ees_eta_ch: float = 0.95
    ees_eta_dis: float = 0.95
    ees_self_discharge: float = 0.0
    ees_soc_min: float = 0.10
    ees_soc_max: float = 1.00
    # EES 初始 SOC 固定为 0.80。
    ees_soc_init: float = 0.80

    # ====================
    # 七、电动汽车 EV 聚合体
    # ====================
    # 重点关注：出行目标 SOC、V2G 缓冲区、离站缺能惩罚、离站风险惩罚。
    ev_soc_min: float = 0.10
    ev_soc_max: float = 0.95
    ev_eta_ch: float = 0.95
    ev_eta_dis: float = 0.95
    ev_self_discharge: float = 0.0
    # V2G 车辆在离站目标 SOC 之上保留额外调度缓冲，用于低价/富余时段充电并支撑高峰放电。
    ev_v2g_buffer_soc_margin: float = 0.10
    # EV dynamic safe discharge boundary: reserve recovery time before departure.
    ev_safe_discharge_reserve_h: float = 1.0
    # Numerical tolerance used when comparing SOC against the safe discharge lower bound.
    ev_safe_discharge_soc_epsilon: float = 1e-4
    c_deg: float = 0.01

    # ====================
    # 八、运维成本系数
    # ====================
    # 各设备单位运维成本系数。
    om_gt: float = 0.10
    om_gb: float = 0.05
    om_whb: float = 0.02
    om_wt: float = 0.01
    om_pv: float = 0.01
    om_ec: float = 0.02
    om_ac: float = 0.03
    # EES 运维成本仅按放电通量计入。
    om_ees: float = 0.01

    # ====================
    # 九、GT 安全层 / 反送电 / 低价时段抑制相关参数
    # ====================
    # GT 动作已直接映射到安全可行区间，不再设置 GT 安全裁剪惩罚。
    # GT 安全区间搜索网格数。
    gt_safe_search_points: int = 61
    # GT 电侧防反送边界的容差。
    gt_export_margin: float = 0.0
    # 在低电价、低电侧压力时，限制 GT 接近热侧最小可行出力。
    gt_low_price_threshold: float = 0.60
    gt_low_price_pressure_threshold_kw: float = 4000.0
    gt_low_price_headroom_kw: float = 1000.0

    # ====================
    # 十、惩罚项与奖励缩放
    # ====================
    penalty_unserved_e: float = 10000.0
    penalty_unserved_h: float = 10000.0
    penalty_unserved_c: float = 10000.0
    # 保留旧字段名；实际惩罚按离站缺电量 kWh 计算，而不是直接按 SOC 缺口计算。
    penalty_depart_soc: float = 5000.0
    # EV 离站缺电惩罚分为轻度、中度和重度三段。
    depart_penalty_soft_kwh: float = 2.0
    depart_penalty_mid_kwh: float = 5.0
    penalty_depart_energy_soft: float = 300.0
    penalty_depart_energy_mid: float = 1200.0
    # 在车辆真正离站前，对潜在缺电风险进行平滑惩罚。
    ev_depart_risk_window_h: float = 2.0
    ev_depart_risk_buffer_h: float = 1.0
    penalty_ev_depart_risk: float = 3.0
    # EV 充放电奖励由实时电价和系统电功率压力触发。
    ev_discharge_price_threshold: float = 0.88
    ev_charge_price_threshold: float = 0.60
    ev_price_tolerance: float = 1e-6
    ev_discharge_pressure_threshold_kw: float = 50.0
    # 目标 SOC 内的柔性充电奖励低于 V2G 缓冲充电奖励。
    reward_storage_discharge_base: float = 0.60
    reward_storage_charge_base: float = 0.10
    reward_ev_target_timing_base: float = 0.05
    ees_reward_discharge_soc_floor: float = 0.20
    ees_reward_charge_soc_target: float = 0.85
    ev_peak_export_tolerance_kw: float = 50.0
    # 防止 EV 放电只是造成外送而不是真正削峰。
    penalty_ev_export_guard: float = 0.20
    # 旧外送溢出惩罚已停用，实际弃电通过弃电惩罚处理。
    penalty_surplus_e: float = 0.10
    penalty_export_e: float = 0.06
    penalty_surplus_h: float = 0.10
    penalty_surplus_c: float = 0.10
    reward_scale: float = 1e-5

    # ====================
    # 十一、观测归一化参考上限
    # ====================
    price_max: float = 1.1
    elec_load_max: float = 22000.0
    heat_load_max: float = 18000.0
    cool_load_max: float = 36000.0
    pv_max: float = 20000.0
    wt_max: float = 13000.0
