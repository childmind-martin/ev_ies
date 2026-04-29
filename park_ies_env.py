# -*- coding: utf-8 -*-
"""
park_ies_env.py（注释版）
======================

这个文件实现的是“园区综合能源系统（IES）+ EV 聚合体 + EES 储能”的
Gymnasium 强化学习环境。核心目标是：

1. 在每个时段接收智能体动作；
2. 将动作映射为 GT / EV / EES / 冷热设备的实际执行功率；
3. 结算电、热、冷三类能量平衡；
4. 计算购电、售电、燃气、运维、退化、惩罚和奖励；
5. 返回下一时刻观测、奖励和诊断信息。

阅读建议：
- 先看 IESConfig：了解所有设备参数、奖励惩罚系数、约束上限；
- 再看 reset()：了解 episode 开始时状态如何初始化；
- 再看 step()：这是环境主流程；
- 最后看各个 _dispatch_* / _compute_* 函数：它们是各子模块的细节实现。
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
# 继续从 park_ies_env 导出 IESConfig，兼容现有训练/测试脚本的导入路径。
from ies_config import IESConfig

class ParkIESEnv(gym.Env):
    """
    单智能体园区 IES 强化学习环境。

    一个 episode 对应 24 个离散时段。每一步由智能体给出：
    1) EES 充/放电动作
    2) GT 出力动作
    3) EV 柔性充电动作
    4) EV 柔性放电动作

    环境内部负责：
    - 根据动作和物理边界生成真实执行功率；
    - 结算冷热电平衡；
    - 计算系统成本与各类惩罚/奖励；
    - 输出下一步观测和详细诊断信息。
    """

    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------
    # 构造函数
    # 作用：
    # 1. 保存配置、外生时序数据、EV 数据；
    # 2. 构建动作空间与观测空间；
    # 3. 初始化环境内部状态和整回合累计统计量。
    # 纠错时优先检查：
    # - action_space 维度是否与训练脚本一致；
    # - obs_dim 是否与 _get_obs() 实际拼接长度一致。
    # ------------------------------------------------------------------
    def __init__(
        self,
        cfg: IESConfig,
        ts_data: Dict[str, np.ndarray],
        ev_data: Dict[str, np.ndarray],
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        # 保存环境配置以及外部输入数据，后续 reset()/step() 都从这里读取。
        self.cfg = cfg
        self.ts = {k: np.array(v, copy=True) for k, v in ts_data.items()}
        self.ev = {k: np.array(v, copy=True) for k, v in ev_data.items()}
        self.render_mode = render_mode

        # T 为单个 episode 的时步长度，n_ev 为当前场景中的车辆总数。
        self.T = int(cfg.episode_length)

        # 在构造阶段先做输入校验，尽早发现字段缺失、长度不一致等问题。
        self._validate_inputs()
        self.n_ev = int(self.ev["arr_step"].shape[0])

        # 4 维连续动作：
        # [0] EES 充/放电归一化指令，[-1, 1]
        # [1] GT 出力归一化指令，[-1, 1]
        # [2] EV 柔性充电强度，[0, 1]
        # [3] EV 柔性放电强度，[0, 1]
        self.action_low = np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        self.action_high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(
            low=self.action_low, high=self.action_high, dtype=np.float32
        )
        # 当前 20 个状态特征。
        # 其中额外加入 EES 当前充电功率状态、当前放电功率状态，
        # 让策略能感知储能当前的执行状态。
        # 加上未来 7 步 EV 刚性充电、柔性充电、柔性放电边界预测。
        # 加上未来 3 步电/热/冷负荷、PV、风电预测。
        self.obs_dim = (
            20
            + 4 * self.cfg.future_horizon
            + 5 * self.cfg.exogenous_future_horizon
        )
        # 观测统一归一化到 [0, 1]，由当前状态和未来预测拼接而成。
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # 运行时状态：当前时刻、储能状态、冷热电设备输出以及 EV 当前 SOC。
        self.t = 0
        self.ees_soc_episode_init = float(
            np.clip(
                cfg.ees_soc_init,
                cfg.ees_soc_min,
                cfg.ees_soc_max,
            )
        )
        self.ees_soc = self.ees_soc_episode_init
        self.ees_power = 0.0
        self.gt_power = 0.0
        self.gb_heat = 0.0
        self.ac_cool = 0.0
        self.ec_cool = 0.0
        self.ev_soc = None
        # episode 级成本统计。
        self.episode_system_cost = 0.0
        self.episode_cost_grid = 0.0
        self.episode_cost_gas = 0.0
        self.episode_cost_deg = 0.0
        self.episode_cost_om = 0.0
        self.episode_penalty_cost = 0.0
        # episode 级奖励拆分，便于分析奖励塑形是否有效。
        self.episode_guide_reward = 0.0
        self.episode_reward_raw = 0.0
        # 供需缺口、弃能与上网电量统计。
        self.episode_unserved_e_kwh = 0.0
        self.episode_unserved_h_kwh = 0.0
        self.episode_unserved_c_kwh = 0.0
        self.episode_surplus_e_kwh = 0.0
        self.episode_export_e_kwh = 0.0
        self.episode_surplus_h_kwh = 0.0
        self.episode_surplus_c_kwh = 0.0
        # 各类惩罚项累计值。
        self.episode_penalty_unserved_e = 0.0
        self.episode_penalty_unserved_h = 0.0
        self.episode_penalty_unserved_c = 0.0
        self.episode_penalty_surplus_e = 0.0
        self.episode_penalty_surplus_h = 0.0
        self.episode_penalty_surplus_c = 0.0
        self.episode_penalty_export_e = 0.0
        self.episode_penalty_depart_energy = 0.0
        self.episode_penalty_depart_risk = 0.0
        self.episode_penalty_ev_export_guard = 0.0
        self.episode_terminal_ees_shortage_kwh = 0.0
        self.episode_penalty_terminal_ees_soc = 0.0
        self.episode_final_ees_soc = float(self.ees_soc)
        # 储能和 EV 引导奖励对应的行为统计量。
        self.episode_reward_storage_discharge = 0.0
        self.episode_reward_storage_charge = 0.0
        self.episode_reward_ev_target_timing = 0.0
        self.episode_storage_peak_shaved_kwh = 0.0
        self.episode_storage_charge_rewarded_kwh = 0.0
        self.episode_ees_charge_rewarded_kwh = 0.0
        self.episode_ev_flex_target_charge_kwh = 0.0
        self.episode_ev_buffer_charge_kwh = 0.0
        self.episode_low_value_charge_kwh = 0.0
        # GT 安全修正相关统计。
        self.episode_gt_export_clip = 0.0
        self.episode_gt_export_clip_steps = 0
        self.episode_gt_safe_infeasible_steps = 0
        # 缓存最近一步 info，便于调试、渲染或训练日志直接读取。
        self.last_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # reset()
    # 作用：重置一个新 episode。
    # 关键动作：
    # - 固定设置 EES 初始 SOC；
    # - 载入 EV 初始 SOC；
    # - 清空所有 episode 统计量；
    # - 通过 _initialize_operating_point() 给 GT/GB/AC/EC 一个可行热启动点。
    # 纠错时优先检查：
    # - EV 初始 SOC 是否与数据一致；
    # - 热启动是否把第 0 时刻的设备状态带偏。
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.t = 0
        self.ees_soc_episode_init = float(
            np.clip(
                self.cfg.ees_soc_init,
                self.cfg.ees_soc_min,
                self.cfg.ees_soc_max,
            )
        )
        self.ees_soc = self.ees_soc_episode_init
        self.ees_power = 0.0
        self.gt_power = 0.0
        self.gb_heat = 0.0
        self.ac_cool = 0.0
        self.ec_cool = 0.0
        self.ev_soc = np.clip(
            self.ev["init_soc"].astype(np.float32).copy(),
            self.cfg.ev_soc_min,
            self.cfg.ev_soc_max,
        )
        self.episode_system_cost = 0.0
        self.episode_cost_grid = 0.0
        self.episode_cost_gas = 0.0
        self.episode_cost_deg = 0.0
        self.episode_cost_om = 0.0
        self.episode_penalty_cost = 0.0
        self.episode_guide_reward = 0.0
        self.episode_reward_raw = 0.0
        self.episode_unserved_e_kwh = 0.0
        self.episode_unserved_h_kwh = 0.0
        self.episode_unserved_c_kwh = 0.0
        self.episode_surplus_e_kwh = 0.0
        self.episode_export_e_kwh = 0.0
        self.episode_surplus_h_kwh = 0.0
        self.episode_surplus_c_kwh = 0.0
        self.episode_penalty_unserved_e = 0.0
        self.episode_penalty_unserved_h = 0.0
        self.episode_penalty_unserved_c = 0.0
        self.episode_penalty_surplus_e = 0.0
        self.episode_penalty_surplus_h = 0.0
        self.episode_penalty_surplus_c = 0.0
        self.episode_penalty_export_e = 0.0
        self.episode_penalty_depart_energy = 0.0
        self.episode_penalty_depart_risk = 0.0
        self.episode_penalty_ev_export_guard = 0.0
        self.episode_terminal_ees_shortage_kwh = 0.0
        self.episode_penalty_terminal_ees_soc = 0.0
        self.episode_final_ees_soc = float(self.ees_soc)
        self.episode_reward_storage_discharge = 0.0
        self.episode_reward_storage_charge = 0.0
        self.episode_reward_ev_target_timing = 0.0
        self.episode_storage_peak_shaved_kwh = 0.0
        self.episode_storage_charge_rewarded_kwh = 0.0
        self.episode_ees_charge_rewarded_kwh = 0.0
        self.episode_ev_flex_target_charge_kwh = 0.0
        self.episode_ev_buffer_charge_kwh = 0.0
        self.episode_low_value_charge_kwh = 0.0
        self.episode_gt_export_clip = 0.0
        self.episode_gt_export_clip_steps = 0
        self.episode_gt_safe_infeasible_steps = 0
        self._initialize_operating_point()

        obs = self._get_obs()
        info = {"msg": "env reset"}
        self.last_info = dict(info)
        return obs, info

    # ------------------------------------------------------------------
    # step() —— 环境主流程
    # 主线顺序：
    # A. 动作预处理与裁剪
    # B. 计算 EV / EES / GT / 冷热设备的实际执行量
    # C. 更新 EES 和 EV 的 SOC
    # D. 计算离站、储备、末端 SOC 等惩罚
    # E. 结算电网功率与系统成本
    # F. 汇总奖励与诊断信息
    # 这部分是纠错的核心。若出现“策略学不对、设备行为异常、供需不平衡、
    # 奖励方向不对”等问题，通常都要回到这里逐段排查。
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        self._ensure_runtime_state_initialized()
        # ===== 1) 动作预处理：类型转换、维度检查、裁剪到动作空间范围 =====
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_space.shape[0]:
            raise ValueError(
                f"Action dimension mismatch: expected {self.action_space.shape[0]}, got {action.shape[0]}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError(f"Action contains non-finite values: {action}")
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # ===== 2) 读取当前时刻外生量 =====
        exo = self._get_exogenous(self.t)

        a_ees = float(action[0])
        a_gt = float(action[1])
        # EV 动作通道取值为 [0, 1]，0 表示不执行额外柔性充放电。
        a_ev_ch = float(action[2])
        a_ev_dis = float(action[3])

        # ===== 3) 根据当前状态计算 EV / EES 的真实可行调度边界 =====
        ev_now = self._compute_ev_boundaries(self.t, self.ev_soc)
        ev_dispatch = self._dispatch_ev(
            a_ev_ch=a_ev_ch,
            a_ev_dis=a_ev_dis,
            ev_state=ev_now,
        )
        ees_dispatch = self._dispatch_ees(a_ees)

        # 计算当前时刻 GT 动态安全出力区间。
        # GT 动作直接映射到可行区间，避免先产生不可行动作再由安全层裁剪。

        # ===== 4) 搜索 GT 当前时刻的安全出力区间 =====
        dispatch_solution = self._resolve_step_dispatch(
            exo=exo,
            a_gt=a_gt,
            ev_dispatch=ev_dispatch,
            ees_dispatch=ees_dispatch,
        )
        ev_dispatch_capped = self._clip_ev_discharge_to_local_absorption(
            ev_dispatch=ev_dispatch,
            p_grid=dispatch_solution["p_grid"],
        )
        if ev_dispatch_capped["p_ev_dis"] + 1e-6 < ev_dispatch["p_ev_dis"]:
            ev_dispatch = ev_dispatch_capped
            dispatch_solution = self._resolve_step_dispatch(
                exo=exo,
                a_gt=a_gt,
                ev_dispatch=ev_dispatch,
                ees_dispatch=ees_dispatch,
            )

        gt_safe = dispatch_solution["gt_safe"]
        p_gt_safe_min = gt_safe["p_gt_safe_min"]
        p_gt_safe_max = gt_safe["p_gt_safe_max"]
        gt_safe_feasible = gt_safe["joint_feasible"]
        p_gt_final = dispatch_solution["p_gt_final"]
        gt_export_clip = dispatch_solution["gt_export_clip"]
        hc_dispatch = dispatch_solution["hc_dispatch"]

        depart_energy_shortage_kwh = 0.0
        penalty_depart_energy = 0.0
        penalty_depart_energy_soft = 0.0
        penalty_depart_energy_mid = 0.0
        penalty_depart_energy_hard = 0.0
        depart_shortage_soft_kwh = 0.0
        depart_shortage_mid_kwh = 0.0
        depart_shortage_hard_kwh = 0.0
        depart_risk_energy_kwh = 0.0
        penalty_depart_risk = 0.0
        depart_risk_vehicle_count = 0
        ev_peak_pressure_without_storage_kw = 0.0
        ev_export_overlap_kwh = 0.0
        penalty_ev_export_guard = 0.0
        storage_peak_shaved_kwh = 0.0
        storage_charge_rewarded_kwh = 0.0
        ees_charge_rewarded_kwh = 0.0
        reward_storage_discharge_bonus = 0.0
        reward_storage_charge_bonus = 0.0
        reward_ev_target_timing_bonus = 0.0
        low_value_energy_before_storage_prepare_kwh = 0.0
        ees_discharge_reward_weight = 0.0
        ees_charge_reward_weight = 0.0

        # ===== 6) 更新储能与 EV 的状态（SOC） =====
        ees_soc_before = float(self.ees_soc)
        self.ees_soc = (
            self.ees_soc * (1.0 - self.cfg.ees_self_discharge)
            + (
                ees_dispatch["p_ees_ch"] * self.cfg.ees_eta_ch * self.cfg.dt
                - ees_dispatch["p_ees_dis"] * self.cfg.dt / self.cfg.ees_eta_dis
            )
            / self.cfg.ees_e_cap
        )
        self.ees_soc = float(np.clip(self.ees_soc, self.cfg.ees_soc_min, self.cfg.ees_soc_max))
        self.ees_power = float(ees_dispatch["p_ees_dis"] - ees_dispatch["p_ees_ch"])
        is_terminal_step = (self.t + 1 >= self.T)
        terminal_ees_required_soc = max(
            self.cfg.ees_soc_min,
            self.ees_soc_episode_init - self.cfg.ees_terminal_soc_tolerance,
        )
        terminal_ees_shortage_kwh = 0.0
        penalty_terminal_ees_soc = 0.0
        if is_terminal_step:
            terminal_ees_shortage_soc = max(
                0.0,
                terminal_ees_required_soc - float(self.ees_soc),
            )
            terminal_ees_shortage_kwh = terminal_ees_shortage_soc * self.cfg.ees_e_cap
            penalty_terminal_ees_soc = (
                terminal_ees_shortage_kwh * self.cfg.penalty_ees_terminal_soc
            )

        active_ev_mask = np.asarray(ev_now["active_mask"], dtype=bool)
        ev_soc_delta = (
            ev_dispatch["p_ch_each"] * self.cfg.ev_eta_ch * self.cfg.dt
            - ev_dispatch["p_dis_each"] * self.cfg.dt / self.cfg.ev_eta_dis
        ) / self.ev["cap_kwh"]
        if np.any(active_ev_mask):
            self.ev_soc = self.ev_soc.copy()
            self.ev_soc[active_ev_mask] = (
                self.ev_soc[active_ev_mask] * (1.0 - self.cfg.ev_self_discharge)
                + ev_soc_delta[active_ev_mask]
            )
        self.ev_soc = np.clip(self.ev_soc, self.cfg.ev_soc_min, self.cfg.ev_soc_max)
        if self.ev_soc.size > 0:
            ev_soc_mean = float(np.mean(self.ev_soc))
        else:
            ev_soc_mean = 0.0
        # ===== 7) 计算 EV 离站缺能惩罚与提前风险惩罚 =====
        dep_mask = self.ev["dep_step"] == (self.t + 1)
        if np.any(dep_mask):
            shortage_soc = np.maximum(
                self.ev["target_soc"][dep_mask] - self.ev_soc[dep_mask],
                0.0,
            )
            shortage_kwh = shortage_soc * self.ev["cap_kwh"][dep_mask]
            depart_penalty = self._compute_departure_shortage_penalty(shortage_kwh)
            depart_energy_shortage_kwh = float(depart_penalty["depart_energy_shortage_kwh"])
            depart_shortage_soft_kwh = float(depart_penalty["depart_shortage_soft_kwh"])
            depart_shortage_mid_kwh = float(depart_penalty["depart_shortage_mid_kwh"])
            depart_shortage_hard_kwh = float(depart_penalty["depart_shortage_hard_kwh"])
            penalty_depart_energy_soft = float(depart_penalty["penalty_depart_energy_soft"])
            penalty_depart_energy_mid = float(depart_penalty["penalty_depart_energy_mid"])
            penalty_depart_energy_hard = float(depart_penalty["penalty_depart_energy_hard"])
            penalty_depart_energy = float(depart_penalty["penalty_depart_energy"])

        next_t = self.t + 1
        depart_risk = self._compute_departure_risk_penalty(next_t)
        depart_risk_energy_kwh = float(depart_risk["depart_risk_energy_kwh"])
        penalty_depart_risk = float(depart_risk["penalty_depart_risk"])
        depart_risk_vehicle_count = int(depart_risk["depart_risk_vehicle_count"])

        self.gt_power = p_gt_final
        self.gb_heat = hc_dispatch["p_gb_heat"]
        self.ac_cool = hc_dispatch["p_ac_cool"]
        self.ec_cool = hc_dispatch["p_ec_cool"]

        # ===== 9) 电侧总平衡：得到净电网功率 p_grid =====
        p_grid = dispatch_solution["p_grid"]
        grid_dispatch = dispatch_solution["grid_dispatch"]
        p_grid_buy = grid_dispatch["p_grid_buy"]
        p_grid_sell = grid_dispatch["p_grid_sell"]
        unmet_e = grid_dispatch["unmet_e"]
        surplus_e = grid_dispatch["surplus_e"]
        unmet_e_kwh = unmet_e * self.cfg.dt
        unmet_h_kwh = hc_dispatch["unmet_heat"] * self.cfg.dt
        unmet_c_kwh = hc_dispatch["unmet_cool"] * self.cfg.dt
        surplus_e_kwh = surplus_e * self.cfg.dt
        export_e_kwh = p_grid_sell * self.cfg.dt
        surplus_h_kwh = hc_dispatch["heat_curtail"] * self.cfg.dt
        surplus_c_kwh = hc_dispatch["cool_curtail"] * self.cfg.dt
        # low_value_energy_before_storage_prepare_kwh:
        # flexible EV / EES charging starts, the export + surplus energy that can still be absorbed.
        p_grid_without_storage_prepare = (
            p_grid
            - ev_dispatch["p_ev_flex_target_ch"]
            - ev_dispatch["p_ev_buffer_ch"]
            - ees_dispatch["p_ees_ch"]
        )
        grid_dispatch_without_storage_prepare = self._settle_grid_power(
            p_grid_without_storage_prepare
        )
        low_value_energy_before_storage_prepare_kwh = (
            grid_dispatch_without_storage_prepare["p_grid_sell"]
            + grid_dispatch_without_storage_prepare["surplus_e"]
        ) * self.cfg.dt

        # ===== 10) 计算各类惩罚项 =====
        # These penalty coefficients are applied to energy, keeping dt != 1 consistent.
        penalty_unserved_e = unmet_e_kwh * self.cfg.penalty_unserved_e
        penalty_unserved_h = unmet_h_kwh * self.cfg.penalty_unserved_h
        penalty_unserved_c = unmet_c_kwh * self.cfg.penalty_unserved_c
        penalty_surplus_e = surplus_e_kwh * self.cfg.penalty_surplus_e
        penalty_export_e = export_e_kwh * self.cfg.penalty_export_e
        ev_export_overlap_kwh = min(ev_dispatch["p_ev_dis"], p_grid_sell) * self.cfg.dt
        penalty_ev_export_guard = ev_export_overlap_kwh * self.cfg.penalty_ev_export_guard
        penalty_surplus_h = surplus_h_kwh * self.cfg.penalty_surplus_h
        penalty_surplus_c = surplus_c_kwh * self.cfg.penalty_surplus_c
        # ===== 11) 计算 EV 的正向引导奖励 =====
        storage_discharge_bonus = self._compute_storage_discharge_reward(
            grid_buy_price=exo["grid_buy_price"],
            p_grid=p_grid,
            p_grid_sell=p_grid_sell,
            p_ev_dis=ev_dispatch["p_ev_dis"],
            p_ees_dis=ees_dispatch["p_ees_dis"],
            ees_soc=ees_soc_before,
        )
        # storage_peak_shaved_kwh: rewarded EV + EES cooperative peak-shaving energy.
        storage_peak_shaved_kwh = float(storage_discharge_bonus["storage_peak_shaved_kwh"])
        reward_storage_discharge_bonus = float(
            storage_discharge_bonus["reward_storage_discharge_bonus"]
        )
        ev_peak_pressure_without_storage_kw = float(
            storage_discharge_bonus["storage_peak_pressure_without_storage_kw"]
        )
        ees_discharge_reward_weight = float(
            storage_discharge_bonus["ees_discharge_reward_weight"]
        )

        storage_charge_bonus = self._compute_storage_charge_preparation_reward(
            grid_buy_price=exo["grid_buy_price"],
            low_value_energy_before_storage_prepare_kwh=low_value_energy_before_storage_prepare_kwh,
            p_ev_flex_target_ch=ev_dispatch["p_ev_flex_target_ch"],
            p_ev_buffer_ch=ev_dispatch["p_ev_buffer_ch"],
            p_ees_ch=ees_dispatch["p_ees_ch"],
            ees_soc=ees_soc_before,
        )
        # storage_charge_rewarded_kwh: rewarded cooperative charging energy from EV buffer + weighted EES charge.
        storage_charge_rewarded_kwh = float(
            storage_charge_bonus["storage_charge_rewarded_kwh"]
        )
        # ees_charge_rewarded_kwh: weighted EES portion inside the cooperative charging reward.
        ees_charge_rewarded_kwh = float(
            storage_charge_bonus["ees_charge_rewarded_kwh"]
        )
        ev_flex_target_charge_kwh = float(
            storage_charge_bonus["ev_flex_target_charge_kwh"]
        )
        ev_buffer_charge_kwh = float(storage_charge_bonus["ev_buffer_charge_kwh"])
        low_value_charge_kwh = float(storage_charge_bonus["low_value_charge_kwh"])
        reward_storage_charge_bonus = float(
            storage_charge_bonus["reward_storage_charge_bonus"]
        )
        reward_ev_target_timing_bonus = float(
            storage_charge_bonus["reward_ev_target_timing_bonus"]
        )
        ees_charge_reward_weight = float(
            storage_charge_bonus["ees_charge_reward_weight"]
        )
        # GT 已直接映射到安全区间，因此实际安全裁剪惩罚恒为 0。

        # ===== 12) 成本项：购售电、燃气、退化、运维 =====
        cost_grid = (
            p_grid_buy * exo["grid_buy_price"] * self.cfg.dt
            - p_grid_sell * exo["grid_sell_price"] * self.cfg.dt
        )

        q_gt_gas = p_gt_final / self.cfg.gt_eta_e
        q_gb_gas = hc_dispatch["p_gb_heat"] / self.cfg.gb_eta_h
        cost_gas = (q_gt_gas + q_gb_gas) * exo["gas_price"] * self.cfg.dt

        cost_deg = ev_dispatch["p_ev_dis"] * self.cfg.c_deg * self.cfg.dt

        cost_om = (
            p_gt_final * self.cfg.om_gt
            + hc_dispatch["p_gb_heat"] * self.cfg.om_gb
            + hc_dispatch["p_whb_heat"] * self.cfg.om_whb
            + exo["wt"] * self.cfg.om_wt
            + exo["pv"] * self.cfg.om_pv
            + hc_dispatch["p_ec_cool"] * self.cfg.om_ec
            + hc_dispatch["p_ac_cool"] * self.cfg.om_ac
            + ees_dispatch["p_ees_dis"] * self.cfg.om_ees
        ) * self.cfg.dt

        # system_cost: direct operating cost before penalties and guide rewards.
        system_cost = cost_grid + cost_gas + cost_deg + cost_om

        # penalty_cost: hard-service and anti-export penalties that the policy should avoid.
        penalty_cost = (
            penalty_unserved_e
            + penalty_unserved_h
            + penalty_unserved_c
            + penalty_depart_energy
            + penalty_depart_risk
            + penalty_surplus_e
            + penalty_export_e
            + penalty_ev_export_guard
            + penalty_surplus_h
            + penalty_surplus_c
            + penalty_terminal_ees_soc
        )

        # guide_reward: cooperative storage guidance used to shape EV/EES timing behavior.
        guide_reward = (
            reward_storage_discharge_bonus
            + reward_storage_charge_bonus
            + reward_ev_target_timing_bonus
        )

        # 4) Final reward.
        reward_raw = -float(system_cost) - float(penalty_cost) + float(guide_reward)
        reward = reward_raw * self.cfg.reward_scale

        # ===== 13) 累计 episode 级统计量 =====
        self.episode_system_cost += float(system_cost)
        self.episode_cost_grid += float(cost_grid)
        self.episode_cost_gas += float(cost_gas)
        self.episode_cost_deg += float(cost_deg)
        self.episode_cost_om += float(cost_om)
        self.episode_penalty_cost += float(penalty_cost)
        self.episode_guide_reward += float(guide_reward)
        self.episode_reward_raw += float(reward_raw)
        self.episode_unserved_e_kwh += float(unmet_e_kwh)
        self.episode_unserved_h_kwh += float(unmet_h_kwh)
        self.episode_unserved_c_kwh += float(unmet_c_kwh)
        self.episode_surplus_e_kwh += float(surplus_e_kwh)
        self.episode_export_e_kwh += float(export_e_kwh)
        self.episode_surplus_h_kwh += float(surplus_h_kwh)
        self.episode_surplus_c_kwh += float(surplus_c_kwh)
        self.episode_penalty_unserved_e += float(penalty_unserved_e)
        self.episode_penalty_unserved_h += float(penalty_unserved_h)
        self.episode_penalty_unserved_c += float(penalty_unserved_c)
        self.episode_penalty_surplus_e += float(penalty_surplus_e)
        self.episode_penalty_surplus_h += float(penalty_surplus_h)
        self.episode_penalty_surplus_c += float(penalty_surplus_c)
        self.episode_penalty_export_e += float(penalty_export_e)
        self.episode_penalty_depart_energy += float(penalty_depart_energy)
        self.episode_penalty_depart_risk += float(penalty_depart_risk)
        self.episode_penalty_ev_export_guard += float(penalty_ev_export_guard)
        self.episode_terminal_ees_shortage_kwh += float(terminal_ees_shortage_kwh)
        self.episode_penalty_terminal_ees_soc += float(penalty_terminal_ees_soc)
        self.episode_final_ees_soc = float(self.ees_soc)
        self.episode_reward_storage_discharge += float(reward_storage_discharge_bonus)
        self.episode_reward_storage_charge += float(reward_storage_charge_bonus)
        self.episode_reward_ev_target_timing += float(reward_ev_target_timing_bonus)
        self.episode_storage_peak_shaved_kwh += float(storage_peak_shaved_kwh)
        self.episode_storage_charge_rewarded_kwh += float(storage_charge_rewarded_kwh)
        self.episode_ees_charge_rewarded_kwh += float(ees_charge_rewarded_kwh)
        self.episode_ev_flex_target_charge_kwh += float(ev_flex_target_charge_kwh)
        self.episode_ev_buffer_charge_kwh += float(ev_buffer_charge_kwh)
        self.episode_low_value_charge_kwh += float(low_value_charge_kwh)
        self.episode_gt_export_clip += float(gt_export_clip)
        if gt_export_clip > 1e-6:
            self.episode_gt_export_clip_steps += 1
        if not gt_safe_feasible:
            self.episode_gt_safe_infeasible_steps += 1

        # ===== 14) 时间推进，生成下一观测 =====
        self.t += 1
        terminated = self.t >= self.T
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(self.obs_dim, dtype=np.float32)

        # ===== 15) 构造调试与评估用的详细 info 字典 =====
        info = {
            "time_step": self.t - 1,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "a_ees": float(a_ees),
            "a_gt": float(a_gt),
            "a_ev_ch": float(a_ev_ch),
            "a_ev_dis": float(a_ev_dis),
            "elec_load": float(exo["elec_load"]),
            "heat_load": float(exo["heat_load"]),
            "cool_load": float(exo["cool_load"]),
            "pv": float(exo["pv"]),
            "wt": float(exo["wt"]),
            "grid_buy_price": float(exo["grid_buy_price"]),
            "grid_sell_price": float(exo["grid_sell_price"]),
            "gas_price": float(exo["gas_price"]),
            "cost_grid": float(cost_grid),
            "cost_gas": float(cost_gas),
            "cost_deg": float(cost_deg),
            "cost_om": float(cost_om),
            "penalty_cost": float(penalty_cost),
            "guide_reward": float(guide_reward),
            "system_cost": float(system_cost),
            "reward_raw": reward_raw,
            "reward_scaled": float(reward),
            "p_grid": float(p_grid),
            "p_grid_buy": float(p_grid_buy),
            "p_grid_sell": float(p_grid_sell),
            "unmet_e": float(unmet_e),
            "unmet_e_kwh": float(unmet_e_kwh),
            "unmet_h": float(hc_dispatch["unmet_heat"]),
            "unmet_h_kwh": float(unmet_h_kwh),
            "unmet_c": float(hc_dispatch["unmet_cool"]),
            "unmet_c_kwh": float(unmet_c_kwh),
            "surplus_e": float(surplus_e),
            "surplus_e_kwh": float(surplus_e_kwh),
            "export_e_kwh": float(export_e_kwh),
            "surplus_h": float(hc_dispatch["heat_curtail"]),
            "surplus_h_kwh": float(surplus_h_kwh),
            "surplus_c": float(hc_dispatch["cool_curtail"]),
            "surplus_c_kwh": float(surplus_c_kwh),
            "depart_energy_shortage_kwh": float(depart_energy_shortage_kwh),
            "grid_overflow": float(surplus_e),
            "grid_overflow_kwh": float(surplus_e_kwh),
            "penalty_unserved_e": float(penalty_unserved_e),
            "penalty_unserved_h": float(penalty_unserved_h),
            "penalty_unserved_c": float(penalty_unserved_c),
            "penalty_depart_energy": float(penalty_depart_energy),
            "penalty_depart_energy_soft": float(penalty_depart_energy_soft),
            "penalty_depart_energy_mid": float(penalty_depart_energy_mid),
            "penalty_depart_energy_hard": float(penalty_depart_energy_hard),
            "depart_shortage_soft_kwh": float(depart_shortage_soft_kwh),
            "depart_shortage_mid_kwh": float(depart_shortage_mid_kwh),
            "depart_shortage_hard_kwh": float(depart_shortage_hard_kwh),
            "depart_risk_energy_kwh": float(depart_risk_energy_kwh),
            "depart_risk_vehicle_count": int(depart_risk_vehicle_count),
            "penalty_depart_risk": float(penalty_depart_risk),
            "penalty_surplus_e": float(penalty_surplus_e),
            "penalty_export_e": float(penalty_export_e),
            "ev_export_overlap_kwh": float(ev_export_overlap_kwh),
            "penalty_ev_export_guard": float(penalty_ev_export_guard),
            "penalty_surplus_h": float(penalty_surplus_h),
            "penalty_surplus_c": float(penalty_surplus_c),
            "terminal_ees_required_soc": float(terminal_ees_required_soc),
            "terminal_ees_shortage_kwh": float(terminal_ees_shortage_kwh),
            "penalty_terminal_ees_soc": float(penalty_terminal_ees_soc),
            "episode_terminal_ees_shortage_kwh": float(self.episode_terminal_ees_shortage_kwh),
            "episode_penalty_terminal_ees_soc": float(self.episode_penalty_terminal_ees_soc),
            "ees_soc_init": float(self.ees_soc_episode_init),
            "final_ees_soc": float(self.ees_soc),
            "ees_terminal_soc_feasible": bool(
                float(self.ees_soc) + 1e-6 >= terminal_ees_required_soc
            ),
            "storage_peak_shaved_kwh": float(storage_peak_shaved_kwh),
            "storage_charge_rewarded_kwh": float(storage_charge_rewarded_kwh),
            "ees_charge_rewarded_kwh": float(ees_charge_rewarded_kwh),
            "ev_flex_target_charge_kwh": float(ev_flex_target_charge_kwh),
            "ev_buffer_charge_kwh": float(ev_buffer_charge_kwh),
            "low_value_charge_kwh": float(low_value_charge_kwh),
            "reward_storage_discharge_bonus": float(reward_storage_discharge_bonus),
            "reward_storage_charge_bonus": float(reward_storage_charge_bonus),
            "reward_ev_target_timing_bonus": float(reward_ev_target_timing_bonus),
            "ev_peak_pressure_without_storage_kw": float(ev_peak_pressure_without_storage_kw),
            "low_value_energy_before_storage_prepare_kwh": float(
                low_value_energy_before_storage_prepare_kwh
            ),
            "ees_discharge_reward_weight": float(ees_discharge_reward_weight),
            "ees_charge_reward_weight": float(ees_charge_reward_weight),
            "p_gt_safe_min": float(p_gt_safe_min),
            "p_gt_safe_max": float(p_gt_safe_max),
            "p_gt_export_clip": float(gt_export_clip),
            "gt_safe_feasible": bool(gt_safe_feasible),
            "p_gt": float(p_gt_final),
            "p_whb_heat": float(hc_dispatch["p_whb_heat"]),
            "p_gb_heat": float(hc_dispatch["p_gb_heat"]),
            "p_ac_cool": float(hc_dispatch["p_ac_cool"]),
            "p_ec_cool": float(hc_dispatch["p_ec_cool"]),
            "p_ec_elec_in": float(hc_dispatch["p_ec_elec_in"]),
            "p_ev_ch": float(ev_dispatch["p_ev_ch"]),
            "p_ev_rigid_ch": float(ev_dispatch["p_ev_rigid_ch"]),
            "p_ev_flex_target_ch": float(ev_dispatch["p_ev_flex_target_ch"]),
            "p_ev_buffer_ch": float(ev_dispatch["p_ev_buffer_ch"]),
            "p_ev_dis": float(ev_dispatch["p_ev_dis"]),
            "p_ev_safe_dis_cap": float(ev_now["p_flex_dis"]),
            "ev_safe_dis_vehicle_count": int(np.count_nonzero(ev_now["flex_discharge_mask"])),
            "ev_safe_lower_soc_mean": float(np.mean(ev_now["safe_discharge_lower_soc"]))
            if self.n_ev > 0
            else 0.0,
            "ev_safe_margin_kwh_sum": float(np.sum(ev_now["safe_discharge_margin_kwh"])),
            "p_ees_ch": float(ees_dispatch["p_ees_ch"]),
            "p_ees_dis": float(ees_dispatch["p_ees_dis"]),
            "ees_soc": float(self.ees_soc),
            "ees_soc_episode_init": float(self.ees_soc_episode_init),
            "ev_soc_mean": float(ev_soc_mean),
        }
        if terminated or truncated:
            info.update(
                {
                    "episode_system_cost": float(self.episode_system_cost),
                    "episode_cost_grid": float(self.episode_cost_grid),
                    "episode_cost_gas": float(self.episode_cost_gas),
                    "episode_cost_deg": float(self.episode_cost_deg),
                    "episode_cost_om": float(self.episode_cost_om),
                    "episode_penalty_cost": float(self.episode_penalty_cost),
                    "episode_guide_reward": float(self.episode_guide_reward),
                    "episode_reward_raw": float(self.episode_reward_raw),
                    "episode_unserved_e_kwh": float(self.episode_unserved_e_kwh),
                    "episode_unserved_h_kwh": float(self.episode_unserved_h_kwh),
                    "episode_unserved_c_kwh": float(self.episode_unserved_c_kwh),
                    "episode_surplus_e_kwh": float(self.episode_surplus_e_kwh),
                    "episode_export_e_kwh": float(self.episode_export_e_kwh),
                    "episode_surplus_h_kwh": float(self.episode_surplus_h_kwh),
                    "episode_surplus_c_kwh": float(self.episode_surplus_c_kwh),
                    # Legacy aliases kept for downstream scripts that still read the old names.
                    "episode_unserved_e": float(self.episode_unserved_e_kwh),
                    "episode_unserved_h": float(self.episode_unserved_h_kwh),
                    "episode_unserved_c": float(self.episode_unserved_c_kwh),
                    "episode_surplus_e": float(self.episode_surplus_e_kwh),
                    "episode_export_e": float(self.episode_export_e_kwh),
                    "episode_surplus_h": float(self.episode_surplus_h_kwh),
                    "episode_surplus_c": float(self.episode_surplus_c_kwh),
                    "episode_penalty_unserved_e": float(self.episode_penalty_unserved_e),
                    "episode_penalty_unserved_h": float(self.episode_penalty_unserved_h),
                    "episode_penalty_unserved_c": float(self.episode_penalty_unserved_c),
                    "episode_penalty_surplus_e": float(self.episode_penalty_surplus_e),
                    "episode_penalty_surplus_h": float(self.episode_penalty_surplus_h),
                    "episode_penalty_surplus_c": float(self.episode_penalty_surplus_c),
                    "episode_penalty_export_e": float(self.episode_penalty_export_e),
                    "episode_penalty_depart_energy": float(self.episode_penalty_depart_energy),
                    "episode_penalty_depart_risk": float(self.episode_penalty_depart_risk),
                    "episode_penalty_ev_export_guard": float(self.episode_penalty_ev_export_guard),
                    "episode_terminal_ees_shortage_kwh": float(self.episode_terminal_ees_shortage_kwh),
                    "episode_penalty_terminal_ees_soc": float(self.episode_penalty_terminal_ees_soc),
                    "episode_final_ees_soc": float(self.episode_final_ees_soc),
                    "episode_reward_storage_discharge": float(self.episode_reward_storage_discharge),
                    "episode_reward_storage_charge": float(self.episode_reward_storage_charge),
                    "episode_reward_ev_target_timing": float(self.episode_reward_ev_target_timing),
                    "episode_storage_peak_shaved_kwh": float(self.episode_storage_peak_shaved_kwh),
                    "episode_storage_charge_rewarded_kwh": float(self.episode_storage_charge_rewarded_kwh),
                    "episode_ees_charge_rewarded_kwh": float(self.episode_ees_charge_rewarded_kwh),
                    "episode_ev_flex_target_charge_kwh": float(self.episode_ev_flex_target_charge_kwh),
                    "episode_ev_buffer_charge_kwh": float(self.episode_ev_buffer_charge_kwh),
                    "episode_low_value_charge_kwh": float(self.episode_low_value_charge_kwh),
                    "episode_gt_export_clip": float(self.episode_gt_export_clip),
                    "episode_gt_export_clip_steps": int(self.episode_gt_export_clip_steps),
                    "episode_gt_safe_infeasible_steps": int(self.episode_gt_safe_infeasible_steps),
                }
            )
        self.last_info = info

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # render()
    # 仅在 human 模式下打印上一时刻 info，用于调试。
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "human":
            print(self.last_info)

    # ------------------------------------------------------------------
    # _ensure_runtime_state_initialized()
    # 鍏紑鎺ュ彛鍦?reset() 涔嬪墠涓嶅簲渚濊禆 EV runtime state锛屽惁鍒欏強鏃╁け璐ャ€?
    # ------------------------------------------------------------------
    def _ensure_runtime_state_initialized(self) -> None:
        if self.ev_soc is None:
            raise RuntimeError(
                "Environment state is not initialized. Call reset() before step() or requesting observations."
            )

    # ------------------------------------------------------------------
    # _validate_config()
    # 妫€鏌?__init__ / _get_obs() / step() 闅愬惈渚濊禆鐨勫叧閿?cfg 鍚堟硶鎬с€?
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        integer_fields = {
            "episode_length": 1,
            "future_horizon": 0,
            "exogenous_future_horizon": 0,
        }
        for name, min_value in integer_fields.items():
            value = getattr(self.cfg, name)
            if not isinstance(value, (int, np.integer)):
                raise TypeError(f"cfg.{name} must be an integer, got {type(value).__name__}")
            if int(value) < min_value:
                raise ValueError(f"cfg.{name} must be >= {min_value}, got {value}")

        if not np.isfinite(self.cfg.dt) or self.cfg.dt <= 0.0:
            raise ValueError(f"cfg.dt must be positive, got {self.cfg.dt}")

        price_profile_fields = [
            "grid_buy_price_24",
            "grid_sell_price_24",
            "gas_price_24",
        ]
        for name in price_profile_fields:
            arr = np.asarray(getattr(self.cfg, name), dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(f"cfg.{name} must be 1-D, got shape {arr.shape}")
            if arr.shape[0] != 24:
                raise ValueError(f"cfg.{name} must have length 24, got {arr.shape[0]}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"cfg.{name} must contain only finite values")
            if np.any(arr < 0.0):
                raise ValueError(f"cfg.{name} must be non-negative")

        positive_ref_fields = [
            "price_max",
            "elec_load_max",
            "heat_load_max",
            "cool_load_max",
            "pv_max",
            "wt_max",
            "ees_p_max",
            "ees_e_cap",
            "gt_p_max",
            "gb_h_max",
            "ac_c_max",
            "ec_c_max",
        ]
        for name in positive_ref_fields:
            value = float(getattr(self.cfg, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"cfg.{name} must be positive, got {value}")

        if not np.isfinite(self.cfg.ees_soc_init):
            raise ValueError(f"cfg.ees_soc_init must be finite, got {self.cfg.ees_soc_init}")
        if not np.isfinite(self.cfg.ees_terminal_soc_tolerance) or self.cfg.ees_terminal_soc_tolerance < 0.0:
            raise ValueError(
                "cfg.ees_terminal_soc_tolerance must be finite and non-negative, "
                f"got {self.cfg.ees_terminal_soc_tolerance}"
            )
        if not np.isfinite(self.cfg.penalty_ees_terminal_soc) or self.cfg.penalty_ees_terminal_soc < 0.0:
            raise ValueError(
                "cfg.penalty_ees_terminal_soc must be finite and non-negative, "
                f"got {self.cfg.penalty_ees_terminal_soc}"
            )

        if not (0.0 <= self.cfg.ees_soc_min < self.cfg.ees_soc_max <= 1.0):
            raise ValueError(
                "EES SOC bounds must satisfy 0 <= ees_soc_min < ees_soc_max <= 1."
            )
        if not (0.0 <= self.cfg.ev_soc_min < self.cfg.ev_soc_max <= 1.0):
            raise ValueError(
                "EV SOC bounds must satisfy 0 <= ev_soc_min < ev_soc_max <= 1."
            )

    # ------------------------------------------------------------------
    # _validate_inputs()
    # 检查 ts_data / ev_data 字段是否齐全，并补充电价与气价序列。
    # 这里也决定了环境实际使用的分时电价。
    # ------------------------------------------------------------------
    def _validate_inputs(self):
        self._validate_config()

        required_ts = ["elec_load", "heat_load", "cool_load", "pv", "wt"]
        for k in required_ts:
            if k not in self.ts:
                raise KeyError(f"Missing timeseries field: {k}")
            arr = np.asarray(self.ts[k], dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(f"Timeseries field {k} must be 1-D, got shape {arr.shape}")
            if arr.shape[0] < self.T:
                raise ValueError(
                    f"Timeseries field {k} length {arr.shape[0]} is shorter than episode_length {self.T}"
                )
            if not np.all(np.isfinite(arr[: self.T])):
                raise ValueError(
                    f"Timeseries field {k} contains non-finite values within the episode window"
                )
            self.ts[k] = arr.copy()

        buy_24 = np.asarray(self.cfg.grid_buy_price_24, dtype=np.float32)
        sell_24 = np.asarray(self.cfg.grid_sell_price_24, dtype=np.float32)
        gas_24 = np.asarray(self.cfg.gas_price_24, dtype=np.float32)
        self.ts["grid_buy_price"] = np.resize(buy_24, self.T).astype(np.float32)
        self.ts["grid_sell_price"] = np.resize(sell_24, self.T).astype(np.float32)
        self.ts["gas_price"] = np.resize(gas_24, self.T).astype(np.float32)

        for k in ["grid_buy_price", "grid_sell_price", "gas_price"]:
            if self.ts[k].shape[0] < self.T:
                raise ValueError(
                    f"Generated timeseries field {k} length {self.ts[k].shape[0]} is shorter than episode_length {self.T}"
                )

        required_ev = [
            "arr_step", "dep_step", "init_soc", "target_soc",
            "cap_kwh", "p_ch_max_kw", "p_dis_max_kw", "v2g_flag",
        ]
        for k in required_ev:
            if k not in self.ev:
                raise KeyError(f"Missing EV field: {k}")

        ev_length = None
        int_ev_fields = ["arr_step", "dep_step"]
        for k in int_ev_fields:
            arr = np.asarray(self.ev[k], dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError(f"EV field {k} must be 1-D, got shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"EV field {k} contains non-finite values")
            if not np.all(np.equal(arr, np.round(arr))):
                raise ValueError(f"EV field {k} must contain integer-valued steps")
            if ev_length is None:
                ev_length = int(arr.shape[0])
            elif arr.shape[0] != ev_length:
                raise ValueError(
                    f"EV field {k} length {arr.shape[0]} does not match arr_step length {ev_length}"
                )
            self.ev[k] = np.round(arr).astype(np.int32)

        float_ev_fields = ["init_soc", "target_soc", "cap_kwh", "p_ch_max_kw", "p_dis_max_kw"]
        for k in float_ev_fields:
            arr = np.asarray(self.ev[k], dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(f"EV field {k} must be 1-D, got shape {arr.shape}")
            if ev_length is None:
                ev_length = int(arr.shape[0])
            elif arr.shape[0] != ev_length:
                raise ValueError(
                    f"EV field {k} length {arr.shape[0]} does not match arr_step length {ev_length}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"EV field {k} contains non-finite values")
            self.ev[k] = arr.copy()

        v2g = np.asarray(self.ev["v2g_flag"])
        if v2g.ndim != 1:
            raise ValueError(f"EV field v2g_flag must be 1-D, got shape {v2g.shape}")
        if ev_length is None:
            ev_length = int(v2g.shape[0])
        elif v2g.shape[0] != ev_length:
            raise ValueError(
                f"EV field v2g_flag length {v2g.shape[0]} does not match arr_step length {ev_length}"
            )
        if not np.all(np.isin(v2g, [0, 1, False, True])):
            raise ValueError("EV field v2g_flag must contain only 0/1 or bool values")
        self.ev["v2g_flag"] = v2g.astype(np.bool_)

        if np.any(self.ev["dep_step"] < self.ev["arr_step"]):
            raise ValueError("Each EV must satisfy dep_step >= arr_step")
        if np.any(self.ev["cap_kwh"] <= 0.0):
            raise ValueError("EV field cap_kwh must be strictly positive")
        if np.any(self.ev["p_ch_max_kw"] < 0.0):
            raise ValueError("EV field p_ch_max_kw must be non-negative")
        if np.any(self.ev["p_dis_max_kw"] < 0.0):
            raise ValueError("EV field p_dis_max_kw must be non-negative")
        if np.any(
            (self.ev["init_soc"] < self.cfg.ev_soc_min)
            | (self.ev["init_soc"] > self.cfg.ev_soc_max)
        ):
            raise ValueError(
                "EV field init_soc must stay within [cfg.ev_soc_min, cfg.ev_soc_max]"
            )
        if np.any(
            (self.ev["target_soc"] < self.cfg.ev_soc_min)
            | (self.ev["target_soc"] > self.cfg.ev_soc_max)
        ):
            raise ValueError(
                "EV field target_soc must stay within [cfg.ev_soc_min, cfg.ev_soc_max]"
            )

    # ------------------------------------------------------------------
    # _get_exogenous()
    # 读取时刻 t 的外生量：电/热/冷负荷、PV、WT、电价、气价。
    # ------------------------------------------------------------------
    def _get_exogenous(self, t: int) -> Dict[str, float]:
        return {
            "elec_load": float(self.ts["elec_load"][t]),
            "heat_load": float(self.ts["heat_load"][t]),
            "cool_load": float(self.ts["cool_load"][t]),
            "pv": float(self.ts["pv"][t]),
            "wt": float(self.ts["wt"][t]),
            "grid_buy_price": float(self.ts["grid_buy_price"][t]),
            "grid_sell_price": float(self.ts["grid_sell_price"][t]),
            "gas_price": float(self.ts["gas_price"][t]),
        }

    # ------------------------------------------------------------------
    # _norm()
    # 简单归一化到 [0, 1]，供观测空间使用。
    # ------------------------------------------------------------------
    def _norm(self, x: float, ref: float) -> float:
        return float(np.clip(x / max(ref, 1e-6), 0.0, 1.0))

    # ------------------------------------------------------------------
    # _get_obs()
    # 拼接观测向量，顺序必须与训练脚本中的网络输入严格一致：
    # 1) 当前 20 维：
    #    [0]  grid_buy_price
    #    [1]  grid_sell_price
    #    [2]  gas_price
    #    [3]  当前时间索引 t_idx
    #    [4]  elec_load
    #    [5]  heat_load
    #    [6]  cool_load
    #    [7]  pv
    #    [8]  wt
    #    [9]  ees_soc
    #    [10] ees 当前充电功率状态
    #    [11] ees 当前放电功率状态
    #    [12] gt_power
    #    [13] gb_heat
    #    [14] ac_cool
    #    [15] ec_cool
    #    [16] 当前 EV rigid 边界 p_rigid
    #    [17] 当前 EV flex_target_ch 边界 p_flex_target_ch
    #    [18] 当前 EV buffer_ch 边界 p_buffer_ch
    #    [19] 当前 EV flex_dis 边界 p_flex_dis
    # 2) 未来 EV 4 组边界：
    #    - future_rigid
    #    - future_flex_target_ch
    #    - future_buffer_ch
    #    - future_flex_dis
    # 3) 未来外生量 5 组预测：
    #    - elec_load
    #    - heat_load
    #    - cool_load
    #    - pv
    #    - wt
    # 纠错时重点检查：obs_dim 与实际拼接长度、顺序是否一致。
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        self._ensure_runtime_state_initialized()
        t_idx = min(self.t, self.T - 1)
        exo = self._get_exogenous(t_idx)
        ev_now = self._compute_ev_boundaries(t_idx, self.ev_soc)
        (
            future_rigid,
            future_flex_target_ch,
            future_buffer_ch,
            future_flex_dis,
        ) = self._forecast_ev_boundaries(
            t_idx, self.ev_soc
        )
        future_exogenous = self._forecast_exogenous_signals(t_idx)

        max_ev_charge_agg = max(float(np.sum(self.ev["p_ch_max_kw"], dtype=np.float64)), 1e-6)
        max_ev_discharge_agg = max(
            float(np.sum(self.ev["p_dis_max_kw"], dtype=np.float64)),
            1e-6,
        )
        ees_p_ch_state = max(-self.ees_power, 0.0)
        ees_p_dis_state = max(self.ees_power, 0.0)

        # 先拼接当前 20 维，再依次拼接未来 EV 4 组边界和未来外生量 5 组预测。
        obs = [
            self._norm(exo["grid_buy_price"], self.cfg.price_max),
            self._norm(exo["grid_sell_price"], self.cfg.price_max),
            self._norm(exo["gas_price"], self.cfg.price_max),
            t_idx / max(self.T - 1, 1),
            self._norm(exo["elec_load"], self.cfg.elec_load_max),
            self._norm(exo["heat_load"], self.cfg.heat_load_max),
            self._norm(exo["cool_load"], self.cfg.cool_load_max),
            self._norm(exo["pv"], self.cfg.pv_max),
            self._norm(exo["wt"], self.cfg.wt_max),
            float(np.clip(self.ees_soc, 0.0, 1.0)),
            self._norm(ees_p_ch_state, self.cfg.ees_p_max),
            self._norm(ees_p_dis_state, self.cfg.ees_p_max),
            self._norm(self.gt_power, self.cfg.gt_p_max),
            self._norm(self.gb_heat, self.cfg.gb_h_max),
            self._norm(self.ac_cool, self.cfg.ac_c_max),
            self._norm(self.ec_cool, self.cfg.ec_c_max),
            self._norm(ev_now["p_rigid"], max_ev_charge_agg),
            self._norm(ev_now["p_flex_target_ch"], max_ev_charge_agg),
            self._norm(ev_now["p_buffer_ch"], max_ev_charge_agg),
            self._norm(ev_now["p_flex_dis"], max_ev_discharge_agg),
        ]
        obs.extend([self._norm(x, max_ev_charge_agg) for x in future_rigid])
        obs.extend([self._norm(x, max_ev_charge_agg) for x in future_flex_target_ch])
        obs.extend([self._norm(x, max_ev_charge_agg) for x in future_buffer_ch])
        obs.extend([self._norm(x, max_ev_discharge_agg) for x in future_flex_dis])
        obs.extend(
            [self._norm(x, self.cfg.elec_load_max) for x in future_exogenous["elec_load"]]
        )
        obs.extend(
            [self._norm(x, self.cfg.heat_load_max) for x in future_exogenous["heat_load"]]
        )
        obs.extend(
            [self._norm(x, self.cfg.cool_load_max) for x in future_exogenous["cool_load"]]
        )
        obs.extend([self._norm(x, self.cfg.pv_max) for x in future_exogenous["pv"]])
        obs.extend([self._norm(x, self.cfg.wt_max) for x in future_exogenous["wt"]])

        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # _compute_ev_boundaries()
    # 按每辆车的到达/离开时间、目标 SOC、当前 SOC，计算：
    # - rigid 必充部分
    # - 可柔性补至目标 SOC 的充电部分
    # - 允许进入 V2G 缓冲补能流程的车辆集合
    # - flex_dis 可放电部分（仅允许动用高于动态安全放电下界的能量）
    # 这里刻意允许 V2G 车同时属于 flex_target_charge_mask 和 buffer_charge_mask：
    # 它表示“该车在本步内可以先补目标 SOC；若该步内仍有剩余可行充电能力，
    # 再继续往目标上方的 V2G 缓冲区补电”。
    # 因此，buffer_charge_mask 不是“已达到 target 的车辆集合”，而是“允许参与缓冲补能的候选集合”。
    # 动态安全放电下界由当前步执行后、未来最大可恢复充电能力决定，
    # 用于在保障离站目标的前提下释放更多 V2G 柔性。
    # 这是 EV 行为逻辑的基础函数。若 EV 充放电异常，优先检查这里。
    # ------------------------------------------------------------------
    def _compute_ev_boundaries(self, t: int, ev_soc: np.ndarray) -> Dict[str, Any]:
        arr = self.ev["arr_step"]
        dep = self.ev["dep_step"]
        target = self.ev["target_soc"]
        cap = self.ev["cap_kwh"]
        p_ch_max = self.ev["p_ch_max_kw"]
        p_dis_max = self.ev["p_dis_max_kw"]
        v2g = self.ev["v2g_flag"].astype(bool)
        buffer_upper = np.where(
            v2g,
            np.minimum(target + self.cfg.ev_v2g_buffer_soc_margin, self.cfg.ev_soc_max),
            target,
        )

        # active: 当前时刻已到站且未离站的车辆
        active = (arr <= t) & (t < dep)
        need_to_target = np.maximum(target - ev_soc, 0.0) * cap
        need_to_buffer_upper = np.maximum(buffer_upper - ev_soc, 0.0) * cap
        time_left_h = np.maximum(dep - t, 0) * self.cfg.dt

        laxity_target = time_left_h - need_to_target / np.maximum(p_ch_max * self.cfg.ev_eta_ch, 1e-6)
        time_after_this_step_h = np.maximum(dep - (t + 1), 0) * self.cfg.dt
        recoverable_time_h = np.maximum(
            time_after_this_step_h - self.cfg.ev_safe_discharge_reserve_h,
            0.0,
        )
        recoverable_energy_kwh = recoverable_time_h * p_ch_max * self.cfg.ev_eta_ch
        safe_discharge_lower_soc = np.maximum(
            self.cfg.ev_soc_min,
            target - recoverable_energy_kwh / np.maximum(cap, 1e-6),
        )
        safe_discharge_margin_soc = np.maximum(ev_soc - safe_discharge_lower_soc, 0.0)
        safe_discharge_margin_kwh = safe_discharge_margin_soc * cap

        rigid_charge_mask = active & (ev_soc < target) & (laxity_target <= self.cfg.dt)
        flex_target_charge_mask = active & (ev_soc < target) & (laxity_target > self.cfg.dt)
        # 注意：这里允许与 flex_target_charge_mask 重叠。重叠仅表示该车在本步内
        # 既可能需要“先补 target”，也可能在补完 target 后继续补缓冲区。
        buffer_charge_mask = active & v2g & (ev_soc < buffer_upper) & (laxity_target > self.cfg.dt)
        # EV 放电仅允许动用高于动态安全放电下界的能量，以保证出行需求。
        flex_discharge_mask = (
            active
            & v2g
            & (laxity_target > self.cfg.dt)
            & (ev_soc > safe_discharge_lower_soc + self.cfg.ev_safe_discharge_soc_epsilon)
        )

        p_ch_to_target_feasible = np.minimum(
            p_ch_max,
            np.maximum(need_to_target / np.maximum(self.cfg.ev_eta_ch * self.cfg.dt, 1e-6), 0.0)
        )
        p_ch_to_buffer_upper_feasible = np.minimum(
            p_ch_max,
            np.maximum(need_to_buffer_upper / np.maximum(self.cfg.ev_eta_ch * self.cfg.dt, 1e-6), 0.0)
        )
        # 在单个 step 内，“缓冲充电”只计算 target 以上的额外可行部分。
        p_ch_above_target_buffer_feasible = np.maximum(
            p_ch_to_buffer_upper_feasible - p_ch_to_target_feasible,
            0.0,
        )

        p_dis_feasible = np.minimum(
            p_dis_max,
            np.maximum(
                safe_discharge_margin_kwh * self.cfg.ev_eta_dis / np.maximum(self.cfg.dt, 1e-6),
                0.0,
            ),
        )

        p_rigid = float(np.sum(p_ch_to_target_feasible[rigid_charge_mask]))
        p_flex_target_ch = float(np.sum(p_ch_to_target_feasible[flex_target_charge_mask]))
        p_buffer_ch = float(np.sum(p_ch_above_target_buffer_feasible[buffer_charge_mask]))
        p_flex_ch = p_flex_target_ch + p_buffer_ch
        p_flex_dis = float(np.sum(p_dis_feasible[flex_discharge_mask]))

        return {
            "active_mask": active,
            "rigid_charge_mask": rigid_charge_mask,
            "flex_target_charge_mask": flex_target_charge_mask,
            "buffer_charge_mask": buffer_charge_mask,
            "flex_discharge_mask": flex_discharge_mask,
            "laxity_target": laxity_target,
            "buffer_upper_soc": buffer_upper,
            "safe_discharge_lower_soc": safe_discharge_lower_soc,
            "safe_discharge_margin_soc": safe_discharge_margin_soc,
            "safe_discharge_margin_kwh": safe_discharge_margin_kwh,
            "p_ch_to_target_feasible": p_ch_to_target_feasible,
            "p_ch_to_buffer_upper_feasible": p_ch_to_buffer_upper_feasible,
            "p_ch_above_target_buffer_feasible": p_ch_above_target_buffer_feasible,
            "p_dis_feasible": p_dis_feasible,
            "p_rigid": p_rigid,
            "p_flex_target_ch": p_flex_target_ch,
            "p_buffer_ch": p_buffer_ch,
            "p_flex_ch": p_flex_ch,
            "p_flex_dis": p_flex_dis,
        }

    # ------------------------------------------------------------------
    # _forecast_ev_boundaries()
    # 基于当前 EV SOC，向后滚动查看未来若干步的 EV 可调边界，
    # 供策略网络提前感知未来出行约束。
    # 注意：这里没有滚动更新未来 SOC，只是“在当前 SOC 下估计未来边界”。
    # ------------------------------------------------------------------
    def _forecast_ev_boundaries(self, t: int, ev_soc: np.ndarray):
        future_rigid = []
        future_flex_target_ch = []
        future_buffer_ch = []
        future_flex_dis = []

        for h in range(1, self.cfg.future_horizon + 1):
            tau = t + h
            if tau >= self.T:
                future_rigid.append(0.0)
                future_flex_target_ch.append(0.0)
                future_buffer_ch.append(0.0)
                future_flex_dis.append(0.0)
            else:
                ev_tau = self._compute_ev_boundaries(tau, ev_soc)
                future_rigid.append(ev_tau["p_rigid"])
                future_flex_target_ch.append(ev_tau["p_flex_target_ch"])
                future_buffer_ch.append(ev_tau["p_buffer_ch"])
                future_flex_dis.append(ev_tau["p_flex_dis"])

        return (
            future_rigid,
            future_flex_target_ch,
            future_buffer_ch,
            future_flex_dis,
        )

    # ------------------------------------------------------------------
    # _forecast_exogenous_signals()
    # 给出未来若干步电/热/冷负荷与 PV/WT 的外生预测量。
    # 当前实现直接读取已知时序数据，不含预测误差。
    # ------------------------------------------------------------------
    def _forecast_exogenous_signals(self, t: int) -> Dict[str, list[float]]:
        future = {
            "elec_load": [],
            "heat_load": [],
            "cool_load": [],
            "pv": [],
            "wt": [],
        }

        for h in range(1, self.cfg.exogenous_future_horizon + 1):
            tau = t + h
            if tau >= self.T:
                for key in future:
                    future[key].append(0.0)
            else:
                exo_tau = self._get_exogenous(tau)
                for key in future:
                    future[key].append(exo_tau[key])

        return future

    # ------------------------------------------------------------------
    # _dispatch_ev()
    # 把“聚合层动作”分解到每辆 EV：
    # - rigid 车辆先满足刚性充电；
    # - 剩余充电动作优先补目标 SOC，再补 V2G 缓冲区；
    # - 若某辆 V2G 车同时属于 flex_target_charge_mask 和 buffer_charge_mask，则含义是：
    #   本步内先给它分配“补到 target”的部分，再在仍有剩余充电指令时，
    #   继续分配“target 以上的缓冲部分”；
    # - 放电动作只允许在高于动态安全放电下界的 V2G 车辆上发生。
    #   该下界由未来最大可恢复充电能力决定，用于在保障离站目标的前提下释放更多柔性。
    # 纠错时要看：充电优先级、同车是否充放电冲突、laxity 排序是否合理。
    # ------------------------------------------------------------------
    def _dispatch_ev(self, a_ev_ch: float, a_ev_dis: float, ev_state: Dict[str, Any]) -> Dict[str, Any]:
        n = self.n_ev
        p_ch_each = np.zeros(n, dtype=np.float32)
        p_dis_each = np.zeros(n, dtype=np.float32)

        p_rigid = ev_state["p_rigid"]
        p_flex_ch = ev_state["p_flex_ch"]
        p_flex_dis = ev_state["p_flex_dis"]

        # 聚合动作先变成“目标总充电/放电功率”，再分配到单车。
        p_ch_cmd = p_rigid + a_ev_ch * p_flex_ch
        p_dis_cmd = a_ev_dis * p_flex_dis

        rigid_idx = np.where(ev_state["rigid_charge_mask"])[0]
        flex_target_ch_idx = np.where(ev_state["flex_target_charge_mask"])[0]
        buffer_charge_idx = np.where(ev_state["buffer_charge_mask"])[0]
        flex_dis_idx = np.where(ev_state["flex_discharge_mask"])[0]

        laxity = ev_state["laxity_target"]
        p_ch_to_target_feasible = ev_state["p_ch_to_target_feasible"]
        p_ch_above_target_buffer_feasible = ev_state["p_ch_above_target_buffer_feasible"]
        p_dis_feasible = ev_state["p_dis_feasible"]
        safe_discharge_margin_soc = ev_state["safe_discharge_margin_soc"]

        for i in rigid_idx:
            p_ch_each[i] = p_ch_to_target_feasible[i]

        rem_ch = max(p_ch_cmd - np.sum(p_ch_each), 0.0)
        flex_target_sorted = flex_target_ch_idx[np.argsort(laxity[flex_target_ch_idx])]
        for i in flex_target_sorted:
            if rem_ch <= 1e-6:
                break
            p = min(p_ch_to_target_feasible[i], rem_ch)
            p_ch_each[i] += p
            rem_ch -= p

        # 缓冲补能是“目标之上的附加部分”，优先分给时间更宽松的车。
        buffer_sorted = buffer_charge_idx[np.argsort(-laxity[buffer_charge_idx])]
        for i in buffer_sorted:
            if rem_ch <= 1e-6:
                break
            assigned_charge_so_far = p_ch_each[i]
            already_assigned_above_target = max(
                assigned_charge_so_far - p_ch_to_target_feasible[i],
                0.0,
            )
            buffer_headroom_after_target = max(
                p_ch_above_target_buffer_feasible[i] - already_assigned_above_target,
                0.0,
            )
            if buffer_headroom_after_target <= 1e-6:
                continue
            p = min(buffer_headroom_after_target, rem_ch)
            p_ch_each[i] += p
            rem_ch -= p

        rem_dis = p_dis_cmd
        flex_dis_sorted = sorted(
            flex_dis_idx.tolist(),
            key=lambda i: (-safe_discharge_margin_soc[i], -laxity[i], i),
        )
        for i in flex_dis_sorted:
            if rem_dis <= 1e-6:
                break
            if p_ch_each[i] > 1e-6:
                continue
            p = min(p_dis_feasible[i], rem_dis)
            p_dis_each[i] += p
            rem_dis -= p

        return {
            "p_ch_each": p_ch_each,
            "p_dis_each": p_dis_each,
            "p_ev_ch": float(np.sum(p_ch_each)),
            "p_ev_dis": float(np.sum(p_dis_each)),
            "p_ev_rigid_ch": float(np.sum(p_ch_each[rigid_idx])),
            "p_ev_flex_target_ch": float(
                max(
                    np.sum(np.minimum(p_ch_each, p_ch_to_target_feasible)) - np.sum(p_ch_each[rigid_idx]),
                    0.0,
                )
            ),
            "p_ev_buffer_ch": float(np.sum(np.maximum(p_ch_each - p_ch_to_target_feasible, 0.0))),
        }

    # ------------------------------------------------------------------
    # _clip_ev_discharge_to_local_absorption()
    # 若 EV 放电会把系统推向超出容差的外送，则按比例裁剪放电，
    # 让 EV 更聚焦于本地削峰而不是叠加外送。
    # ------------------------------------------------------------------
    def _clip_ev_discharge_to_local_absorption(
        self,
        *,
        ev_dispatch: Dict[str, Any],
        p_grid: float,
    ) -> Dict[str, Any]:
        p_ev_dis = float(ev_dispatch["p_ev_dis"])
        if p_ev_dis <= 1e-6:
            return ev_dispatch

        export_tol_kw = max(float(self.cfg.ev_peak_export_tolerance_kw), 0.0)
        p_ev_dis_cap_local = max(p_grid + p_ev_dis + export_tol_kw, 0.0)
        p_ev_dis_cap_local = min(p_ev_dis_cap_local, p_ev_dis)

        if p_ev_dis <= p_ev_dis_cap_local + 1e-6:
            return ev_dispatch

        scale = p_ev_dis_cap_local / max(p_ev_dis, 1e-6)
        p_dis_each = np.asarray(ev_dispatch["p_dis_each"], dtype=np.float32).copy()
        p_dis_each *= np.float32(scale)

        clipped = dict(ev_dispatch)
        clipped["p_dis_each"] = p_dis_each
        clipped["p_ev_dis"] = float(np.sum(p_dis_each, dtype=np.float64))
        return clipped

    # ------------------------------------------------------------------
    # _dispatch_ees()
    # Hard-map the scalar EES action to the currently feasible charge/discharge power.
    # p_ch_cap / p_dis_cap are the per-step maximum feasible charge/discharge powers
    # under the current SOC and power rating, so the action never produces an invalid request.
    # ------------------------------------------------------------------
    def _dispatch_ees(self, a_ees: float) -> Dict[str, float]:
        # a_ees > 0 表示充电；a_ees < 0 表示放电。
        # 动作幅值不再对应固定额定功率，而是对应“当前时刻可行充/放功率”的比例。
        e_room = max(self.cfg.ees_soc_max - self.ees_soc, 0.0) * self.cfg.ees_e_cap
        e_avail = max(self.ees_soc - self.cfg.ees_soc_min, 0.0) * self.cfg.ees_e_cap

        p_ch_cap = float(
            min(
                self.cfg.ees_p_max,
                e_room / max(self.cfg.ees_eta_ch * self.cfg.dt, 1e-6),
            )
        )
        p_dis_cap = float(
            min(
                self.cfg.ees_p_max,
                e_avail * self.cfg.ees_eta_dis / max(self.cfg.dt, 1e-6),
            )
        )

        if a_ees >= 0.0:
            p_ees_ch = float(a_ees * p_ch_cap)
            p_ees_dis = 0.0
        else:
            p_ees_ch = 0.0
            p_ees_dis = float((-a_ees) * p_dis_cap)

        return {
            "p_ees_ch": float(p_ees_ch),
            "p_ees_dis": float(p_ees_dis),
        }

    # ------------------------------------------------------------------
    # _compute_net_grid_power()
    # 在给定 GT / 热冷 / EV / EES 执行功率后，结算净电网功率。
    # ------------------------------------------------------------------
    def _compute_net_grid_power(
        self,
        *,
        exo: Dict[str, float],
        p_gt: float,
        hc_dispatch: Dict[str, float],
        ev_dispatch: Dict[str, Any],
        ees_dispatch: Dict[str, float],
    ) -> float:
        return float(
            exo["elec_load"]
            + hc_dispatch["p_ec_elec_in"]
            + ev_dispatch["p_ev_ch"]
            + ees_dispatch["p_ees_ch"]
            - ev_dispatch["p_ev_dis"]
            - ees_dispatch["p_ees_dis"]
            - exo["pv"]
            - exo["wt"]
            - p_gt
        )

    # ------------------------------------------------------------------
    # _resolve_step_dispatch()
    # 在给定 EV / EES 执行值后，串联求解 GT 安全带、冷热调度和电网结算。
    # 这里返回的是“当前 EV / EES 条件下的一次完整求解结果”。
    # 若 step() 后续对 EV 放电做了本地吸收裁剪，则会基于更新后的 EV 再次调用本函数，
    # 因而整步最终结果应以裁剪后的那次重算结果为准。
    # ------------------------------------------------------------------
    def _resolve_step_dispatch(
        self,
        *,
        exo: Dict[str, float],
        a_gt: float,
        ev_dispatch: Dict[str, Any],
        ees_dispatch: Dict[str, float],
    ) -> Dict[str, Any]:
        gt_safe = self._find_gt_min_safe(
            exo=exo,
            p_ev_ch=ev_dispatch["p_ev_ch"],
            p_ev_dis=ev_dispatch["p_ev_dis"],
            p_ees_ch=ees_dispatch["p_ees_ch"],
            p_ees_dis=ees_dispatch["p_ees_dis"],
        )
        p_gt_final = self._map_gt_action_to_band(
            a_gt=a_gt,
            lower=gt_safe["p_gt_safe_min"],
            upper=gt_safe["p_gt_safe_max"],
        )

        hc_dispatch = self._dispatch_heat_cold(
            exo=exo,
            p_gt=p_gt_final,
            p_ev_ch=ev_dispatch["p_ev_ch"],
            p_ev_dis=ev_dispatch["p_ev_dis"],
            p_ees_ch=ees_dispatch["p_ees_ch"],
            p_ees_dis=ees_dispatch["p_ees_dis"],
        )
        p_grid = self._compute_net_grid_power(
            exo=exo,
            p_gt=p_gt_final,
            hc_dispatch=hc_dispatch,
            ev_dispatch=ev_dispatch,
            ees_dispatch=ees_dispatch,
        )
        # gt_export_clip: final-step electric export overflow beyond the tolerated margin.
        gt_export_clip = max(-p_grid - self.cfg.gt_export_margin, 0.0)
        grid_dispatch = self._settle_grid_power(p_grid)

        return {
            "gt_safe": gt_safe,
            "p_gt_final": float(p_gt_final),
            "gt_export_clip": float(gt_export_clip),
            "hc_dispatch": hc_dispatch,
            "p_grid": float(p_grid),
            "grid_dispatch": grid_dispatch,
        }

    # ------------------------------------------------------------------
    # _compute_departure_shortage_penalty()
    # 对 EV 离站缺电按 kWh 缺口分段惩罚：soft / mid / hard。
    # 这比直接按 SOC 缺口罚更物理。
    # ------------------------------------------------------------------
    def _compute_departure_shortage_penalty(self, shortage_kwh: np.ndarray) -> Dict[str, float]:
        shortage_kwh = np.asarray(shortage_kwh, dtype=np.float64)
        if shortage_kwh.size == 0:
            return {
                "depart_energy_shortage_kwh": 0.0,
                "depart_shortage_soft_kwh": 0.0,
                "depart_shortage_mid_kwh": 0.0,
                "depart_shortage_hard_kwh": 0.0,
                "penalty_depart_energy_soft": 0.0,
                "penalty_depart_energy_mid": 0.0,
                "penalty_depart_energy_hard": 0.0,
                "penalty_depart_energy": 0.0,
            }

        soft_end = max(float(self.cfg.depart_penalty_soft_kwh), 0.0)
        mid_end = max(float(self.cfg.depart_penalty_mid_kwh), soft_end)

        soft_band = np.minimum(shortage_kwh, soft_end)
        mid_band = np.minimum(np.maximum(shortage_kwh - soft_end, 0.0), mid_end - soft_end)
        hard_band = np.maximum(shortage_kwh - mid_end, 0.0)

        penalty_soft = float(np.sum(soft_band) * self.cfg.penalty_depart_energy_soft)
        penalty_mid = float(np.sum(mid_band) * self.cfg.penalty_depart_energy_mid)
        penalty_hard = float(np.sum(hard_band) * self.cfg.penalty_depart_soc)
        penalty_total = penalty_soft + penalty_mid + penalty_hard

        return {
            "depart_energy_shortage_kwh": float(np.sum(shortage_kwh)),
            "depart_shortage_soft_kwh": float(np.sum(soft_band)),
            "depart_shortage_mid_kwh": float(np.sum(mid_band)),
            "depart_shortage_hard_kwh": float(np.sum(hard_band)),
            "penalty_depart_energy_soft": penalty_soft,
            "penalty_depart_energy_mid": penalty_mid,
            "penalty_depart_energy_hard": penalty_hard,
            "penalty_depart_energy": penalty_total,
        }

    # ------------------------------------------------------------------
    # _compute_departure_risk_penalty()
    # 在真正离站之前，对“快离站但仍可能充不满”的 EV 做平滑风险惩罚。
    # 作用是让策略提前准备，而不是最后一刻抢充。
    # ------------------------------------------------------------------
    def _compute_departure_risk_penalty(self, next_t: int) -> Dict[str, float]:
        if next_t >= self.T or self.cfg.penalty_ev_depart_risk <= 0.0:
            return {
                "depart_risk_energy_kwh": 0.0,
                "penalty_depart_risk": 0.0,
                "depart_risk_vehicle_count": 0,
            }

        ev_next = self._compute_ev_boundaries(next_t, self.ev_soc)
        need_to_target = np.maximum(self.ev["target_soc"] - self.ev_soc, 0.0) * self.ev["cap_kwh"]
        time_left_h = np.maximum(self.ev["dep_step"] - next_t, 0) * self.cfg.dt

        risk_window_h = max(float(self.cfg.ev_depart_risk_window_h), self.cfg.dt)
        risk_buffer_h = max(min(float(self.cfg.ev_depart_risk_buffer_h), risk_window_h), self.cfg.dt)

        risk_mask = (
            ev_next["active_mask"]
            & (need_to_target > 1e-6)
            & (time_left_h <= risk_window_h)
        )
        if not np.any(risk_mask):
            return {
                "depart_risk_energy_kwh": 0.0,
                "penalty_depart_risk": 0.0,
                "depart_risk_vehicle_count": 0,
            }

        laxity_gap = np.maximum(risk_buffer_h - ev_next["laxity_target"], 0.0)
        risk_weight = np.clip(laxity_gap / max(risk_buffer_h, 1e-6), 0.0, 1.0)
        risk_energy = need_to_target * risk_weight * risk_mask.astype(np.float32)
        risk_energy_total = float(np.sum(risk_energy))

        return {
            "depart_risk_energy_kwh": risk_energy_total,
            "penalty_depart_risk": risk_energy_total * self.cfg.penalty_ev_depart_risk,
            "depart_risk_vehicle_count": int(np.count_nonzero(risk_energy > 1e-6)),
        }

    # ------------------------------------------------------------------
    # _is_ev_discharge_reward_active()：判断 EV 放电奖励是否激活。
    # 判断 EV 放电奖励是否激活：高电价 + 存在系统电功率压力。
    # ------------------------------------------------------------------
    def _is_ev_discharge_reward_active(
        self,
        *,
        grid_buy_price: float,
        peak_pressure_without_storage: float,
    ) -> bool:
        price_tol = max(float(self.cfg.ev_price_tolerance), 0.0)
        return bool(
            grid_buy_price >= self.cfg.ev_discharge_price_threshold - price_tol
            and peak_pressure_without_storage > self.cfg.ev_discharge_pressure_threshold_kw + 1e-6
        )

    # ------------------------------------------------------------------
    # _compute_ev_discharge_reward()：计算 EV 放电削峰奖励。
    # 在高价高压场景下，对 EV 真正参与削峰的放电给予奖励。
    # ------------------------------------------------------------------
    def _compute_ees_discharge_reward_weight(self, ees_soc: float) -> float:
        floor = float(
            np.clip(
                self.cfg.ees_reward_discharge_soc_floor,
                self.cfg.ees_soc_min,
                self.cfg.ees_soc_max,
            )
        )
        if self.cfg.ees_soc_max <= floor + 1e-6:
            return float(1.0 if ees_soc > floor + 1e-6 else 0.0)
        return float(
            np.clip(
                (ees_soc - floor) / max(self.cfg.ees_soc_max - floor, 1e-6),
                0.0,
                1.0,
            )
        )

    # ------------------------------------------------------------------
    # _compute_ees_charge_reward_weight()
    # Increase EES charge encouragement while SOC is below the target range.
    # ------------------------------------------------------------------
    def _compute_ees_charge_reward_weight(self, ees_soc: float) -> float:
        target = float(
            np.clip(
                self.cfg.ees_reward_charge_soc_target,
                self.cfg.ees_soc_min,
                self.cfg.ees_soc_max,
            )
        )
        if target <= self.cfg.ees_soc_min + 1e-6:
            return float(1.0 if ees_soc < target - 1e-6 else 0.0)
        return float(
            np.clip(
                (target - ees_soc) / max(target - self.cfg.ees_soc_min, 1e-6),
                0.0,
                1.0,
            )
        )

    # ------------------------------------------------------------------
    # _compute_storage_discharge_reward()
    # Reward EV + EES cooperative peak shaving instead of EV-only peak shaving.
    # ------------------------------------------------------------------
    def _compute_storage_discharge_reward(
        self,
        *,
        grid_buy_price: float,
        p_grid: float,
        p_grid_sell: float,
        p_ev_dis: float,
        p_ees_dis: float,
        ees_soc: float,
    ) -> Dict[str, float]:
        total_storage_dis = max(p_ev_dis, 0.0) + max(p_ees_dis, 0.0)
        if (
            total_storage_dis <= 1e-6
            or self.cfg.reward_storage_discharge_base <= 0.0
            or p_grid_sell > self.cfg.ev_peak_export_tolerance_kw + 1e-6
        ):
            return {
                "storage_peak_pressure_without_storage_kw": 0.0,
                "storage_peak_shaved_kwh": 0.0,
                "reward_storage_discharge_bonus": 0.0,
                "ees_discharge_reward_weight": 0.0,
            }

        peak_pressure_without_storage = max(p_grid + p_ev_dis + p_ees_dis, 0.0)
        if not self._is_ev_discharge_reward_active(
            grid_buy_price=grid_buy_price,
            peak_pressure_without_storage=peak_pressure_without_storage,
        ):
            return {
                "storage_peak_pressure_without_storage_kw": 0.0,
                "storage_peak_shaved_kwh": 0.0,
                "reward_storage_discharge_bonus": 0.0,
                "ees_discharge_reward_weight": 0.0,
            }

        ees_weight = self._compute_ees_discharge_reward_weight(ees_soc)
        effective_storage_dis = max(p_ev_dis, 0.0) + max(p_ees_dis, 0.0) * ees_weight
        if effective_storage_dis <= 1e-6:
            return {
                "storage_peak_pressure_without_storage_kw": float(peak_pressure_without_storage),
                "storage_peak_shaved_kwh": 0.0,
                "reward_storage_discharge_bonus": 0.0,
                "ees_discharge_reward_weight": float(ees_weight),
            }

        storage_peak_shaved_kwh = (
            min(effective_storage_dis, peak_pressure_without_storage) * self.cfg.dt
        )
        reward_unit = self.cfg.reward_storage_discharge_base * max(
            grid_buy_price / max(self.cfg.ev_discharge_price_threshold, 1e-6),
            1.0,
        )
        reward_bonus = storage_peak_shaved_kwh * reward_unit
        return {
            "storage_peak_pressure_without_storage_kw": float(peak_pressure_without_storage),
            "storage_peak_shaved_kwh": float(storage_peak_shaved_kwh),
            "reward_storage_discharge_bonus": float(reward_bonus),
            "ees_discharge_reward_weight": float(ees_weight),
        }

    # ------------------------------------------------------------------
    # _compute_storage_charge_preparation_reward()
    # Reward EV buffer charge and EES charge together, while keeping a small
    # EV target-timing bonus for travel readiness.
    # ------------------------------------------------------------------
    def _compute_storage_charge_preparation_reward(
        self,
        *,
        grid_buy_price: float,
        low_value_energy_before_storage_prepare_kwh: float,
        p_ev_flex_target_ch: float,
        p_ev_buffer_ch: float,
        p_ees_ch: float,
        ees_soc: float,
    ) -> Dict[str, float]:
        target_kwh = max(p_ev_flex_target_ch, 0.0) * self.cfg.dt
        buffer_kwh = max(p_ev_buffer_ch, 0.0) * self.cfg.dt
        ees_charge_kwh = max(p_ees_ch, 0.0) * self.cfg.dt
        low_value_charge_kwh = 0.0

        if (
            self.cfg.reward_storage_charge_base <= 0.0
            and self.cfg.reward_ev_target_timing_base <= 0.0
        ):
            return {
                "storage_charge_rewarded_kwh": 0.0,
                "ev_flex_target_charge_kwh": float(target_kwh),
                "ev_buffer_charge_kwh": float(buffer_kwh),
                "ees_charge_kwh": float(ees_charge_kwh),
                "ees_charge_rewarded_kwh": 0.0,
                "low_value_charge_kwh": 0.0,
                "reward_storage_charge_bonus": 0.0,
                "reward_ev_target_timing_bonus": 0.0,
                "ees_charge_reward_weight": 0.0,
            }

        price_tol = max(float(self.cfg.ev_price_tolerance), 0.0)
        low_price_active = grid_buy_price <= self.cfg.ev_charge_price_threshold + price_tol
        low_value_active = low_value_energy_before_storage_prepare_kwh > 1e-9
        if not low_price_active and not low_value_active:
            return {
                "storage_charge_rewarded_kwh": 0.0,
                "ev_flex_target_charge_kwh": float(target_kwh),
                "ev_buffer_charge_kwh": float(buffer_kwh),
                "ees_charge_kwh": float(ees_charge_kwh),
                "ees_charge_rewarded_kwh": 0.0,
                "low_value_charge_kwh": 0.0,
                "reward_storage_charge_bonus": 0.0,
                "reward_ev_target_timing_bonus": 0.0,
                "ees_charge_reward_weight": 0.0,
            }

        ees_weight = self._compute_ees_charge_reward_weight(ees_soc)
        effective_storage_charge_kwh = buffer_kwh + ees_charge_kwh * ees_weight
        target_rewarded_kwh = 0.0
        storage_rewarded_kwh = 0.0
        reward_storage_charge_bonus = 0.0
        reward_ev_target_timing_bonus = 0.0

        if low_price_active and self.cfg.reward_ev_target_timing_base > 0.0 and target_kwh > 1e-9:
            target_unit = self.cfg.reward_ev_target_timing_base * min(
                self.cfg.ev_charge_price_threshold / max(grid_buy_price, 1e-6),
                1.5,
            )
            target_rewarded_kwh = target_kwh
            reward_ev_target_timing_bonus += target_rewarded_kwh * target_unit

        if (
            low_price_active
            and self.cfg.reward_storage_charge_base > 0.0
            and effective_storage_charge_kwh > 1e-9
        ):
            charge_unit = self.cfg.reward_storage_charge_base * min(
                self.cfg.ev_charge_price_threshold / max(grid_buy_price, 1e-6),
                1.5,
            )
            storage_rewarded_kwh += effective_storage_charge_kwh
            reward_storage_charge_bonus += effective_storage_charge_kwh * charge_unit

        if (
            low_value_active
            and self.cfg.reward_storage_charge_base > 0.0
            and effective_storage_charge_kwh > 1e-9
        ):
            low_value_charge_kwh = min(
                effective_storage_charge_kwh,
                max(low_value_energy_before_storage_prepare_kwh, 0.0),
            )
            storage_rewarded_kwh += low_value_charge_kwh
            reward_storage_charge_bonus += (
                low_value_charge_kwh * self.cfg.reward_storage_charge_base
            )

        return {
            "storage_charge_rewarded_kwh": float(storage_rewarded_kwh),
            "ev_flex_target_charge_kwh": float(target_kwh),
            "ev_buffer_charge_kwh": float(buffer_kwh),
            "ees_charge_kwh": float(ees_charge_kwh),
            "ees_charge_rewarded_kwh": float(ees_charge_kwh * ees_weight),
            "low_value_charge_kwh": float(low_value_charge_kwh),
            "reward_storage_charge_bonus": float(reward_storage_charge_bonus),
            "reward_ev_target_timing_bonus": float(reward_ev_target_timing_bonus),
            "ees_charge_reward_weight": float(ees_weight),
        }

    # _evaluate_gt_candidate()
    # 给定一个候选 GT 功率，测试它对：
    # - 热/冷可行性
    # - 电侧净功率 p_grid
    # 的影响。
    # 供 GT 安全带搜索使用。
    # ------------------------------------------------------------------
    def _evaluate_gt_candidate(
        self,
        exo: Dict[str, float],
        p_gt: float,
        p_ev_ch: float,
        p_ev_dis: float,
        p_ees_ch: float,
        p_ees_dis: float,
        apply_ramp: bool = True,
    ) -> Dict[str, float]:
        # 给定一个候选 GT 出力，先结算与之耦合的冷热设备可行执行值。
        # 注意：这里的 apply_ramp 主要影响 GB/AC/EC 所对应的冷热侧可行性；
        # GT 自身的爬坡约束已经由外层搜索区间 [p_gt_lower, p_gt_upper] 体现。
        hc_try = self._dispatch_heat_cold(
            exo=exo,
            p_gt=float(p_gt),
            p_ev_ch=p_ev_ch,
            p_ev_dis=p_ev_dis,
            p_ees_ch=p_ees_ch,
            p_ees_dis=p_ees_dis,
            apply_ramp=apply_ramp,
        )
        # 复用统一电侧口径，避免 GT 候选评估与主结算链路发生漂移。
        # p_grid > 0 表示购电，p_grid < 0 表示向电网送电。
        p_grid_try = self._compute_net_grid_power(
            exo=exo,
            p_gt=float(p_gt),
            hc_dispatch=hc_try,
            ev_dispatch={
                "p_ev_ch": float(p_ev_ch),
                "p_ev_dis": float(p_ev_dis),
            },
            ees_dispatch={
                "p_ees_ch": float(p_ees_ch),
                "p_ees_dis": float(p_ees_dis),
            },
        )
        return {
            "p_grid": float(p_grid_try),
            "unmet_heat": float(hc_try["unmet_heat"]),
            "unmet_cool": float(hc_try["unmet_cool"]),
            "p_ec_elec_in": float(hc_try["p_ec_elec_in"]),
        }

    # ------------------------------------------------------------------
    # _find_gt_min_safe()
    # 这是 GT 安全层的关键函数。它会在当前可行爬坡区间内搜索：
    # - 冷热联合可行的最小 GT 出力
    # - 尽量避免反送电时允许的最大 GT 出力
    # - 低价低压时进一步压缩 GT 的经济上限
    # 最终返回一个 [safe_min, safe_max] 区间。
    # ------------------------------------------------------------------
    def _find_gt_min_safe(
        self,
        exo: Dict[str, float],
        p_ev_ch: float,
        p_ev_dis: float,
        p_ees_ch: float,
        p_ees_dis: float,
    ) -> Dict[str, Any]:
        """搜索当前时刻 GT 安全出力区间。

        该区间同时考虑热/冷可行性、电侧防反送电约束，以及低电价低压力时段的经济上限。
        """
        # 在“当前 GT 上一步出力 ± 爬坡限值”形成的物理区间内搜索可行带。
        p_gt_lower = max(self.gt_power - self.cfg.gt_ramp, 0.0)
        p_gt_upper = min(self.gt_power + self.cfg.gt_ramp, self.cfg.gt_p_max)

        search_points = max(int(self.cfg.gt_safe_search_points), 3)
        # thermal_min_feasible: 扫描到的第一个冷热联合可行点，即本步满足冷热负荷的最小 GT。
        # electric_max_feasible: 扫描到的最后一个不过度外送的点，即电侧可接受的最大 GT。
        # joint_max: 同时满足冷热可行和电侧不过度外送的最后一个点。
        # 由于这里是离散搜索，边界精度受 gt_safe_search_points 影响。
        thermal_min_feasible = None
        electric_max_feasible = None
        joint_max = None

        for p_gt_try in np.linspace(p_gt_lower, p_gt_upper, search_points):
            gt_eval = self._evaluate_gt_candidate(
                exo=exo,
                p_gt=float(p_gt_try),
                p_ev_ch=p_ev_ch,
                p_ev_dis=p_ev_dis,
                p_ees_ch=p_ees_ch,
                p_ees_dis=p_ees_dis,
                apply_ramp=True,
            )
            thermal_ok = (
                gt_eval["unmet_heat"] <= 1e-6
                and gt_eval["unmet_cool"] <= 1e-6
            )
            electric_ok = gt_eval["p_grid"] >= -self.cfg.gt_export_margin

            if electric_ok:
                electric_max_feasible = float(p_gt_try)
            if thermal_ok and thermal_min_feasible is None:
                thermal_min_feasible = float(p_gt_try)
            if thermal_ok and electric_ok:
                joint_max = float(p_gt_try)

        thermal_feasible = thermal_min_feasible is not None
        if thermal_feasible:
            thermal_min_ref = float(thermal_min_feasible)
        else:
            # 若当前爬坡带内连最大 GT 都无法满足冷热侧，则将安全带压缩到上边界，
            # 表示“本步已经尽力抬高 GT，但冷热侧仍不可完全满足”。
            thermal_min_ref = float(p_gt_upper)

        electric_feasible = electric_max_feasible is not None
        if electric_feasible:
            electric_max_ref = float(electric_max_feasible)
        else:
            # 若当前爬坡带内所有 GT 都会导致外送超过容忍值，则保守记录为下边界参考点。
            electric_max_ref = float(p_gt_lower)

        joint_feasible = (
            thermal_feasible
            and electric_feasible
            and joint_max is not None
            and joint_max + 1e-9 >= thermal_min_ref
        )

        if joint_feasible:
            # 存在冷热/电侧共同可行区间时，上边界取共同可行区间的最大点。
            p_gt_safe_max_base = float(joint_max)
        else:
            # 冷热联合可行性优先级高于防反送电偏好。
            # 因此若两类约束冲突，则至少保证 GT 不低于冷热侧最小可行参考点。
            p_gt_safe_max_base = float(thermal_min_ref)

        p_gt_safe_max_base = float(min(max(p_gt_safe_max_base, thermal_min_ref), p_gt_upper))

        gt_eval_at_thermal_ref = self._evaluate_gt_candidate(
            exo=exo,
            p_gt=float(thermal_min_ref),
            p_ev_ch=p_ev_ch,
            p_ev_dis=p_ev_dis,
            p_ees_ch=p_ees_ch,
            p_ees_dis=p_ees_dis,
            apply_ramp=True,
        )
        import_pressure_at_thermal_ref = max(float(gt_eval_at_thermal_ref["p_grid"]), 0.0)
        price_tol = max(float(self.cfg.ev_price_tolerance), 0.0)
        # 仅当“电价低”且“在冷热联合可行最小 GT 参考点处电侧购电压力也不大”时，
        # 才进一步压缩 GT 的经济上限，避免低价时把 GT 开得过高。
        gt_low_price_active = bool(
            exo["grid_buy_price"] <= self.cfg.gt_low_price_threshold + price_tol
            and import_pressure_at_thermal_ref <= self.cfg.gt_low_price_pressure_threshold_kw + 1e-6
        )

        if gt_low_price_active:
            # 低价低压时只收缩上边界，不抬高下边界，
            # 从而仍保持冷热侧可行性优先。
            p_gt_economic_max = min(
                p_gt_safe_max_base,
                thermal_min_ref + max(float(self.cfg.gt_low_price_headroom_kw), 0.0),
            )
        else:
            p_gt_economic_max = p_gt_safe_max_base

        p_gt_safe_max = float(min(max(p_gt_economic_max, thermal_min_ref), p_gt_upper))

        return {
            "p_gt_safe_min": float(thermal_min_ref),
            "p_gt_safe_max": float(p_gt_safe_max),
            "p_gt_safe_max_base": float(p_gt_safe_max_base),
            "p_gt_economic_max": float(p_gt_economic_max),
            # 下述 *_ref 字段表示“当前控制与诊断实际采用的参考点”，
            # 在可行时与真实可行边界相同；在不可行时会退化到物理边界参考点。
            "p_gt_thermal_min_ref": float(thermal_min_ref),
            "gt_import_pressure_at_thermal": float(import_pressure_at_thermal_ref),
            "gt_import_pressure_at_thermal_ref": float(import_pressure_at_thermal_ref),
            "gt_low_price_active": bool(gt_low_price_active),
            "p_gt_electric_max": float(electric_max_ref),
            "p_gt_electric_max_ref": float(electric_max_ref),
            # 下述 *_feasible 字段仅在对应可行边界真实存在时给出数值，否则返回 None。
            "p_gt_thermal_min_feasible": (
                float(thermal_min_feasible) if thermal_feasible else None
            ),
            "p_gt_electric_max_feasible": (
                float(electric_max_feasible) if electric_feasible else None
            ),
            "p_gt_physical_max": float(p_gt_upper),
            "thermal_feasible": bool(thermal_feasible),
            "electric_feasible": bool(electric_feasible),
            "joint_feasible": bool(joint_feasible),
        }

    # ------------------------------------------------------------------
    # _map_gt_action_to_band()
    # 将 [-1, 1] 的 GT 动作直接仿射映射到当前安全带 [lower, upper]。
    # 这样避免“先给不安全动作，再被安全层纠正”的学习错配。
    # ------------------------------------------------------------------
    def _map_gt_action_to_band(self, a_gt: float, lower: float, upper: float) -> float:
        """将 GT 动作从 [-1, 1] 仿射映射到当前安全可行区间。"""
        lower = float(lower)
        upper = float(upper)
        if upper <= lower + 1e-9:
            # 可行带退化成单点时，动作失去自由度，直接返回该唯一可行值。
            return lower

        # a_gt=-1 对应下边界，a_gt=+1 对应上边界，中间线性插值。
        alpha = 0.5 * (float(a_gt) + 1.0)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return float(lower + alpha * (upper - lower))

    # ------------------------------------------------------------------
    # _initialize_operating_point()
    # 给回合开始时的 GT/GB/AC/EC 一个弱热启动点。
    # 重点：GT 不是按电平衡启动，而是按“冷热侧最小可行”启动，避免夜间继承过高 GT。
    # ------------------------------------------------------------------
    def _initialize_operating_point(self) -> None:
        """初始化回合开始时的设备状态。

        GT 按第 0 小时热侧最小可行出力启动，避免初始 GT 出力过高影响夜间调度。
        """
        exo0 = self._get_exogenous(0)

        search_points = max(int(self.cfg.gt_safe_search_points), 3)
        best_p_gt = 0.0
        best_hc = self._dispatch_heat_cold(
            exo=exo0,
            p_gt=0.0,
            p_ev_ch=0.0,
            p_ev_dis=0.0,
            p_ees_ch=0.0,
            p_ees_dis=0.0,
            apply_ramp=False,
        )
        best_score = (
            best_hc["unmet_heat"] + best_hc["unmet_cool"] > 1e-6,
            best_hc["unmet_heat"] + best_hc["unmet_cool"],
            0.0,
        )

        for p_gt_try in np.linspace(0.0, self.cfg.gt_p_max, search_points):
            hc_try = self._dispatch_heat_cold(
                exo=exo0,
                p_gt=float(p_gt_try),
                p_ev_ch=0.0,
                p_ev_dis=0.0,
                p_ees_ch=0.0,
                p_ees_dis=0.0,
                apply_ramp=False,
            )
            unmet_total = hc_try["unmet_heat"] + hc_try["unmet_cool"]
            score = (unmet_total > 1e-6, unmet_total, float(p_gt_try))
            if score < best_score:
                best_score = score
                best_p_gt = float(p_gt_try)
                best_hc = hc_try
            if unmet_total <= 1e-6:
                break

        self.gt_power = float(best_p_gt)
        self.gb_heat = float(best_hc["p_gb_heat"])
        self.ac_cool = float(best_hc["p_ac_cool"])
        self.ec_cool = float(best_hc["p_ec_cool"])

    # ------------------------------------------------------------------
    # _apply_ramp_limit()：施加爬坡约束。
    # 通用对称爬坡限制工具函数。
    # ------------------------------------------------------------------
    def _apply_ramp_limit(
        self,
        target: float,
        previous: float,
        ramp: float,
        upper: float,
    ) -> float:
        """对设备出力施加对称爬坡约束，并裁剪到物理上限内。"""
        lower_bound = max(previous - ramp, 0.0)
        upper_bound = min(previous + ramp, upper)
        return float(np.clip(target, lower_bound, upper_bound))

    # ------------------------------------------------------------------
    # _settle_grid_power()
    # 电网功率结算：
    # - p_grid > 0 表示购电
    # - p_grid < 0 表示上网
    # 超过购电/上网上限的部分分别记为缺供电量和弃电量。
    # ------------------------------------------------------------------
    def _settle_grid_power(self, p_grid: float) -> Dict[str, float]:
        """结算电网净交互功率。p_grid 为正表示购电，为负表示上网。"""
        export_power = max(-p_grid, 0.0)
        p_grid_buy = min(max(p_grid, 0.0), self.cfg.grid_import_max)
        p_grid_sell = min(export_power, self.cfg.grid_export_max)

        unmet_e = max(p_grid - self.cfg.grid_import_max, 0.0)
        surplus_e = max(export_power - self.cfg.grid_export_max, 0.0)

        return {
            "p_grid_buy": float(p_grid_buy),
            "p_grid_sell": float(p_grid_sell),
            "unmet_e": float(unmet_e),
            "surplus_e": float(surplus_e),
        }

    # ------------------------------------------------------------------
    # _effective_export_value()：计算上网电量有效价值。
    # 计算“有效售电价值” = 售电价 - 显式外送惩罚。
    # 目的是让上网只是泄压口，而不是主盈利来源。
    # ------------------------------------------------------------------
    def _effective_export_value(self, grid_sell_price: float) -> float:
        """计算单位上网电量的有效边际价值。

        售电收入仍参与结算，但扣除外送惩罚，使上网只作为泄压手段而非主要盈利来源。
        """
        return float(max(grid_sell_price - self.cfg.penalty_export_e, 0.0))

    # ------------------------------------------------------------------
    # _piecewise_ec_cooling_target()：计算分段经济 EC 制冷目标。
    # 用分段边际电价决定 EC 应承担多少制冷：
    # 1. 超限外送电对应的边际价值最低；
    # 2. 限额内外送电对应有效售电价值；
    # 3. 购电对应购电价。
    # 只在 EC 的单位制冷成本不高于 AC-from-GB 时使用 EC。
    # ------------------------------------------------------------------
    def _piecewise_ec_cooling_target(
        self,
        cool_need: float,
        p_grid_base: float,
        grid_buy_price: float,
        grid_sell_price: float,
        c_ac_gb: float,
    ) -> float:
        """按分段电能边际价值计算经济型 EC 制冷目标。

        EC 单位冷量成本按“电能边际价值 / COP + EC 运维成本”计算。
        """
        cool_need = float(max(cool_need, 0.0))
        if cool_need <= 1e-6:
            return 0.0

        spill_export_e = max(-p_grid_base - self.cfg.grid_export_max, 0.0)
        export_within_limit_e = min(max(-p_grid_base, 0.0), self.cfg.grid_export_max)
        import_headroom_e = max(
            self.cfg.grid_import_max - max(p_grid_base, 0.0),
            0.0,
        )

        segments = [
            (spill_export_e, 0.0),
            (export_within_limit_e, self._effective_export_value(grid_sell_price)),
            (import_headroom_e, grid_buy_price),
        ]

        ec_cool = 0.0
        ec_cool_cap_left = self.cfg.ec_c_max

        for seg_e_cap, elec_price in segments:
            if cool_need <= 1e-6 or ec_cool_cap_left <= 1e-6:
                break

            c_ec = elec_price / max(self.cfg.ec_cop, 1e-6) + self.cfg.om_ec
            if c_ec > c_ac_gb + 1e-12:
                continue

            seg_cool_cap = seg_e_cap * self.cfg.ec_cop
            use_cool = min(cool_need, ec_cool_cap_left, seg_cool_cap)

            ec_cool += use_cool
            ec_cool_cap_left -= use_cool
            cool_need -= use_cool

        return float(ec_cool)

    # ------------------------------------------------------------------
    # _compute_heat_cold_targets()：计算冷热目标。
    # 计算“未加爬坡约束前”的理想冷热目标：
    # - GT 余热优先供热；
    # - WHB 富余热优先供 AC；
    # - 剩余冷负荷在 EC 与 AC-from-GB 之间按边际成本分配；
    # - 如果 AC 仍不足，再用额外 EC 兜底可行性。
    # 这是冷热子系统决策逻辑的核心。
    # ------------------------------------------------------------------
    def _compute_heat_cold_targets(
        self,
        exo: Dict[str, float],
        p_gt: float,
        p_ev_ch: float,
        p_ev_dis: float,
        p_ees_ch: float,
        p_ees_dis: float,
    ) -> Dict[str, float]:
        """计算未施加爬坡约束前的理想冷热目标。

        该层先保证冷热可行性，再根据边际成本生成 AC/EC 理想目标，
        并给出协调层所需的热侧边界信息。
        """
        heat_load = exo["heat_load"]
        cool_load = exo["cool_load"]
        grid_buy_price = exo["grid_buy_price"]
        grid_sell_price = exo["grid_sell_price"]
        gas_price = exo["gas_price"]

        # GT 发电后，经 WHB 回收得到可用热量。
        p_whb_heat = min(
            p_gt * (1.0 - self.cfg.gt_eta_e) * self.cfg.whb_eta_h / self.cfg.gt_eta_e,
            self.cfg.whb_h_max,
        )

        # p_grid_base: 先不考虑 EC 耗电时的基础电侧净负荷。
        p_grid_base = (
            exo["elec_load"]
            + p_ev_ch
            + p_ees_ch
            - p_ev_dis
            - p_ees_dis
            - exo["pv"]
            - exo["wt"]
            - p_gt
        )

        q_whb_to_heat = min(heat_load, p_whb_heat)
        q_whb_surplus = max(p_whb_heat - q_whb_to_heat, 0.0)
        heat_gap_after_whb = max(heat_load - p_whb_heat, 0.0)
        q_gb_to_heat_target = min(heat_gap_after_whb, self.cfg.gb_h_max)
        gb_remain_target = max(self.cfg.gb_h_max - q_gb_to_heat_target, 0.0)

        p_ac_from_whb_target = min(
            self.cfg.ac_c_max,
            cool_load,
            q_whb_surplus * self.cfg.ac_cop,
        )

        cool_remain = max(cool_load - p_ac_from_whb_target, 0.0)
        p_ac_from_gb_cap_target = min(
            max(self.cfg.ac_c_max - p_ac_from_whb_target, 0.0),
            gb_remain_target * self.cfg.ac_cop,
        )

        # 在不造成电侧缺供的前提下计算 EC 最大可承担制冷量。
        p_ec_elec_cap_target = max(self.cfg.grid_import_max - p_grid_base, 0.0)
        p_ec_cap_target = min(
            self.cfg.ec_c_max,
            p_ec_elec_cap_target * self.cfg.ec_cop,
        )

        # 冷侧目标层按完整边际成本比较 EC 与 AC-from-GB。

        # GB 供热驱动 AC 制冷的单位冷量成本包括：
        # 1）GB 产热燃气成本；
        # 2）GB 对应热出力运维成本；
        # 3）AC 制冷运维成本。
        c_ac_gb = (
            gas_price / max(self.cfg.gb_eta_h * self.cfg.ac_cop, 1e-6)
            + self.cfg.om_gb / max(self.cfg.ac_cop, 1e-6)
            + self.cfg.om_ac
        )

        # 优先使用边际制冷成本不高于 AC-from-GB 的 EC 制冷。


        p_ec_cool_economic = self._piecewise_ec_cooling_target(
            cool_need=cool_remain,
            p_grid_base=p_grid_base,
            grid_buy_price=grid_buy_price,
            grid_sell_price=grid_sell_price,
            c_ac_gb=c_ac_gb,
        )

        cool_after_economic_ec = max(cool_remain - p_ec_cool_economic, 0.0)

        # 其次使用 GB 供热驱动 AC 作为主要兜底。
        p_ac_from_gb_target = min(
            cool_after_economic_ec,
            p_ac_from_gb_cap_target,
        )
        cool_after_ac = max(cool_after_economic_ec - p_ac_from_gb_target, 0.0)

        # 如果 AC 仍不足，则使用剩余 EC 能力兜底，优先保证冷负荷可行。

        p_ec_cool_extra = min(
            cool_after_ac,
            max(p_ec_cap_target - p_ec_cool_economic, 0.0),
        )
        p_ec_cool_target = p_ec_cool_economic + p_ec_cool_extra

        p_ac_cool_target = p_ac_from_whb_target + p_ac_from_gb_target

        return {
            "p_whb_heat": float(p_whb_heat),
            "q_whb_surplus": float(q_whb_surplus),
            "heat_gap_after_whb": float(heat_gap_after_whb),
            "p_ac_cool_target": float(p_ac_cool_target),
            "p_ec_cool_target": float(p_ec_cool_target),
            "p_ec_cool_cap": float(p_ec_cap_target),
        }

    # ------------------------------------------------------------------
    # _coordinate_cooling_dispatch()
    # 给定总制冷目标后，在 AC 与 EC 之间做协调，
    # 同时考虑 GB 热支撑与爬坡约束。
    # 这个函数负责把“总冷量”变成“AC 冷量 + EC 冷量 + GB 热量”。
    # ------------------------------------------------------------------
    def _coordinate_cooling_dispatch(
        self,
        *,
        cool_target_total: float,
        desired_ac_target: float,
        p_ec_cool_cap: float,
        q_whb_surplus: float,
        heat_gap_after_whb: float,
        apply_ramp: bool,
    ) -> Dict[str, float]:
        """在给定总制冷目标下协调 AC 与 EC 出力。

        当前小时尺度下，AC/EC 视为快速响应设备，爬坡约束主要作用于 GT、GB 和 EES。
        """
        free_ac_cap = min(self.cfg.ac_c_max, q_whb_surplus * self.cfg.ac_cop)

        if apply_ramp:
            gb_lower = max(self.gb_heat - self.cfg.gb_ramp, 0.0)
            gb_upper = min(self.gb_heat + self.cfg.gb_ramp, self.cfg.gb_h_max)
        else:
            gb_lower = 0.0
            gb_upper = self.cfg.gb_h_max

        gb_heat_for_ac_lower = max(gb_lower - heat_gap_after_whb, 0.0)
        gb_heat_for_ac_upper = max(gb_upper - heat_gap_after_whb, 0.0)
        ac_preferred_lower = min(
            self.cfg.ac_c_max,
            free_ac_cap + gb_heat_for_ac_lower * self.cfg.ac_cop,
        )
        ac_upper_heat = min(
            self.cfg.ac_c_max,
            free_ac_cap + gb_heat_for_ac_upper * self.cfg.ac_cop,
        )

        ac_upper = ac_upper_heat
        ac_lower = 0.0
        ec_lower = 0.0
        ec_upper = min(self.cfg.ec_c_max, p_ec_cool_cap)

        total_feasible_min = ac_lower + ec_lower
        total_feasible_max = ac_upper + ec_upper
        total_cool = float(np.clip(cool_target_total, total_feasible_min, total_feasible_max))

        ac_needed_min = total_cool - ec_upper
        ac_needed_max = total_cool - ec_lower
        ac_feasible_low = max(ac_lower, ac_needed_min, 0.0)
        ac_feasible_high = min(ac_upper, ac_needed_max)
        ac_dispatch_low = min(ac_feasible_high, max(ac_feasible_low, ac_preferred_lower))

        desired_ac = float(np.clip(desired_ac_target, 0.0, self.cfg.ac_c_max))

        if ac_feasible_low <= ac_feasible_high:
            p_ac_cool = float(np.clip(desired_ac, ac_dispatch_low, ac_feasible_high))
            p_ec_cool = float(total_cool - p_ac_cool)
        else:
            # 数值保护：若可行区间异常，则退化为最接近目标总冷量的可行分配。
            total_cool_target_feasible = total_cool
            p_ac_cool = float(np.clip(desired_ac, ac_lower, ac_upper))
            p_ec_cool = float(np.clip(total_cool_target_feasible - p_ac_cool, ec_lower, ec_upper))
            residual = total_cool_target_feasible - (p_ac_cool + p_ec_cool)
            if abs(residual) > 1e-6:
                p_ac_adjusted = float(np.clip(p_ac_cool + residual, ac_lower, ac_upper))
                residual -= p_ac_adjusted - p_ac_cool
                p_ac_cool = p_ac_adjusted
            if abs(residual) > 1e-6:
                p_ec_cool = float(np.clip(p_ec_cool + residual, ec_lower, ec_upper))
                total_cool = p_ac_cool + p_ec_cool
            else:
                total_cool = total_cool_target_feasible

        p_ac_from_whb = min(p_ac_cool, free_ac_cap)
        p_ac_from_gb = max(p_ac_cool - p_ac_from_whb, 0.0)
        q_ac_heat_from_gb = p_ac_from_gb / max(self.cfg.ac_cop, 1e-6)
        q_gb_needed = heat_gap_after_whb + q_ac_heat_from_gb

        if apply_ramp:
            p_gb_heat = self._apply_ramp_limit(
                target=q_gb_needed,
                previous=self.gb_heat,
                ramp=self.cfg.gb_ramp,
                upper=self.cfg.gb_h_max,
            )
        else:
            p_gb_heat = float(np.clip(q_gb_needed, 0.0, self.cfg.gb_h_max))

        q_gb_to_heat_actual = min(heat_gap_after_whb, p_gb_heat)
        gb_heat_for_ac_actual = max(p_gb_heat - q_gb_to_heat_actual, 0.0)
        p_ac_from_gb_actual = min(p_ac_from_gb, gb_heat_for_ac_actual * self.cfg.ac_cop)
        p_ac_cool_actual = p_ac_from_whb + p_ac_from_gb_actual
        q_ac_heat_from_gb_actual = p_ac_from_gb_actual / max(self.cfg.ac_cop, 1e-6)

        return {
            "p_gb_heat": float(p_gb_heat),
            "p_ac_cool": float(p_ac_cool_actual),
            "p_ec_cool": float(p_ec_cool),
            "q_ac_heat_from_gb": float(q_ac_heat_from_gb_actual),
            "total_cool_target": float(total_cool),
        }

    # ------------------------------------------------------------------
    # _dispatch_heat_cold()
    # 冷热子系统最终执行入口。
    # 顺序：
    # 1. 先算当前时刻未施加 GB 爬坡约束前的冷热目标；
    # 2. 再调用 _coordinate_cooling_dispatch() 做 AC/EC/GB 的可行协调；
    # 3. 最后在本层把协调层中间结果重新闭合成最终执行值，并结算 unmet / curtail。
    # 因此本函数返回的是冷热最终执行结果；
    # _coordinate_cooling_dispatch() 只负责协调层中间结果。
    # ------------------------------------------------------------------
    def _dispatch_heat_cold(
        self,
        exo: Dict[str, float],
        p_gt: float,
        p_ev_ch: float,
        p_ev_dis: float,
        p_ees_ch: float,
        p_ees_dis: float,
        apply_ramp: bool = True,
    ) -> Dict[str, float]:
        heat_load = exo["heat_load"]
        cool_load = exo["cool_load"]

        current_targets = self._compute_heat_cold_targets(
            exo=exo,
            p_gt=p_gt,
            p_ev_ch=p_ev_ch,
            p_ev_dis=p_ev_dis,
            p_ees_ch=p_ees_ch,
            p_ees_dis=p_ees_dis,
        )

        p_whb_heat = current_targets["p_whb_heat"]
        q_whb_surplus = current_targets["q_whb_surplus"]
        heat_gap_after_whb = current_targets["heat_gap_after_whb"]
        p_ac_cool_target = current_targets["p_ac_cool_target"]
        p_ec_cool_target = current_targets["p_ec_cool_target"]
        p_ec_cool_cap = current_targets["p_ec_cool_cap"]
        current_total_cool_target = p_ac_cool_target + p_ec_cool_target
        desired_ac_target = p_ac_cool_target

        # 先求目标，再由协调层给出 GB 爬坡可行的 AC / EC / GB 中间协调结果。
        cooling_dispatch = self._coordinate_cooling_dispatch(
            cool_target_total=current_total_cool_target,
            desired_ac_target=desired_ac_target,
            p_ec_cool_cap=p_ec_cool_cap,
            q_whb_surplus=q_whb_surplus,
            heat_gap_after_whb=heat_gap_after_whb,
            apply_ramp=apply_ramp,
        )
        # 注意：cooling_dispatch 仍是协调层结果，本函数下面会重新闭合成最终执行值。
        p_gb_heat = float(cooling_dispatch["p_gb_heat"])

        p_ac_cool = cooling_dispatch["p_ac_cool"]
        p_ec_cool = cooling_dispatch["p_ec_cool"]
        total_cool_target = cooling_dispatch["total_cool_target"]

        p_ac_from_whb_actual = min(
            p_ac_cool,
            self.cfg.ac_c_max,
            q_whb_surplus * self.cfg.ac_cop,
        )
        q_gb_to_heat_actual = min(heat_gap_after_whb, p_gb_heat)
        gb_remain_actual = max(p_gb_heat - q_gb_to_heat_actual, 0.0)
        p_ac_from_gb_actual = min(
            max(p_ac_cool - p_ac_from_whb_actual, 0.0),
            gb_remain_actual * self.cfg.ac_cop,
        )
        p_ac_cool = p_ac_from_whb_actual + p_ac_from_gb_actual
        # 按爬坡可行的总制冷目标重新闭合 AC/EC 分配。

        p_ec_cool = float(np.clip(total_cool_target - p_ac_cool, 0.0, p_ec_cool_cap))
        # AC 总制冷量都需要消耗热量，无论热量来自 WHB 还是 GB。

        q_ac_heat_in = p_ac_cool / max(self.cfg.ac_cop, 1e-6)

        total_heat_supply = p_whb_heat + p_gb_heat
        total_heat_demand = heat_load + q_ac_heat_in
        total_cool_supply = p_ac_cool + p_ec_cool

        unmet_heat = max(total_heat_demand - total_heat_supply, 0.0)
        unmet_cool = max(cool_load - total_cool_supply, 0.0)
        heat_curtail = max(total_heat_supply - total_heat_demand, 0.0)
        cool_curtail = max(total_cool_supply - cool_load, 0.0)

        p_ec_elec_in = p_ec_cool / max(self.cfg.ec_cop, 1e-6)

        return {
            "p_whb_heat": float(p_whb_heat),
            "p_gb_heat": float(p_gb_heat),
            "p_ac_cool": float(p_ac_cool),
            "p_ec_cool": float(p_ec_cool),
            "p_ec_elec_in": float(p_ec_elec_in),
            "unmet_heat": float(unmet_heat),
            "unmet_cool": float(unmet_cool),
            "heat_curtail": float(heat_curtail),
            "cool_curtail": float(cool_curtail),
        }
