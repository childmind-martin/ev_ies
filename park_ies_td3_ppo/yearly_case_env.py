from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Optional

import gymnasium as gym
import numpy as np

from park_ies_env import IESConfig, ParkIESEnv
from yearly_csv_loader import DailyIESCase, YearlyCSVDataLoader


class YearlyEVProvider:
    """
    从全年 EV 数据文件中，按 day_of_year 读取当天 EV 数据，
    并转换成 ParkIESEnv 需要的 ev_data 字典。

    原始 EV 数据最后一维默认字段顺序：
        0: 到达时间(h)
        1: 离开时间(h)
        2: 初始SOC
        3: 目标SOC
        4: V2G标识
        5: 电池容量(kWh)
        6: 最大充电功率(kW)
        7: 最大放电功率(kW)
        8: 日行驶里程(km)
        9: 充放电效率
    """

    def __init__(self, ev_npy_path: str | Path):
        self.ev_npy_path = Path(ev_npy_path)
        if not self.ev_npy_path.exists():
            raise FileNotFoundError(f"EV 年度数据文件不存在: {self.ev_npy_path}")

        ev_year = np.load(self.ev_npy_path)
        if ev_year.ndim != 3:
            raise ValueError(f"EV 数据应为三维数组，当前 shape={ev_year.shape}")
        if ev_year.shape[2] < 8:
            raise ValueError(f"EV 数据最后一维字段不足，当前 shape={ev_year.shape}")

        self.ev_year = ev_year.astype(np.float32)
        self.n_days = self.ev_year.shape[0]
        self.n_evs = self.ev_year.shape[1]

        print(f"已加载全年 EV 数据: {self.ev_npy_path}")
        print(f"EV 数据形状: {self.ev_year.shape} = (天数, 车辆数, 特征数)")

    def __call__(self, case: DailyIESCase) -> Dict[str, np.ndarray]:
        day_idx = int(case.day_of_year) - 1
        if not (0 <= day_idx < self.n_days):
            raise IndexError(
                f"day_of_year={case.day_of_year} 超出 EV 数据范围，EV 数据天数={self.n_days}"
            )

        day_ev = self.ev_year[day_idx]

        arr_hour = day_ev[:, 0]
        dep_hour = day_ev[:, 1]

        arr_step = np.ceil(arr_hour).astype(np.int32)
        dep_step = np.ceil(dep_hour).astype(np.int32)

        arr_step = np.clip(arr_step, 0, 23)
        dep_step = np.clip(dep_step, 1, 24)
        dep_step = np.maximum(dep_step, arr_step + 1)
        dep_step = np.clip(dep_step, 1, 24)

        ev_data = {
            "arr_step": arr_step.astype(np.int32),
            "dep_step": dep_step.astype(np.int32),
            "init_soc": np.clip(day_ev[:, 2], 0.0, 1.0).astype(np.float32),
            "target_soc": np.clip(day_ev[:, 3], 0.0, 1.0).astype(np.float32),
            "v2g_flag": day_ev[:, 4].astype(np.int32),
            "cap_kwh": day_ev[:, 5].astype(np.float32),
            "p_ch_max_kw": day_ev[:, 6].astype(np.float32),
            "p_dis_max_kw": day_ev[:, 7].astype(np.float32),
        }

        self._validate_ev_data(ev_data)
        return ev_data

    @staticmethod
    def _validate_ev_data(ev_data: Dict[str, np.ndarray]) -> None:
        required_ev = [
            "arr_step", "dep_step", "init_soc", "target_soc",
            "cap_kwh", "p_ch_max_kw", "p_dis_max_kw", "v2g_flag",
        ]
        for k in required_ev:
            if k not in ev_data:
                raise KeyError(f"EV 数据缺少字段: {k}")

        n = len(ev_data["arr_step"])
        for k in required_ev:
            if len(ev_data[k]) != n:
                raise ValueError(f"EV 字段长度不一致: {k}")


class YearlyCaseEnv(gym.Env):
    """
    用于训练阶段的上层环境封装。

    - 一个 episode = 一天 24h
    - 训练时：从 train_cases 中按天打乱、无放回抽取
    - 验证/测试时：按 val/test_cases 的固定自然时间顺序逐天取样
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        loader: YearlyCSVDataLoader,
        split: str,
        ev_provider: Callable[[DailyIESCase], Dict[str, np.ndarray]],
        cfg: Optional[IESConfig] = None,
        shuffle_train: bool = True,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.loader = loader
        self.split = split.strip().lower()
        self.ev_provider = ev_provider
        self.cfg = cfg if cfg is not None else IESConfig()
        self.shuffle_train = shuffle_train
        self.render_mode = render_mode

        self.cases = self.loader.get_cases(self.split)
        if not self.cases:
            raise ValueError(f"{self.split} cases 为空，请先 load 数据")

        self.rng = np.random.default_rng(seed)
        self._order: list[int] = []
        self._cursor = 0

        self.current_case: Optional[DailyIESCase] = None
        self.inner_env: Optional[ParkIESEnv] = None

        first_case = self.cases[0]
        first_ev = self.ev_provider(first_case)
        temp_env = ParkIESEnv(
            cfg=self.cfg,
            ts_data=first_case.ts_data,
            ev_data=first_ev,
            render_mode=self.render_mode,
        )
        self.action_space = temp_env.action_space
        self.observation_space = temp_env.observation_space

        self._refill_order()

    def _refill_order(self) -> None:
        n = len(self.cases)
        if self.split == "train" and self.shuffle_train:
            self._order = self.rng.permutation(n).tolist()
        else:
            self._order = list(range(n))
        self._cursor = 0

    def _next_case(self) -> DailyIESCase:
        if self._cursor >= len(self._order):
            self._refill_order()
        idx = self._order[self._cursor]
        self._cursor += 1
        return self.cases[idx]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_case = self._next_case()
        ev_data = self.ev_provider(self.current_case)

        self.inner_env = ParkIESEnv(
            cfg=deepcopy(self.cfg),
            ts_data=self.current_case.ts_data,
            ev_data=ev_data,
            render_mode=self.render_mode,
        )

        obs, info = self.inner_env.reset(seed=seed, options=options)
        info = dict(info)
        info.update(
            {
                "month": self.current_case.month,
                "day_of_year": self.current_case.day_of_year,
                "season": self.current_case.season,
                "split": self.current_case.set_name,
            }
        )
        return obs, info

    def step(self, action):
        if self.inner_env is None:
            raise RuntimeError("请先调用 reset()")

        obs, reward, terminated, truncated, info = self.inner_env.step(action)
        info = dict(info)
        if self.current_case is not None:
            info.update(
                {
                    "month": self.current_case.month,
                    "day_of_year": self.current_case.day_of_year,
                    "season": self.current_case.season,
                    "split": self.current_case.set_name,
                }
            )
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.inner_env is not None:
            return self.inner_env.render()
        return None

    def close(self):
        if self.inner_env is not None:
            self.inner_env.close()
