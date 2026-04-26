from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DailyIESCase:
    month: int
    day_of_year: int
    season: str
    set_name: str
    ts_data: Dict[str, np.ndarray]


class YearlyCSVDataLoader:
    """
    Read yearly_data_sci.csv and build daily cases.

    The CSV is expected to provide only two coarse splits via the Set column:
    `train` and `test`.

    We further split the original train portion into:
    - train: used for policy updates
    - val: fixed validation set used by EvalCallback

    The recommended policy is to reserve the last `val_days_per_month`
    training days of each month as validation days. This keeps validation
    samples fixed, seasonal, and disjoint from the final test set.
    """

    REQUIRED_COLUMNS = [
        "Month",
        "Day",
        "Hour",
        "Season",
        "Electric Load",
        "Heat Load",
        "Cold Load",
        "PV",
        "Wind",
        "Set",
    ]

    def __init__(self, csv_path: str | Path, val_days_per_month: int = 2):
        self.csv_path = Path(csv_path)
        self.val_days_per_month = int(val_days_per_month)

        self.df: Optional[pd.DataFrame] = None
        self.train_cases: List[DailyIESCase] = []
        self.val_cases: List[DailyIESCase] = []
        self.test_cases: List[DailyIESCase] = []

    def load(self) -> Tuple[List[DailyIESCase], List[DailyIESCase], List[DailyIESCase]]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        self._validate_dataframe(df)
        df = df.sort_values(by=["Month", "Day", "Hour"]).reset_index(drop=True)

        self.df = df
        self.train_cases, self.val_cases, self.test_cases = self._build_daily_cases(df)
        return self.train_cases, self.val_cases, self.test_cases

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required CSV columns: {missing_cols}")

        if df[self.REQUIRED_COLUMNS].isnull().any().any():
            bad_cols = df[self.REQUIRED_COLUMNS].columns[df[self.REQUIRED_COLUMNS].isnull().any()].tolist()
            raise ValueError(f"CSV contains NaN in required columns: {bad_cols}")

        valid_sets = {"train", "test"}
        set_values = set(df["Set"].astype(str).str.strip().str.lower().unique())
        if not set_values.issubset(valid_sets):
            raise ValueError(f"Set column must be within {valid_sets}, got: {set_values}")

        grouped = df.groupby(["Month", "Day", "Set"], sort=True)
        sizes = grouped.size()
        bad_groups = sizes[sizes != 24]
        if not bad_groups.empty:
            raise ValueError(
                "Each daily sample must contain exactly 24 rows.\n"
                f"Invalid groups:\n{bad_groups.head()}"
            )

        for (month, day, set_name), g in grouped:
            hours = sorted(g["Hour"].astype(int).tolist())
            if hours != list(range(24)):
                raise ValueError(
                    f"Sample Month={month}, Day={day}, Set={set_name} does not contain Hour=0..23."
                )

    def _build_daily_cases(
        self, df: pd.DataFrame
    ) -> Tuple[List[DailyIESCase], List[DailyIESCase], List[DailyIESCase]]:
        coarse_train_cases: List[DailyIESCase] = []
        test_cases: List[DailyIESCase] = []

        grouped = df.groupby(["Month", "Day", "Set"], sort=True)
        for (month, day_of_year, set_name), g in grouped:
            g = g.sort_values("Hour").reset_index(drop=True)

            season = str(g["Season"].iloc[0]).strip().lower()
            set_name = str(set_name).strip().lower()

            ts_data = {
                "elec_load": g["Electric Load"].to_numpy(dtype=np.float32),
                "heat_load": g["Heat Load"].to_numpy(dtype=np.float32),
                "cool_load": g["Cold Load"].to_numpy(dtype=np.float32),
                "pv": g["PV"].to_numpy(dtype=np.float32),
                "wt": g["Wind"].to_numpy(dtype=np.float32),
            }

            case = DailyIESCase(
                month=int(month),
                day_of_year=int(day_of_year),
                season=season,
                set_name=set_name,
                ts_data=ts_data,
            )

            if set_name == "train":
                coarse_train_cases.append(case)
            else:
                test_cases.append(case)

        train_cases, val_cases = self._split_train_val_cases(coarse_train_cases)
        return train_cases, val_cases, test_cases

    def _split_train_val_cases(
        self, coarse_train_cases: List[DailyIESCase]
    ) -> Tuple[List[DailyIESCase], List[DailyIESCase]]:
        if self.val_days_per_month < 0:
            raise ValueError("val_days_per_month must be >= 0")

        train_cases: List[DailyIESCase] = []
        val_cases: List[DailyIESCase] = []

        month_to_cases: Dict[int, List[DailyIESCase]] = {}
        for case in coarse_train_cases:
            month_to_cases.setdefault(case.month, []).append(case)

        for month in sorted(month_to_cases):
            month_cases = sorted(month_to_cases[month], key=lambda item: item.day_of_year)

            if self.val_days_per_month == 0:
                train_cases.extend(month_cases)
                continue

            if len(month_cases) <= self.val_days_per_month:
                raise ValueError(
                    f"Month {month} has only {len(month_cases)} train days; "
                    f"cannot reserve {self.val_days_per_month} validation days."
                )

            split_idx = len(month_cases) - self.val_days_per_month
            month_train = month_cases[:split_idx]
            month_val = month_cases[split_idx:]

            train_cases.extend(month_train)
            val_cases.extend(replace(case, set_name="val") for case in month_val)

        return train_cases, val_cases

    def get_cases(self, split: str) -> List[DailyIESCase]:
        split = split.strip().lower()
        if split == "train":
            return self.train_cases
        if split == "val":
            return self.val_cases
        if split == "test":
            return self.test_cases
        raise ValueError("split must be one of: 'train', 'val', 'test'")

    def summary(self) -> None:
        if self.df is None:
            print("Please call load() first.")
            return

        print("========== YearlyCSVDataLoader Summary ==========")
        print(f"CSV path: {self.csv_path}")
        print(f"Total hourly rows: {len(self.df)}")
        print(f"Train days: {len(self.train_cases)}")
        print(f"Validation days: {len(self.val_cases)}")
        print(f"Test days: {len(self.test_cases)}")
        print(f"Validation days reserved per month: {self.val_days_per_month}")

        records = []
        for case in self.train_cases + self.val_cases + self.test_cases:
            records.append({"Month": case.month, "Set": case.set_name})

        stat = (
            pd.DataFrame(records)
            .groupby(["Month", "Set"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        print("\nPer-month train/val/test day counts:")
        print(stat)
        print("================================================")
