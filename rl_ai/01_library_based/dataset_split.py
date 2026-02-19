from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    min_rows_per_split: int = 10

    def validate(self):
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        if self.min_rows_per_split < 1:
            raise ValueError("min_rows_per_split must be >= 1")
        if min(self.train_ratio, self.validation_ratio, self.test_ratio) <= 0:
            raise ValueError("All split ratios must be > 0")


def split_time_series(
    data: pd.DataFrame,
    config: SplitConfig | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if data is None or data.empty:
        raise ValueError("Cannot split empty dataset")

    cfg = config or SplitConfig()
    cfg.validate()

    total_rows = len(data)
    min_required = cfg.min_rows_per_split * 3
    if total_rows < min_required:
        raise ValueError(
            f"Not enough rows to split. Need at least {min_required}, got {total_rows}"
        )

    train_end = int(total_rows * cfg.train_ratio)
    validation_end = train_end + int(total_rows * cfg.validation_ratio)

    # Enforce minimum rows for each segment while preserving order.
    train_end = max(train_end, cfg.min_rows_per_split)
    validation_end = max(validation_end, train_end + cfg.min_rows_per_split)
    validation_end = min(validation_end, total_rows - cfg.min_rows_per_split)
    train_end = min(train_end, validation_end - cfg.min_rows_per_split)

    if train_end < cfg.min_rows_per_split:
        raise ValueError("Training split became too small after constraints")
    if validation_end - train_end < cfg.min_rows_per_split:
        raise ValueError("Validation split became too small after constraints")
    if total_rows - validation_end < cfg.min_rows_per_split:
        raise ValueError("Test split became too small after constraints")

    train_data = data.iloc[:train_end].copy().reset_index(drop=True)
    validation_data = data.iloc[train_end:validation_end].copy().reset_index(drop=True)
    test_data = data.iloc[validation_end:].copy().reset_index(drop=True)

    return train_data, validation_data, test_data
