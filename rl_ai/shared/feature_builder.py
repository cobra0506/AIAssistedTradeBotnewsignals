from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .feature_schema import FeatureSchema


@dataclass
class AccountSnapshot:
    balance: float
    initial_balance: float
    has_position: bool = False
    is_short: bool = False
    entry_price: Optional[float] = None
    current_price: Optional[float] = None
    drawdown_pct: float = 0.0


class FeatureBuilder:
    """
    Creates fixed-order RL feature vectors from candles, indicators, and account state.
    """

    def __init__(self, schema: FeatureSchema):
        self.schema = schema

    def build_vector(
        self,
        data: pd.DataFrame,
        row_index: int,
        account: Optional[AccountSnapshot] = None,
    ) -> np.ndarray:
        if data is None or data.empty:
            raise ValueError("Input data is empty")

        if row_index < 0 or row_index >= len(data):
            raise IndexError(f"row_index {row_index} out of bounds for length {len(data)}")

        row = data.iloc[row_index]
        prev_row = data.iloc[row_index - 1] if row_index > 0 else row

        open_price = self._safe_float(row.get("open", 0.0))
        high_price = self._safe_float(row.get("high", 0.0))
        low_price = self._safe_float(row.get("low", 0.0))
        close_price = self._safe_float(row.get("close", 0.0))
        volume = self._safe_float(row.get("volume", 0.0))

        prev_close = self._safe_float(prev_row.get("close", close_price))
        prev_volume = self._safe_float(prev_row.get("volume", volume))

        return_1 = self._pct_change(close_price, prev_close)
        hl_spread = (high_price - low_price) / close_price if close_price > 0 else 0.0
        oc_change = (close_price - open_price) / open_price if open_price > 0 else 0.0
        volume_change_1 = self._pct_change(volume, prev_volume)

        feature_values = [
            open_price,
            high_price,
            low_price,
            close_price,
            volume,
            return_1,
            hl_spread,
            oc_change,
            volume_change_1,
        ]

        for indicator_name in self.schema.indicator_features:
            if indicator_name in row and pd.notna(row[indicator_name]):
                feature_values.append(self._safe_float(row[indicator_name]))
            else:
                feature_values.append(0.0)

        feature_values.extend(self._build_account_features(account, close_price))

        return np.asarray(feature_values, dtype=np.float32)

    @staticmethod
    def _safe_float(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _pct_change(current_value: float, previous_value: float) -> float:
        if previous_value == 0:
            return 0.0
        return (current_value / previous_value) - 1.0

    def _build_account_features(self, account: Optional[AccountSnapshot], close_price: float):
        if account is None:
            return [0.0, 0.0, 0.0, 0.0, 1.0]

        has_position = 1.0 if account.has_position else 0.0
        is_short = 1.0 if account.is_short else 0.0

        current_price = close_price if account.current_price is None else account.current_price
        unrealized_pnl_pct = 0.0
        if (
            account.has_position
            and account.entry_price is not None
            and account.entry_price > 0
            and current_price is not None
        ):
            direction = -1.0 if account.is_short else 1.0
            unrealized_pnl_pct = direction * ((float(current_price) - account.entry_price) / account.entry_price)

        drawdown_pct = max(0.0, float(account.drawdown_pct))

        if account.initial_balance and account.initial_balance > 0:
            balance_ratio = float(account.balance) / float(account.initial_balance)
        else:
            balance_ratio = 1.0

        return [has_position, is_short, unrealized_pnl_pct, drawdown_pct, balance_ratio]
