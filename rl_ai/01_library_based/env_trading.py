from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from rl_ai.shared.feature_builder import AccountSnapshot, FeatureBuilder
from rl_ai.shared.feature_schema import FeatureSchema
from rl_ai.shared.reward import calculate_reward
from rl_ai.shared.signal_adapter import ACTION_TO_SIGNAL, SignalAdapter

from .config import RLTradingConfig


class RLTradingEnv:
    """
    Lightweight RL trading environment for historical candle replay.

    This environment is intentionally simple for Phase 1:
    - Single active position at a time
    - Fixed signal schema via SignalAdapter
    - Reward from equity change, drawdown growth, and fees
    """

    REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[RLTradingConfig] = None,
        schema: Optional[FeatureSchema] = None,
    ):
        if data is None or data.empty:
            raise ValueError("Environment requires non-empty candle data")

        missing = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.data = data.copy().reset_index(drop=True)
        self.config = config or RLTradingConfig()

        if schema is None:
            schema = FeatureSchema(indicator_features=list(self.config.indicator_columns))
        self.schema = schema
        self.feature_builder = FeatureBuilder(schema=self.schema)

        self.current_step = 0
        self.balance = 0.0
        self.equity = 0.0
        self.peak_equity = 0.0
        self.current_drawdown_pct = 0.0
        self.max_drawdown_pct = 0.0
        self.position: Optional[Dict[str, float]] = None
        self.last_signal = "HOLD"
        self.reset()

    @property
    def action_space_size(self) -> int:
        return len(ACTION_TO_SIGNAL)

    @property
    def observation_size(self) -> int:
        return self.schema.size

    def reset(self) -> Tuple[np.ndarray, Dict[str, float]]:
        self.current_step = max(1, int(self.config.min_bars_warmup))
        self.current_step = min(self.current_step, len(self.data) - 1)

        self.balance = float(self.config.initial_balance)
        self.equity = float(self.config.initial_balance)
        self.peak_equity = float(self.config.initial_balance)
        self.current_drawdown_pct = 0.0
        self.max_drawdown_pct = 0.0
        self.position = None
        self.last_signal = "HOLD"

        observation = self._get_observation()
        info = self._build_info(
            signal="HOLD",
            action=0,
            trade_executed=False,
            fee_paid=0.0,
            realized_pnl=0.0,
        )
        return observation, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.current_step >= len(self.data) - 1:
            observation = self._get_observation()
            info = self._build_info(
                signal="HOLD",
                action=action,
                trade_executed=False,
                fee_paid=0.0,
                realized_pnl=0.0,
                reason="end_of_data",
            )
            return observation, 0.0, True, info

        current_price = float(self.data.iloc[self.current_step]["close"])
        previous_equity = self._mark_to_market_equity(current_price)
        previous_drawdown_pct = self.current_drawdown_pct

        signal = SignalAdapter.action_to_signal(
            action,
            has_position=self.position is not None,
            is_short=bool(self.position and self.position.get("is_short", False)),
        )
        self.last_signal = signal

        realized_pnl, fee_paid, trade_executed = self._execute_signal(signal, current_price)

        self.current_step += 1
        next_price = float(self.data.iloc[self.current_step]["close"])
        self.equity = self._mark_to_market_equity(next_price)
        self.peak_equity = max(self.peak_equity, self.equity)

        if self.peak_equity > 0:
            self.current_drawdown_pct = max(0.0, (self.peak_equity - self.equity) / self.peak_equity)
        else:
            self.current_drawdown_pct = 0.0

        self.max_drawdown_pct = max(self.max_drawdown_pct, self.current_drawdown_pct)

        reward_breakdown = calculate_reward(
            previous_equity=previous_equity,
            current_equity=self.equity,
            previous_drawdown_pct=previous_drawdown_pct,
            current_drawdown_pct=self.current_drawdown_pct,
            fee_paid=fee_paid,
            trade_executed=trade_executed,
            config=self.config.reward_config,
        )

        done = (
            self.current_step >= len(self.data) - 1
            or self.current_drawdown_pct >= float(self.config.max_drawdown_stop_pct)
            or self.equity <= 0
        )

        observation = self._get_observation()
        info = self._build_info(
            signal=signal,
            action=action,
            trade_executed=trade_executed,
            fee_paid=fee_paid,
            realized_pnl=realized_pnl,
        )
        info["reward_breakdown"] = reward_breakdown.as_dict()

        return observation, float(reward_breakdown.total), done, info

    def _execute_signal(self, signal: str, price: float) -> Tuple[float, float, bool]:
        if price <= 0:
            return 0.0, 0.0, False

        if signal in ("OPEN_LONG", "OPEN_SHORT"):
            if self.position is not None:
                return 0.0, 0.0, False

            size_fraction = min(max(float(self.config.position_size_fraction), 0.0), 1.0)
            notional = max(self.balance, 0.0) * size_fraction
            if notional <= 0:
                return 0.0, 0.0, False

            quantity = notional / price
            fee_paid = notional * float(self.config.fee_pct)
            self.balance -= fee_paid

            self.position = {
                "is_short": signal == "OPEN_SHORT",
                "entry_price": price,
                "quantity": quantity,
            }
            return 0.0, fee_paid, True

        if signal in ("CLOSE_LONG", "CLOSE_SHORT"):
            if self.position is None:
                return 0.0, 0.0, False

            is_short = bool(self.position.get("is_short", False))
            if signal == "CLOSE_LONG" and is_short:
                return 0.0, 0.0, False
            if signal == "CLOSE_SHORT" and not is_short:
                return 0.0, 0.0, False

            entry_price = float(self.position["entry_price"])
            quantity = float(self.position["quantity"])
            notional_close = quantity * price
            fee_paid = notional_close * float(self.config.fee_pct)

            if is_short:
                gross_pnl = (entry_price - price) * quantity
            else:
                gross_pnl = (price - entry_price) * quantity

            realized_pnl = gross_pnl - fee_paid
            self.balance += realized_pnl
            self.position = None
            return realized_pnl, fee_paid, True

        return 0.0, 0.0, False

    def _unrealized_pnl(self, price: float) -> float:
        if self.position is None:
            return 0.0

        entry_price = float(self.position["entry_price"])
        quantity = float(self.position["quantity"])
        is_short = bool(self.position.get("is_short", False))

        if is_short:
            return (entry_price - price) * quantity
        return (price - entry_price) * quantity

    def _mark_to_market_equity(self, price: float) -> float:
        return float(self.balance + self._unrealized_pnl(price))

    def _get_observation(self) -> np.ndarray:
        close_price = float(self.data.iloc[self.current_step]["close"])
        account = AccountSnapshot(
            balance=self.balance,
            initial_balance=float(self.config.initial_balance),
            has_position=self.position is not None,
            is_short=bool(self.position and self.position.get("is_short", False)),
            entry_price=float(self.position["entry_price"]) if self.position else None,
            current_price=close_price,
            drawdown_pct=self.current_drawdown_pct,
        )
        return self.feature_builder.build_vector(self.data, self.current_step, account)

    def _build_info(
        self,
        signal: str,
        action,
        trade_executed: bool,
        fee_paid: float,
        realized_pnl: float,
        reason: str = "",
    ) -> Dict:
        close_price = float(self.data.iloc[self.current_step]["close"])
        unrealized_pnl = self._unrealized_pnl(close_price)
        return {
            "signal": signal,
            "action": SignalAdapter.normalize_action(action),
            "trade_executed": bool(trade_executed),
            "fee_paid": float(fee_paid),
            "realized_pnl": float(realized_pnl),
            "unrealized_pnl": float(unrealized_pnl),
            "balance": float(self.balance),
            "equity": float(self.equity),
            "drawdown_pct": float(self.current_drawdown_pct),
            "max_drawdown_pct": float(self.max_drawdown_pct),
            "has_position": bool(self.position is not None),
            "is_short": bool(self.position and self.position.get("is_short", False)),
            "current_step": int(self.current_step),
            "reason": reason,
        }
