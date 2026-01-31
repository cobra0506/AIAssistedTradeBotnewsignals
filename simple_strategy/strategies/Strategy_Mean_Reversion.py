"""
Strategy: Mean Reversion (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Build Bollinger-style bands using SMA and standard deviation.
- OPEN_LONG when price closes below the lower band.
- OPEN_SHORT when price closes above the upper band.
- CLOSE_LONG when price crosses back above the lower exit band.
- CLOSE_SHORT when price crosses back below the upper exit band.
- HOLD otherwise.
"""

import os
import sys
import logging
from typing import Dict, List, Any

import pandas as pd

from simple_strategy.shared.strategy_base import StrategyBase

# Add parent directories to path for proper imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

STRATEGY_PARAMETERS = {
    'lookback_period': {
        'type': 'int',
        'default': 20,
        'min': 5,
        'max': 100,
        'description': 'SMA lookback period for Bollinger bands',
        'gui_hint': 'Typical values: 20'
    },
    'entry_threshold': {
        'type': 'float',
        'default': 2.0,
        'min': 1.0,
        'max': 3.0,
        'description': 'Std-dev multiplier for entry bands',
        'gui_hint': 'Typical values: 2.0'
    },
    'exit_threshold': {
        'type': 'float',
        'default': 0.5,
        'min': 0.1,
        'max': 2.0,
        'description': 'Std-dev multiplier for exit bands',
        'gui_hint': 'Typical values: 0.5-1.0'
    }
}


class MeanReversionStrategy(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Mean_Reversion_Bollinger",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )
        self.lookback_period = config.get('lookback_period', 20)
        self.entry_threshold = config.get('entry_threshold', 2.0)
        self.exit_threshold = config.get('exit_threshold', 0.5)
        self._position_state: Dict[tuple, Dict[str, Any]] = {}

    def _apply_position_rules(self, position_key: tuple, raw_signal: str) -> str:
        position = self._position_state.get(position_key)

        if raw_signal == 'OPEN_LONG':
            if position is not None:
                return 'HOLD'
            self._position_state[position_key] = {'is_short': False}
            return raw_signal

        if raw_signal == 'OPEN_SHORT':
            if position is not None:
                return 'HOLD'
            self._position_state[position_key] = {'is_short': True}
            return raw_signal

        if raw_signal == 'CLOSE_LONG':
            if position is None or position.get('is_short', False):
                return 'HOLD'
            self._position_state.pop(position_key, None)
            return raw_signal

        if raw_signal == 'CLOSE_SHORT':
            if position is None or not position.get('is_short', False):
                return 'HOLD'
            self._position_state.pop(position_key, None)
            return raw_signal

        return 'HOLD'

    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        signals: Dict[str, Dict[str, str]] = {}
        min_periods = self.lookback_period + 2

        for symbol in data:
            signals[symbol] = {}
            for timeframe, df in data[symbol].items():
                if df is None or len(df) < min_periods:
                    signals[symbol][timeframe] = 'HOLD'
                    continue

                close = df['close']
                sma = close.rolling(window=self.lookback_period).mean()
                std = close.rolling(window=self.lookback_period).std()

                upper_band = sma + (std * self.entry_threshold)
                lower_band = sma - (std * self.entry_threshold)
                exit_upper = sma + (std * self.exit_threshold)
                exit_lower = sma - (std * self.exit_threshold)

                prev_close = close.iloc[-2]
                last_close = close.iloc[-1]

                position_key = (symbol, timeframe)
                position = self._position_state.get(position_key)
                is_short = position.get('is_short') if position else None

                if last_close < lower_band.iloc[-1]:
                    raw_signal = 'OPEN_LONG'
                elif last_close > upper_band.iloc[-1]:
                    raw_signal = 'OPEN_SHORT'
                elif prev_close < exit_lower.iloc[-2] and last_close > exit_lower.iloc[-1]:
                    raw_signal = 'CLOSE_LONG'
                elif prev_close > exit_upper.iloc[-2] and last_close < exit_upper.iloc[-1]:
                    raw_signal = 'CLOSE_SHORT'
                else:
                    raw_signal = 'HOLD'

                signals[symbol][timeframe] = self._apply_position_rules(position_key, raw_signal)

        return signals


def create_strategy(symbols=None, timeframes=None, **params):
    if symbols is None or len(symbols) == 0:
        symbols = ['BTCUSDT']
    if timeframes is None or len(timeframes) == 0:
        timeframes = ['1m']

    config = {
        'lookback_period': params.get('lookback_period', 20),
        'entry_threshold': params.get('entry_threshold', 2.0),
        'exit_threshold': params.get('exit_threshold', 0.5)
    }
    return MeanReversionStrategy(symbols, timeframes, config)
