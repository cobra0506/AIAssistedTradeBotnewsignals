"""
Strategy: Simple Multi-Timeframe (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Use 5m EMA trend filter: price above trend EMA = bullish, below = bearish.
- On 1m timeframe:
  - If bullish trend and fast EMA > slow EMA -> OPEN_LONG.
  - If bearish trend and fast EMA < slow EMA -> OPEN_SHORT.
- If in a position and the opposite conditions appear -> CLOSE the position.
- HOLD otherwise.
"""

import os
import sys
import logging
from typing import Dict, List, Any

import pandas as pd

from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.strategies.indicators_library import ema

# Add parent directories to path for proper imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

STRATEGY_PARAMETERS = {
    'fast_ema_period': {
        'type': 'int',
        'default': 9,
        'min': 5,
        'max': 20,
        'description': 'Fast EMA period for entry signals',
        'gui_hint': 'For 1m timeframe. Lower = more sensitive'
    },
    'slow_ema_period': {
        'type': 'int',
        'default': 21,
        'min': 15,
        'max': 50,
        'description': 'Slow EMA period for entry signals',
        'gui_hint': 'For 1m timeframe. Higher = smoother'
    },
    'trend_ema_period': {
        'type': 'int',
        'default': 50,
        'min': 20,
        'max': 100,
        'description': 'Trend EMA period for higher timeframe',
        'gui_hint': 'For 5m timeframe. Higher = longer term trend'
    },
    'trend_timeframe': {
        'type': 'str',
        'default': '5m',
        'options': ['5m', '15m', '30m'],
        'description': 'Higher timeframe for trend confirmation',
        'gui_hint': 'Default is 5m'
    },
    'entry_timeframe': {
        'type': 'str',
        'default': '1m',
        'options': ['1m', '3m', '5m'],
        'description': 'Entry timeframe for signals',
        'gui_hint': 'Default is 1m'
    }
}


class SimpleMultiTimeframeStrategy(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Simple_Multi_Timeframe",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )

        self.fast_ema_period = config.get('fast_ema_period', 9)
        self.slow_ema_period = config.get('slow_ema_period', 21)
        self.trend_ema_period = config.get('trend_ema_period', 50)
        self.trend_timeframe = config.get('trend_timeframe', '5m')
        self.entry_timeframe = config.get('entry_timeframe', '1m')

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
        min_periods = max(self.slow_ema_period, self.trend_ema_period) + 2

        for symbol in data:
            signals[symbol] = {}
            for timeframe, df in data[symbol].items():
                if df is None or len(df) < min_periods:
                    signals[symbol][timeframe] = 'HOLD'
                    continue

                if timeframe != self.entry_timeframe:
                    signals[symbol][timeframe] = 'HOLD'
                    continue

                close = df['close']
                ema_fast = ema(close, period=self.fast_ema_period)
                ema_slow = ema(close, period=self.slow_ema_period)

                fast_now = ema_fast.iloc[-1]
                fast_prev = ema_fast.iloc[-2]
                slow_now = ema_slow.iloc[-1]
                slow_prev = ema_slow.iloc[-2]

                # Trend filter (higher timeframe)
                bullish_trend = True
                trend_df = data.get(symbol, {}).get(self.trend_timeframe)
                if trend_df is not None and len(trend_df) >= self.trend_ema_period + 1:
                    trend_close = trend_df['close']
                    trend_ema = ema(trend_close, period=self.trend_ema_period)
                    bullish_trend = trend_close.iloc[-1] > trend_ema.iloc[-1]

                bullish_cross = fast_now > slow_now and fast_prev <= slow_prev
                bearish_cross = fast_now < slow_now and fast_prev >= slow_prev

                position_key = (symbol, timeframe)
                position = self._position_state.get(position_key)

                if bullish_trend and bullish_cross:
                    raw_signal = 'OPEN_LONG'
                elif (not bullish_trend) and bearish_cross:
                    raw_signal = 'OPEN_SHORT'
                elif position and not position.get('is_short', False) and bearish_cross:
                    raw_signal = 'CLOSE_LONG'
                elif position and position.get('is_short', False) and bullish_cross:
                    raw_signal = 'CLOSE_SHORT'
                else:
                    raw_signal = 'HOLD'

                signals[symbol][timeframe] = self._apply_position_rules(position_key, raw_signal)

        return signals


def create_strategy(symbols=None, timeframes=None, **params):
    if symbols is None or len(symbols) == 0:
        symbols = ['BTCUSDT']
    if timeframes is None or len(timeframes) == 0:
        timeframes = ['1m', '5m']

    trend_timeframe = params.get('trend_timeframe', '5m')
    entry_timeframe = params.get('entry_timeframe', '1m')
    if trend_timeframe not in timeframes:
        timeframes.append(trend_timeframe)
    if entry_timeframe not in timeframes:
        timeframes.append(entry_timeframe)

    config = {
        'fast_ema_period': params.get('fast_ema_period', 9),
        'slow_ema_period': params.get('slow_ema_period', 21),
        'trend_ema_period': params.get('trend_ema_period', 50),
        'trend_timeframe': trend_timeframe,
        'entry_timeframe': entry_timeframe
    }
    return SimpleMultiTimeframeStrategy(symbols, timeframes, config)
