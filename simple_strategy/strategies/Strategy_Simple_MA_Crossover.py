"""
Strategy: Simple MA Crossover (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Use 5m EMA trend filter: price above trend EMA = bullish, below = bearish.
- On 1m timeframe:
  - If bullish trend and fast EMA > slow EMA and RSI is bullish -> OPEN_LONG.
  - If bearish trend and fast EMA < slow EMA and RSI is bearish -> OPEN_SHORT.
- If in a position and the opposite conditions appear -> CLOSE the position.
- HOLD otherwise.
"""

import os
import sys
import logging
from typing import Dict, List, Any

import pandas as pd

from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.strategies.indicators_library import ema, rsi

# Add parent directories to path for proper imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

STRATEGY_PARAMETERS = {
    'fast_ma_period': {
        'type': 'int',
        'default': 5,
        'min': 3,
        'max': 10,
        'description': 'Fast EMA period for entry signals',
        'gui_hint': 'For 1m timeframe. Lower = more responsive'
    },
    'slow_ma_period': {
        'type': 'int',
        'default': 15,
        'min': 10,
        'max': 30,
        'description': 'Slow EMA period for entry signals',
        'gui_hint': 'For 1m timeframe. Higher = smoother'
    },
    'trend_ma_period': {
        'type': 'int',
        'default': 50,
        'min': 20,
        'max': 100,
        'description': 'Trend EMA period for higher timeframe',
        'gui_hint': 'For 5m timeframe. Higher = longer term trend'
    },
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 7,
        'max': 21,
        'description': 'RSI period for signal confirmation',
        'gui_hint': 'Standard values: 14, 10 for faster signals'
    },
    'rsi_bullish_threshold': {
        'type': 'int',
        'default': 55,
        'min': 50,
        'max': 60,
        'description': 'RSI threshold for bullish confirmation',
        'gui_hint': 'Above this level confirms bullish signals'
    },
    'rsi_bearish_threshold': {
        'type': 'int',
        'default': 45,
        'min': 40,
        'max': 50,
        'description': 'RSI threshold for bearish confirmation',
        'gui_hint': 'Below this level confirms bearish signals'
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


class ImprovedSimpleMACrossoverStrategy(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Improved_Simple_MA_Crossover",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )

        self.fast_ma_period = config.get('fast_ma_period', 5)
        self.slow_ma_period = config.get('slow_ma_period', 15)
        self.trend_ma_period = config.get('trend_ma_period', 50)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_bullish_threshold = config.get('rsi_bullish_threshold', 55)
        self.rsi_bearish_threshold = config.get('rsi_bearish_threshold', 45)
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
        min_periods = max(self.slow_ma_period, self.rsi_period, self.trend_ma_period) + 2

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
                ema_fast = ema(close, period=self.fast_ma_period)
                ema_slow = ema(close, period=self.slow_ma_period)
                rsi_series = rsi(close, period=self.rsi_period)

                # Trend filter (higher timeframe)
                bullish_trend = True
                trend_df = data.get(symbol, {}).get(self.trend_timeframe)
                if trend_df is not None and len(trend_df) >= self.trend_ma_period + 1:
                    trend_close = trend_df['close']
                    trend_ema = ema(trend_close, period=self.trend_ma_period)
                    bullish_trend = trend_close.iloc[-1] > trend_ema.iloc[-1]

                fast_now = ema_fast.iloc[-1]
                fast_prev = ema_fast.iloc[-2]
                slow_now = ema_slow.iloc[-1]
                slow_prev = ema_slow.iloc[-2]
                rsi_now = rsi_series.iloc[-1]

                bullish_setup = fast_now > slow_now and fast_prev <= slow_prev and rsi_now > self.rsi_bullish_threshold
                bearish_setup = fast_now < slow_now and fast_prev >= slow_prev and rsi_now < self.rsi_bearish_threshold

                position_key = (symbol, timeframe)
                position = self._position_state.get(position_key)

                if bullish_trend and bullish_setup:
                    raw_signal = 'OPEN_LONG'
                elif (not bullish_trend) and bearish_setup:
                    raw_signal = 'OPEN_SHORT'
                elif position and not position.get('is_short', False) and bearish_setup:
                    raw_signal = 'CLOSE_LONG'
                elif position and position.get('is_short', False) and bullish_setup:
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
        'fast_ma_period': params.get('fast_ma_period', 5),
        'slow_ma_period': params.get('slow_ma_period', 15),
        'trend_ma_period': params.get('trend_ma_period', 50),
        'rsi_period': params.get('rsi_period', 14),
        'rsi_bullish_threshold': params.get('rsi_bullish_threshold', 55),
        'rsi_bearish_threshold': params.get('rsi_bearish_threshold', 45),
        'trend_timeframe': trend_timeframe,
        'entry_timeframe': entry_timeframe
    }
    return ImprovedSimpleMACrossoverStrategy(symbols, timeframes, config)
