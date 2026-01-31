"""
Strategy: Simple Frequent Trading (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Use fast/slow SMA crossover for direction.
- Use relaxed RSI extremes for extra confirmation.
- Use tiny price change as an additional directional nudge.
- If the combined signal is bullish -> OPEN_LONG.
- If the combined signal is bearish -> OPEN_SHORT.
- If the signal flips against the current position -> CLOSE position.
- HOLD otherwise.
"""

import os
import sys
import logging
from typing import Dict, List, Any

import pandas as pd

from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.strategies.indicators_library import sma, rsi

# Add parent directories to path for proper imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

STRATEGY_PARAMETERS = {
    'fast_ma_period': {
        'type': 'int',
        'default': 3,
        'min': 2,
        'max': 10,
        'description': 'Fast SMA period for entry signals',
        'gui_hint': 'Lower values = more frequent signals'
    },
    'slow_ma_period': {
        'type': 'int',
        'default': 8,
        'min': 5,
        'max': 20,
        'description': 'Slow SMA period for entry signals',
        'gui_hint': 'Higher values = smoother signals'
    },
    'rsi_period': {
        'type': 'int',
        'default': 7,
        'min': 3,
        'max': 14,
        'description': 'RSI period for signal confirmation',
        'gui_hint': 'Lower values = more responsive'
    },
    'rsi_overbought': {
        'type': 'int',
        'default': 60,
        'min': 55,
        'max': 70,
        'description': 'RSI overbought threshold',
        'gui_hint': 'Lower = more sell signals'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 40,
        'min': 30,
        'max': 45,
        'description': 'RSI oversold threshold',
        'gui_hint': 'Higher = more buy signals'
    },
    'price_change_threshold': {
        'type': 'float',
        'default': 0.001,
        'min': 0.0005,
        'max': 0.005,
        'description': 'Price change threshold for signals',
        'gui_hint': 'Lower = more frequent signals'
    }
}


class SimpleFrequentTradingStrategy(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Simple_Frequent_Trading",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )

        self.fast_ma_period = config.get('fast_ma_period', 3)
        self.slow_ma_period = config.get('slow_ma_period', 8)
        self.rsi_period = config.get('rsi_period', 7)
        self.rsi_overbought = config.get('rsi_overbought', 60)
        self.rsi_oversold = config.get('rsi_oversold', 40)
        self.price_change_threshold = config.get('price_change_threshold', 0.001)

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
        min_periods = max(self.fast_ma_period, self.slow_ma_period, self.rsi_period) + 2

        for symbol in data:
            signals[symbol] = {}
            for timeframe, df in data[symbol].items():
                if df is None or len(df) < min_periods:
                    signals[symbol][timeframe] = 'HOLD'
                    continue

                close = df['close']

                fast_ma = sma(close, period=self.fast_ma_period)
                slow_ma = sma(close, period=self.slow_ma_period)
                rsi_series = rsi(close, period=self.rsi_period)

                prev_close = close.iloc[-2]
                last_close = close.iloc[-1]

                price_change = (last_close - prev_close) / prev_close if prev_close else 0.0

                # Build simple signals
                ma_signal = 0
                if fast_ma.iloc[-1] > slow_ma.iloc[-1] and fast_ma.iloc[-2] <= slow_ma.iloc[-2]:
                    ma_signal = 1
                elif fast_ma.iloc[-1] < slow_ma.iloc[-1] and fast_ma.iloc[-2] >= slow_ma.iloc[-2]:
                    ma_signal = -1

                rsi_signal = 0
                if rsi_series.iloc[-1] < self.rsi_oversold:
                    rsi_signal = 1
                elif rsi_series.iloc[-1] > self.rsi_overbought:
                    rsi_signal = -1

                price_signal = 0
                if price_change > self.price_change_threshold:
                    price_signal = 1
                elif price_change < -self.price_change_threshold:
                    price_signal = -1

                combined_signal = ma_signal + rsi_signal + price_signal

                position_key = (symbol, timeframe)
                position = self._position_state.get(position_key)

                if position:
                    if not position.get('is_short', False) and combined_signal < 0:
                        signals[symbol][timeframe] = self._apply_position_rules(position_key, 'CLOSE_LONG')
                        continue
                    if position.get('is_short', False) and combined_signal > 0:
                        signals[symbol][timeframe] = self._apply_position_rules(position_key, 'CLOSE_SHORT')
                        continue

                if combined_signal > 0:
                    raw_signal = 'OPEN_LONG'
                elif combined_signal < 0:
                    raw_signal = 'OPEN_SHORT'
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
        'fast_ma_period': params.get('fast_ma_period', 3),
        'slow_ma_period': params.get('slow_ma_period', 8),
        'rsi_period': params.get('rsi_period', 7),
        'rsi_overbought': params.get('rsi_overbought', 60),
        'rsi_oversold': params.get('rsi_oversold', 40),
        'price_change_threshold': params.get('price_change_threshold', 0.001)
    }
    return SimpleFrequentTradingStrategy(symbols, timeframes, config)
