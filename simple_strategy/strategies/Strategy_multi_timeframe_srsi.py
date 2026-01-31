"""
Strategy: Multi-Timeframe SRSI (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Use RSI extremes (overbought/oversold) plus a fast/slow SMA crossover.
- Combine the RSI signal and the MA crossover signal (majority vote style).
- If combined signal is bullish -> OPEN_LONG.
- If combined signal is bearish -> OPEN_SHORT.
- If already in a position and the combined signal flips -> CLOSE the position.
- HOLD otherwise.
"""

import os
import sys
import logging
from typing import Dict, List, Any

import pandas as pd

from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.strategies.indicators_library import rsi, sma

# Add parent directories to path for proper imports when run directly
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

STRATEGY_PARAMETERS = {
    'oversold_threshold': {
        'type': 'int',
        'default': 20,
        'min': 5,
        'max': 30,
        'description': 'RSI oversold level (buy signal)',
        'gui_hint': 'Lower values = more conservative BUY signals. Recommended: 20'
    },
    'overbought_threshold': {
        'type': 'int',
        'default': 80,
        'min': 70,
        'max': 95,
        'description': 'RSI overbought level (sell signal)',
        'gui_hint': 'Higher values = more conservative SELL signals. Recommended: 80'
    },
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 7,
        'max': 21,
        'description': 'RSI calculation period',
        'gui_hint': 'Standard values: 14, 21. Lower = more sensitive'
    },
    'sma_fast_period': {
        'type': 'int',
        'default': 12,
        'min': 5,
        'max': 30,
        'description': 'Fast SMA period',
        'gui_hint': 'Lower = more responsive'
    },
    'sma_slow_period': {
        'type': 'int',
        'default': 26,
        'min': 10,
        'max': 50,
        'description': 'Slow SMA period',
        'gui_hint': 'Higher = smoother trend'
    }
}


class MultiTimeframeSRSIStrategy(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Multi_Timeframe_SRSI",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )

        self.oversold_threshold = config.get('oversold_threshold', 20)
        self.overbought_threshold = config.get('overbought_threshold', 80)
        self.rsi_period = config.get('rsi_period', 14)
        self.sma_fast_period = config.get('sma_fast_period', 12)
        self.sma_slow_period = config.get('sma_slow_period', 26)

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
        min_periods = max(self.rsi_period, self.sma_slow_period) + 2

        for symbol in data:
            signals[symbol] = {}
            for timeframe, df in data[symbol].items():
                if df is None or len(df) < min_periods:
                    signals[symbol][timeframe] = 'HOLD'
                    continue

                close = df['close']

                rsi_series = rsi(close, period=self.rsi_period)
                sma_fast = sma(close, period=self.sma_fast_period)
                sma_slow = sma(close, period=self.sma_slow_period)

                rsi_now = rsi_series.iloc[-1]
                sma_fast_now = sma_fast.iloc[-1]
                sma_fast_prev = sma_fast.iloc[-2]
                sma_slow_now = sma_slow.iloc[-1]
                sma_slow_prev = sma_slow.iloc[-2]

                rsi_signal = 0
                if rsi_now <= self.oversold_threshold:
                    rsi_signal = 1
                elif rsi_now >= self.overbought_threshold:
                    rsi_signal = -1

                ma_signal = 0
                if sma_fast_now > sma_slow_now and sma_fast_prev <= sma_slow_prev:
                    ma_signal = 1
                elif sma_fast_now < sma_slow_now and sma_fast_prev >= sma_slow_prev:
                    ma_signal = -1

                combined_signal = rsi_signal + ma_signal

                position_key = (symbol, timeframe)
                position = self._position_state.get(position_key)

                if combined_signal > 0:
                    raw_signal = 'OPEN_LONG'
                elif combined_signal < 0:
                    raw_signal = 'OPEN_SHORT'
                else:
                    raw_signal = 'HOLD'

                if position:
                    if not position.get('is_short', False) and combined_signal < 0:
                        raw_signal = 'CLOSE_LONG'
                    elif position.get('is_short', False) and combined_signal > 0:
                        raw_signal = 'CLOSE_SHORT'

                signals[symbol][timeframe] = self._apply_position_rules(position_key, raw_signal)

        return signals


def create_strategy(symbols=None, timeframes=None, **params):
    if symbols is None or len(symbols) == 0:
        symbols = ['BTCUSDT']
    if timeframes is None or len(timeframes) == 0:
        timeframes = ['1m', '5m', '15m']

    config = {
        'oversold_threshold': params.get('oversold_threshold', 20),
        'overbought_threshold': params.get('overbought_threshold', 80),
        'rsi_period': params.get('rsi_period', 14),
        'sma_fast_period': params.get('sma_fast_period', 12),
        'sma_slow_period': params.get('sma_slow_period', 26)
    }
    return MultiTimeframeSRSIStrategy(symbols, timeframes, config)
