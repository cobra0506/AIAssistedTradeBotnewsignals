"""
Fast Test Scalper Strategy
==========================

Simple, deterministic strategy intended to open and close positions quickly
so you can verify the paper trader/backtester execution path.

Logic:
- OPEN_LONG when the latest close is above the previous close.
- CLOSE_LONG when the latest close is below the previous close.

This intentionally creates frequent opens/closes to validate execution.
"""

import os
import sys
from typing import Dict, List, Any

import pandas as pd
import numpy as np

from simple_strategy.shared.strategy_base import StrategyBase

# Add parent directory to path for proper imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


STRATEGY_PARAMETERS = {
    'min_periods': {
        'type': 'int',
        'default': 2,
        'min': 2,
        'max': 10,
        'description': 'Minimum candles required before emitting signals',
        'gui_hint': 'Keep at 2 for fastest signal generation'
    }
}


class FastTestScalperStrategy(StrategyBase):
    """
    Generates rapid open/close signals for testing execution paths.
    """

    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Fast_Test_Scalper",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )
        self.min_periods = config.get('min_periods', 2)
        self._position_state = {}

    def _apply_position_rules(self, position_key, signal):
        position = self._position_state.get(position_key)
        if signal == 'OPEN_LONG':
            if position is not None:
                return 'HOLD'
            self._position_state[position_key] = {'is_short': False}
            return signal
        if signal == 'CLOSE_LONG':
            if position is None or position.get('is_short', False):
                return 'HOLD'
            self._position_state.pop(position_key, None)
            return signal
        return signal

    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        signals: Dict[str, Dict[str, str]] = {}
        for symbol in data:
            signals[symbol] = {}
            for timeframe, df in data[symbol].items():
                if df is None or len(df) < self.min_periods:
                    signals[symbol][timeframe] = 'HOLD'
                    continue

                prev_close = df['close'].iloc[-2]
                last_close = df['close'].iloc[-1]
                position_key = (symbol, timeframe)

                if last_close > prev_close:
                    raw_signal = 'OPEN_LONG'
                elif last_close < prev_close:
                    raw_signal = 'CLOSE_LONG'
                else:
                    raw_signal = 'HOLD'

                signals[symbol][timeframe] = self._apply_position_rules(position_key, raw_signal)

        return signals

    def generate_signals_vectorized(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.Series]]:
        signals: Dict[str, Dict[str, pd.Series]] = {}
        for symbol in data:
            signals[symbol] = {}
            for timeframe, df in data[symbol].items():
                if df is None or len(df) < self.min_periods:
                    signals[symbol][timeframe] = pd.Series(['HOLD'] * len(df), index=df.index)
                    continue

                close = df['close']
                raw = np.where(close > close.shift(1), 'OPEN_LONG',
                      np.where(close < close.shift(1), 'CLOSE_LONG', 'HOLD'))

                position_key = (symbol, timeframe)
                position = self._position_state.get(position_key)
                signals_list = []

                for raw_signal in raw:
                    if raw_signal == 'OPEN_LONG':
                        if position is None:
                            position = {'is_short': False}
                            signals_list.append('OPEN_LONG')
                        else:
                            signals_list.append('HOLD')
                    elif raw_signal == 'CLOSE_LONG':
                        if position is not None and not position.get('is_short', False):
                            position = None
                            signals_list.append('CLOSE_LONG')
                        else:
                            signals_list.append('HOLD')
                    else:
                        signals_list.append('HOLD')

                if position is None:
                    self._position_state.pop(position_key, None)
                else:
                    self._position_state[position_key] = position

                signals[symbol][timeframe] = pd.Series(signals_list, index=df.index)

        return signals


def create_strategy(symbols=None, timeframes=None, **params):
    if symbols is None or len(symbols) == 0:
        symbols = ['BTCUSDT']
    if timeframes is None or len(timeframes) == 0:
        timeframes = ['1m']

    config = {'min_periods': params.get('min_periods', 2)}
    return FastTestScalperStrategy(symbols, timeframes, config)
