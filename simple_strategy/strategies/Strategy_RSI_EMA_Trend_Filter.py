"""
Strategy: RSI EMA Trend Filter (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Determine trend using EMA fast vs EMA slow.
- If uptrend (EMA fast > EMA slow):
  - OPEN_LONG when RSI crosses below oversold.
  - CLOSE_LONG when RSI crosses above overbought.
- If downtrend (EMA fast < EMA slow):
  - OPEN_SHORT when RSI crosses above overbought.
  - CLOSE_SHORT when RSI crosses below oversold.
- HOLD otherwise.
"""

import os
import sys
import logging
from typing import Dict, List, Any

import pandas as pd
import numpy as np

from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.strategies.indicators_library import rsi, ema
from simple_strategy.strategies.signals_library import oversold_cross, overbought_cross

# Add parent directories for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STRATEGY_PARAMETERS = {
    'rsi_period': {'type': 'int', 'default': 14, 'min': 7, 'max': 21, 'description': 'RSI period'},
    'rsi_overbought': {'type': 'int', 'default': 75, 'min': 70, 'max': 80, 'description': 'RSI overbought level'},
    'rsi_oversold': {'type': 'int', 'default': 25, 'min': 20, 'max': 30, 'description': 'RSI oversold level'},
    'ema_fast_period': {'type': 'int', 'default': 20, 'min': 10, 'max': 50, 'description': 'Fast EMA period for trend'},
    'ema_slow_period': {'type': 'int', 'default': 50, 'min': 20, 'max': 100, 'description': 'Slow EMA period for trend'}
}


class RSIEMATrendFilterStrategy(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__("RSI_EMA_Trend_Filter", symbols, timeframes, config)

        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 75)
        self.rsi_oversold = config.get('rsi_oversold', 25)
        self.ema_fast_period = config.get('ema_fast_period', 20)
        self.ema_slow_period = config.get('ema_slow_period', 50)

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
        min_periods = max(self.rsi_period, self.ema_fast_period, self.ema_slow_period) + 1

        for symbol in data:
            signals[symbol] = {}
            for timeframe, df in data[symbol].items():
                if df is None or len(df) < min_periods:
                    signals[symbol][timeframe] = 'HOLD'
                    continue

                df = df.copy()
                df['rsi'] = rsi(df['close'], period=self.rsi_period)
                df['ema_fast'] = ema(df['close'], period=self.ema_fast_period)
                df['ema_slow'] = ema(df['close'], period=self.ema_slow_period)

                uptrend = df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1]
                downtrend = df['ema_fast'].iloc[-1] < df['ema_slow'].iloc[-1]

                if uptrend:
                    if oversold_cross(df['rsi'], self.rsi_oversold).iloc[-1]:
                        raw_signal = 'OPEN_LONG'
                    elif overbought_cross(df['rsi'], self.rsi_overbought).iloc[-1]:
                        raw_signal = 'CLOSE_LONG'
                    else:
                        raw_signal = 'HOLD'
                elif downtrend:
                    if overbought_cross(df['rsi'], self.rsi_overbought).iloc[-1]:
                        raw_signal = 'OPEN_SHORT'
                    elif oversold_cross(df['rsi'], self.rsi_oversold).iloc[-1]:
                        raw_signal = 'CLOSE_SHORT'
                    else:
                        raw_signal = 'HOLD'
                else:
                    raw_signal = 'HOLD'

                position_key = (symbol, timeframe)
                signals[symbol][timeframe] = self._apply_position_rules(position_key, raw_signal)

        return signals

    def generate_signals_vectorized(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.Series]]:
        signals: Dict[str, Dict[str, pd.Series]] = {}
        min_periods = max(self.rsi_period, self.ema_fast_period, self.ema_slow_period) + 1

        for symbol in data:
            signals[symbol] = {}
            for timeframe, df in data[symbol].items():
                if df is None or len(df) < min_periods:
                    signals[symbol][timeframe] = pd.Series(['HOLD'] * len(df), index=df.index)
                    continue

                rsi_series = rsi(df['close'], period=self.rsi_period)
                ema_fast = ema(df['close'], period=self.ema_fast_period)
                ema_slow = ema(df['close'], period=self.ema_slow_period)

                uptrend = ema_fast > ema_slow
                downtrend = ema_fast < ema_slow

                os_cross = oversold_cross(rsi_series, self.rsi_oversold)
                ob_cross = overbought_cross(rsi_series, self.rsi_overbought)

                raw = np.where(uptrend & os_cross, 'OPEN_LONG',
                      np.where(uptrend & ob_cross, 'CLOSE_LONG',
                      np.where(downtrend & ob_cross, 'OPEN_SHORT',
                      np.where(downtrend & os_cross, 'CLOSE_SHORT', 'HOLD'))))

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
                    elif raw_signal == 'OPEN_SHORT':
                        if position is None:
                            position = {'is_short': True}
                            signals_list.append('OPEN_SHORT')
                        else:
                            signals_list.append('HOLD')
                    elif raw_signal == 'CLOSE_LONG':
                        if position is not None and not position.get('is_short', False):
                            position = None
                            signals_list.append('CLOSE_LONG')
                        else:
                            signals_list.append('HOLD')
                    elif raw_signal == 'CLOSE_SHORT':
                        if position is not None and position.get('is_short', False):
                            position = None
                            signals_list.append('CLOSE_SHORT')
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
    if symbols is None:
        symbols = ['BTCUSDT']
    if timeframes is None:
        timeframes = ['5m']
    return RSIEMATrendFilterStrategy(symbols, timeframes, params)
