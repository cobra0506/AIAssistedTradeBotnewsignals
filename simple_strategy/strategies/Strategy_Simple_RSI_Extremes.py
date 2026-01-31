"""
Strategy: Simple RSI Extremes (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Use RSI extremes for mean-reversion signals.
- Use a trend SMA filter to trade with the trend.
- Use volume SMA confirmation to avoid weak signals.
- If oversold + uptrend + volume confirmed -> OPEN_LONG.
- If overbought + downtrend + volume confirmed -> OPEN_SHORT.
- If in a position and the opposite extreme appears -> CLOSE.
- HOLD otherwise.
"""

import os
import sys
import logging
from typing import Dict, List, Any

import pandas as pd

from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.strategies.indicators_library import rsi, sma, volume_sma

# Add parent directories to path for proper imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

STRATEGY_PARAMETERS = {
    'rsi_period': {
        'type': 'int',
        'default': 8,
        'min': 7,
        'max': 21,
        'description': 'RSI period for mean reversion signals',
        'gui_hint': 'Standard values: 14, 10 for faster signals'
    },
    'rsi_overbought': {
        'type': 'int',
        'default': 77,
        'min': 70,
        'max': 80,
        'description': 'RSI overbought level for sell signals',
        'gui_hint': 'Higher = more extreme overbought, fewer but higher quality signals'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 26,
        'min': 20,
        'max': 30,
        'description': 'RSI oversold level for buy signals',
        'gui_hint': 'Lower = more extreme oversold, fewer but higher quality signals'
    },
    'trend_sma_period': {
        'type': 'int',
        'default': 233,
        'min': 50,
        'max': 300,
        'description': 'SMA period for trend filter',
        'gui_hint': 'Higher = longer term trend. 200 is standard for daily charts'
    },
    'volume_sma_period': {
        'type': 'int',
        'default': 34,
        'min': 10,
        'max': 50,
        'description': 'Volume SMA period for confirmation',
        'gui_hint': 'Higher = smoother volume trend'
    },
    'volume_multiplier': {
        'type': 'float',
        'default': 1.79,
        'min': 1.0,
        'max': 2.0,
        'description': 'Volume multiplier for signal confirmation',
        'gui_hint': 'Higher = stronger volume confirmation required'
    }
}


class ImprovedSimpleRSIExtremesStrategy(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Improved_Simple_RSI_Extremes",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )

        self.rsi_period = config.get('rsi_period', 8)
        self.rsi_overbought = config.get('rsi_overbought', 77)
        self.rsi_oversold = config.get('rsi_oversold', 26)
        self.trend_sma_period = config.get('trend_sma_period', 233)
        self.volume_sma_period = config.get('volume_sma_period', 34)
        self.volume_multiplier = config.get('volume_multiplier', 1.79)

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
        min_periods = max(self.rsi_period, self.trend_sma_period, self.volume_sma_period) + 1

        for symbol in data:
            signals[symbol] = {}
            for timeframe, df in data[symbol].items():
                if df is None or len(df) < min_periods:
                    signals[symbol][timeframe] = 'HOLD'
                    continue

                close = df['close']
                volume = df['volume']

                rsi_series = rsi(close, period=self.rsi_period)
                trend_sma = sma(close, period=self.trend_sma_period)
                volume_sma_series = volume_sma(volume, period=self.volume_sma_period)

                current_close = close.iloc[-1]
                current_rsi = rsi_series.iloc[-1]
                current_volume = volume.iloc[-1]
                current_trend = trend_sma.iloc[-1]
                current_volume_sma = volume_sma_series.iloc[-1]

                volume_confirmed = current_volume > current_volume_sma * self.volume_multiplier
                bullish_trend = current_close > current_trend
                bearish_trend = current_close < current_trend

                position_key = (symbol, timeframe)
                position = self._position_state.get(position_key)

                if current_rsi <= self.rsi_oversold and bullish_trend and volume_confirmed:
                    raw_signal = 'OPEN_LONG'
                elif current_rsi >= self.rsi_overbought and bearish_trend and volume_confirmed:
                    raw_signal = 'OPEN_SHORT'
                elif position and not position.get('is_short', False) and current_rsi >= self.rsi_overbought:
                    raw_signal = 'CLOSE_LONG'
                elif position and position.get('is_short', False) and current_rsi <= self.rsi_oversold:
                    raw_signal = 'CLOSE_SHORT'
                else:
                    raw_signal = 'HOLD'

                signals[symbol][timeframe] = self._apply_position_rules(position_key, raw_signal)

        return signals


def create_strategy(symbols=None, timeframes=None, **params):
    if symbols is None or len(symbols) == 0:
        symbols = ['BTCUSDT']
    if timeframes is None or len(timeframes) == 0:
        timeframes = ['5m']

    config = {
        'rsi_period': params.get('rsi_period', 8),
        'rsi_overbought': params.get('rsi_overbought', 77),
        'rsi_oversold': params.get('rsi_oversold', 26),
        'trend_sma_period': params.get('trend_sma_period', 233),
        'volume_sma_period': params.get('volume_sma_period', 34),
        'volume_multiplier': params.get('volume_multiplier', 1.79)
    }
    return ImprovedSimpleRSIExtremesStrategy(symbols, timeframes, config)
