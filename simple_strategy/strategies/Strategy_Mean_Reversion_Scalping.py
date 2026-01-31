"""
Strategy: Mean Reversion Scalping (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Use RSI + Bollinger Bands to detect overbought/oversold.
- OPEN_SHORT when RSI > overbought AND price > upper band (with volume confirmation).
- OPEN_LONG when RSI < oversold AND price < lower band (with volume confirmation).
- CLOSE_SHORT when RSI drops below exit level OR price returns to middle band.
- CLOSE_LONG when RSI rises above exit level OR price returns to middle band.
- HOLD otherwise.
"""

import os
import sys
import logging
from typing import Dict, List, Any

import pandas as pd
import numpy as np

from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.strategies.indicators_library import rsi, bollinger_bands, atr, volume_sma

# Add parent directories to path for proper imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

STRATEGY_PARAMETERS = {
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 7,
        'max': 21,
        'description': 'RSI period for mean reversion signals',
        'gui_hint': 'Standard values: 14, 10 for faster signals'
    },
    'rsi_overbought': {
        'type': 'int',
        'default': 70,
        'min': 65,
        'max': 80,
        'description': 'RSI overbought level for sell signals',
        'gui_hint': 'Higher = more extreme overbought, fewer signals'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 30,
        'min': 20,
        'max': 35,
        'description': 'RSI oversold level for buy signals',
        'gui_hint': 'Lower = more extreme oversold, fewer signals'
    },
    'bb_period': {
        'type': 'int',
        'default': 20,
        'min': 15,
        'max': 30,
        'description': 'Bollinger Bands period for mean reversion',
        'gui_hint': 'Standard values: 20, 15 for faster signals'
    },
    'bb_std_dev': {
        'type': 'float',
        'default': 2.0,
        'min': 1.8,
        'max': 2.5,
        'description': 'Bollinger Bands standard deviation',
        'gui_hint': 'Higher = wider bands, more extreme signals'
    },
    'atr_period': {
        'type': 'int',
        'default': 14,
        'min': 10,
        'max': 20,
        'description': 'ATR period for volatility measurement',
        'gui_hint': 'Standard values: 14, 10 for faster reaction'
    },
    'atr_stop_loss': {
        'type': 'float',
        'default': 0.5,
        'min': 0.3,
        'max': 1.0,
        'description': 'ATR multiplier for stop loss (tight for scalping)',
        'gui_hint': 'Lower = tighter stops, more conservative'
    },
    'atr_take_profit': {
        'type': 'float',
        'default': 1.0,
        'min': 0.5,
        'max': 2.0,
        'description': 'ATR multiplier for take profit (quick exits)',
        'gui_hint': 'Higher = larger profit targets'
    },
    'volume_sma_period': {
        'type': 'int',
        'default': 20,
        'min': 10,
        'max': 50,
        'description': 'Volume SMA period for confirmation',
        'gui_hint': 'Higher = smoother volume trend'
    },
    'rsi_exit_level': {
        'type': 'int',
        'default': 50,
        'min': 45,
        'max': 55,
        'description': 'RSI level for exit (mean reversion target)',
        'gui_hint': 'Standard is 50 (middle of RSI range)'
    },
    'risk_per_trade': {
        'type': 'float',
        'default': 0.5,
        'min': 0.1,
        'max': 1.0,
        'description': 'Risk per trade as % of account balance',
        'gui_hint': 'Lower = more conservative. Recommended: 0.5%'
    }
}


class MeanReversionScalpingStrategy(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Mean_Reversion_Scalping",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )

        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_exit_level = config.get('rsi_exit_level', 50)
        self.bb_period = config.get('bb_period', 20)
        self.bb_std_dev = config.get('bb_std_dev', 2.0)
        self.atr_period = config.get('atr_period', 14)
        self.atr_stop_loss = config.get('atr_stop_loss', 0.5)
        self.atr_take_profit = config.get('atr_take_profit', 1.0)
        self.volume_sma_period = config.get('volume_sma_period', 20)
        self.risk_per_trade = config.get('risk_per_trade', 0.5) / 100.0

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
        min_periods = max(self.rsi_period, self.bb_period, self.atr_period, self.volume_sma_period) + 1

        for symbol in data:
            signals[symbol] = {}
            for timeframe, df in data[symbol].items():
                if df is None or len(df) < min_periods:
                    signals[symbol][timeframe] = 'HOLD'
                    continue

                close = df['close']
                volume = df['volume']

                rsi_series = rsi(close, period=self.rsi_period)
                bb_upper, bb_middle, bb_lower = bollinger_bands(close, period=self.bb_period, std_dev=self.bb_std_dev)
                volume_sma_series = volume_sma(volume, period=self.volume_sma_period)

                current_close = close.iloc[-1]
                current_rsi = rsi_series.iloc[-1]
                current_volume = volume.iloc[-1]
                current_volume_sma = volume_sma_series.iloc[-1]

                volume_confirmed = current_volume > current_volume_sma * 0.8

                position_key = (symbol, timeframe)
                position = self._position_state.get(position_key)
                is_short = position.get('is_short') if position else None

                if current_rsi > self.rsi_overbought and current_close > bb_upper.iloc[-1] and volume_confirmed:
                    raw_signal = 'OPEN_SHORT'
                elif current_rsi < self.rsi_oversold and current_close < bb_lower.iloc[-1] and volume_confirmed:
                    raw_signal = 'OPEN_LONG'
                elif is_short and (current_rsi < self.rsi_exit_level or current_close <= bb_middle.iloc[-1]):
                    raw_signal = 'CLOSE_SHORT'
                elif is_short is False and (current_rsi > self.rsi_exit_level or current_close >= bb_middle.iloc[-1]):
                    raw_signal = 'CLOSE_LONG'
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
        'rsi_period': params.get('rsi_period', 14),
        'rsi_overbought': params.get('rsi_overbought', 70),
        'rsi_oversold': params.get('rsi_oversold', 30),
        'bb_period': params.get('bb_period', 20),
        'bb_std_dev': params.get('bb_std_dev', 2.0),
        'atr_period': params.get('atr_period', 14),
        'atr_stop_loss': params.get('atr_stop_loss', 0.5),
        'atr_take_profit': params.get('atr_take_profit', 1.0),
        'volume_sma_period': params.get('volume_sma_period', 20),
        'rsi_exit_level': params.get('rsi_exit_level', 50),
        'risk_per_trade': params.get('risk_per_trade', 0.5)
    }
    return MeanReversionScalpingStrategy(symbols, timeframes, config)
