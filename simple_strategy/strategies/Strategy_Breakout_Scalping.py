"""
Strategy: Breakout Scalping (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Find consolidation using low ATR and tight range.
- Confirm breakout with strong volume.
- If price breaks ABOVE range and above EMA:
    - OPEN_LONG
    - CLOSE_LONG when stop loss or take profit is hit
- If price breaks BELOW range and below EMA:
    - OPEN_SHORT
    - CLOSE_SHORT when stop loss or take profit is hit
- HOLD otherwise.
"""

import os
import sys
import logging
from typing import Dict, List, Any

import pandas as pd

from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.strategies.indicators_library import atr, ema, volume_sma, highest, lowest

# Add parent directories to path for proper imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

STRATEGY_PARAMETERS = {
    'atr_period': {
        'type': 'int',
        'default': 14,
        'min': 10,
        'max': 20,
        'description': 'ATR period for volatility measurement',
        'gui_hint': 'Standard values: 14, 10 for faster reaction'
    },
    'atr_threshold': {
        'type': 'float',
        'default': 0.15,
        'min': 0.05,
        'max': 0.3,
        'description': 'ATR threshold for consolidation (as % of price)',
        'gui_hint': 'Lower = tighter consolidation, fewer breakouts'
    },
    'range_period': {
        'type': 'int',
        'default': 20,
        'min': 10,
        'max': 50,
        'description': 'Lookback period for range detection',
        'gui_hint': 'Higher = longer consolidation periods'
    },
    'ema_period': {
        'type': 'int',
        'default': 20,
        'min': 10,
        'max': 50,
        'description': 'EMA period for trend filter',
        'gui_hint': 'Higher = smoother trend filter'
    },
    'volume_sma_period': {
        'type': 'int',
        'default': 20,
        'min': 10,
        'max': 50,
        'description': 'Volume SMA period for confirmation',
        'gui_hint': 'Higher = smoother volume trend'
    },
    'volume_multiplier': {
        'type': 'float',
        'default': 2.0,
        'min': 1.5,
        'max': 3.0,
        'description': 'Volume multiplier for breakout confirmation',
        'gui_hint': 'Higher = stronger volume confirmation'
    },
    'atr_stop_loss': {
        'type': 'float',
        'default': 0.8,
        'min': 0.5,
        'max': 1.5,
        'description': 'ATR multiplier for stop loss',
        'gui_hint': 'Lower = tighter stops, more conservative'
    },
    'atr_take_profit': {
        'type': 'float',
        'default': 1.5,
        'min': 1.0,
        'max': 3.0,
        'description': 'ATR multiplier for take profit',
        'gui_hint': 'Higher = larger profit targets'
    },
    'breakout_threshold': {
        'type': 'float',
        'default': 0.1,
        'min': 0.05,
        'max': 0.2,
        'description': 'Breakout threshold as % of range',
        'gui_hint': 'Lower = easier breakouts, more signals'
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


class BreakoutScalpingStrategy(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Breakout_Scalping",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )

        self.atr_period = config.get('atr_period', 14)
        self.atr_threshold = config.get('atr_threshold', 0.15)
        self.range_period = config.get('range_period', 20)
        self.ema_period = config.get('ema_period', 20)
        self.volume_sma_period = config.get('volume_sma_period', 20)
        self.volume_multiplier = config.get('volume_multiplier', 2.0)
        self.atr_stop_loss = config.get('atr_stop_loss', 0.8)
        self.atr_take_profit = config.get('atr_take_profit', 1.5)
        self.breakout_threshold = config.get('breakout_threshold', 0.1)
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

        min_periods = max(self.atr_period, self.range_period, self.ema_period, self.volume_sma_period) + 1

        for symbol in data:
            signals[symbol] = {}
            for timeframe, df in data[symbol].items():
                if df is None or len(df) < min_periods:
                    signals[symbol][timeframe] = 'HOLD'
                    continue

                close_series = df['close']
                high_series = df['high']
                low_series = df['low']
                volume_series = df['volume']

                atr_series = atr(high_series, low_series, close_series, period=self.atr_period)
                ema_series = ema(close_series, period=self.ema_period)
                highest_series = highest(high_series, period=self.range_period)
                lowest_series = lowest(low_series, period=self.range_period)
                volume_sma_series = volume_sma(volume_series, period=self.volume_sma_period)

                current_close = close_series.iloc[-1]
                current_atr = atr_series.iloc[-1]
                current_ema = ema_series.iloc[-1]
                highest_high = highest_series.iloc[-1]
                lowest_low = lowest_series.iloc[-1]
                current_volume = volume_series.iloc[-1]
                current_volume_sma = volume_sma_series.iloc[-1]

                atr_percent = (current_atr / current_close) * 100 if current_close else 0
                in_consolidation = atr_percent < self.atr_threshold

                range_size = highest_high - lowest_low
                breakout_high = highest_high + (range_size * self.breakout_threshold / 100)
                breakout_low = lowest_low - (range_size * self.breakout_threshold / 100)

                volume_confirmed = current_volume > current_volume_sma * self.volume_multiplier

                position_key = (symbol, timeframe)
                position = self._position_state.get(position_key)

                # Exit logic if already in a position
                if position:
                    if not position.get('is_short', False):
                        if current_close >= position['take_profit'] or current_close <= position['stop_loss']:
                            signals[symbol][timeframe] = self._apply_position_rules(position_key, 'CLOSE_LONG')
                            continue
                    else:
                        if current_close <= position['take_profit'] or current_close >= position['stop_loss']:
                            signals[symbol][timeframe] = self._apply_position_rules(position_key, 'CLOSE_SHORT')
                            continue

                # Entry logic
                if in_consolidation and volume_confirmed:
                    if current_close > breakout_high and current_close > current_ema:
                        stop_loss = current_close - (current_atr * self.atr_stop_loss)
                        take_profit = current_close + (current_atr * self.atr_take_profit)
                        self._position_state[position_key] = {
                            'is_short': False,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        }
                        signals[symbol][timeframe] = 'OPEN_LONG'
                        continue

                    if current_close < breakout_low and current_close < current_ema:
                        stop_loss = current_close + (current_atr * self.atr_stop_loss)
                        take_profit = current_close - (current_atr * self.atr_take_profit)
                        self._position_state[position_key] = {
                            'is_short': True,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        }
                        signals[symbol][timeframe] = 'OPEN_SHORT'
                        continue

                signals[symbol][timeframe] = 'HOLD'

        return signals


def create_strategy(symbols=None, timeframes=None, **params):
    if symbols is None or len(symbols) == 0:
        symbols = ['BTCUSDT']
    if timeframes is None or len(timeframes) == 0:
        timeframes = ['1m']

    config = {
        'atr_period': params.get('atr_period', 14),
        'atr_threshold': params.get('atr_threshold', 0.15),
        'range_period': params.get('range_period', 20),
        'ema_period': params.get('ema_period', 20),
        'volume_sma_period': params.get('volume_sma_period', 20),
        'volume_multiplier': params.get('volume_multiplier', 2.0),
        'atr_stop_loss': params.get('atr_stop_loss', 0.8),
        'atr_take_profit': params.get('atr_take_profit', 1.5),
        'breakout_threshold': params.get('breakout_threshold', 0.1),
        'risk_per_trade': params.get('risk_per_trade', 0.5)
    }
    return BreakoutScalpingStrategy(symbols, timeframes, config)
