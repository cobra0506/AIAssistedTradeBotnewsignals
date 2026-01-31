"""
Strategy: Improved Multi-Timeframe Scalping (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Use a higher timeframe EMA trend filter.
- On the entry timeframe (1m by default):
  - In uptrend: OPEN_LONG when price is above EMAs, RSI not overbought, and volume confirms.
  - In downtrend: OPEN_SHORT when price is below EMAs, RSI not oversold, and volume confirms.
- Exit using ATR-based stop loss or take profit:
  - CLOSE_LONG when stop loss or take profit hit.
  - CLOSE_SHORT when stop loss or take profit hit.
- HOLD otherwise.
"""

import os
import sys
import logging
from typing import Dict, List, Any

import pandas as pd

from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.strategies.indicators_library import ema, rsi, atr, volume_sma

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
        'gui_hint': 'Lower values = more sensitive entries. Recommended: 8-12'
    },
    'slow_ema_period': {
        'type': 'int',
        'default': 21,
        'min': 15,
        'max': 50,
        'description': 'Slow EMA period for trend direction',
        'gui_hint': 'Higher values = smoother trend. Recommended: 20-25'
    },
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 7,
        'max': 21,
        'description': 'RSI period for momentum confirmation',
        'gui_hint': 'Standard values: 14, 10 for faster signals'
    },
    'rsi_overbought': {
        'type': 'int',
        'default': 70,
        'min': 60,
        'max': 80,
        'description': 'RSI overbought level for sell signals',
        'gui_hint': 'Higher = more conservative sells'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 30,
        'min': 20,
        'max': 40,
        'description': 'RSI oversold level for buy signals',
        'gui_hint': 'Lower = more conservative buys'
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
        'default': 1.0,
        'min': 0.5,
        'max': 2.0,
        'description': 'ATR multiplier for stop loss',
        'gui_hint': 'Lower = tighter stops, more conservative'
    },
    'atr_take_profit': {
        'type': 'float',
        'default': 2.0,
        'min': 1.0,
        'max': 4.0,
        'description': 'ATR multiplier for take profit',
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
    'trend_timeframe': {
        'type': 'str',
        'default': '5m',
        'options': ['5m', '15m', '30m'],
        'description': 'Higher timeframe for trend confirmation',
        'gui_hint': 'Use 5m for scalping, 15m for swing trades'
    },
    'min_atr_threshold': {
        'type': 'float',
        'default': 0.1,
        'min': 0.05,
        'max': 0.5,
        'description': 'Minimum ATR threshold for trading (as % of price)',
        'gui_hint': 'Filter out low volatility periods. 0.1 = 0.1%'
    },
    'risk_per_trade': {
        'type': 'float',
        'default': 0.5,
        'min': 0.1,
        'max': 2.0,
        'description': 'Risk per trade as % of account balance',
        'gui_hint': 'Lower = more conservative. Recommended: 0.5-1.0%'
    }
}


class ImprovedMultiTimeframeScalpingStrategy(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Improved_Multi_Timeframe_Scalping",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )

        self.fast_ema_period = config.get('fast_ema_period', 9)
        self.slow_ema_period = config.get('slow_ema_period', 21)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.atr_period = config.get('atr_period', 14)
        self.atr_stop_loss = config.get('atr_stop_loss', 1.0)
        self.atr_take_profit = config.get('atr_take_profit', 2.0)
        self.volume_sma_period = config.get('volume_sma_period', 20)
        self.trend_timeframe = config.get('trend_timeframe', '5m')
        self.min_atr_threshold = config.get('min_atr_threshold', 0.1)
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
        min_periods = max(self.slow_ema_period, self.rsi_period, self.atr_period, self.volume_sma_period) + 1

        for symbol in data:
            signals[symbol] = {}
            for timeframe, df in data[symbol].items():
                if df is None or len(df) < min_periods:
                    signals[symbol][timeframe] = 'HOLD'
                    continue

                # Indicators on the current timeframe
                close_series = df['close']
                volume_series = df['volume']

                ema_fast = ema(close_series, period=self.fast_ema_period)
                ema_slow = ema(close_series, period=self.slow_ema_period)
                rsi_series = rsi(close_series, period=self.rsi_period)
                atr_series = atr(df['high'], df['low'], close_series, period=self.atr_period)
                volume_sma_series = volume_sma(volume_series, period=self.volume_sma_period)

                current_close = close_series.iloc[-1]
                current_rsi = rsi_series.iloc[-1]
                current_atr = atr_series.iloc[-1]
                current_volume = volume_series.iloc[-1]
                current_volume_sma = volume_sma_series.iloc[-1]

                atr_percent = (current_atr / current_close) * 100 if current_close else 0
                if atr_percent < self.min_atr_threshold:
                    signals[symbol][timeframe] = 'HOLD'
                    continue

                trend_signal = self._get_trend_signal(data, symbol)

                # Only generate entries on entry timeframe
                entry_timeframe = '1m' if '1m' in self.timeframes else self.timeframes[0]
                if timeframe != entry_timeframe:
                    signals[symbol][timeframe] = 'HOLD'
                    continue

                position_key = (symbol, timeframe)
                position = self._position_state.get(position_key)

                # Exit logic
                if position:
                    if not position.get('is_short', False):
                        if current_close >= position['take_profit'] or current_close <= position['stop_loss']:
                            signals[symbol][timeframe] = self._apply_position_rules(position_key, 'CLOSE_LONG')
                            continue
                    else:
                        if current_close <= position['take_profit'] or current_close >= position['stop_loss']:
                            signals[symbol][timeframe] = self._apply_position_rules(position_key, 'CLOSE_SHORT')
                            continue

                volume_confirmed = current_volume > current_volume_sma * 1.1

                if trend_signal == 'BULLISH':
                    price_above_emas = current_close > ema_fast.iloc[-1] and current_close > ema_slow.iloc[-1]
                    rsi_ok = current_rsi < self.rsi_overbought
                    if price_above_emas and rsi_ok and volume_confirmed:
                        stop_loss = current_close - (current_atr * self.atr_stop_loss)
                        take_profit = current_close + (current_atr * self.atr_take_profit)
                        self._position_state[position_key] = {
                            'is_short': False,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        }
                        signals[symbol][timeframe] = 'OPEN_LONG'
                        continue

                if trend_signal == 'BEARISH':
                    price_below_emas = current_close < ema_fast.iloc[-1] and current_close < ema_slow.iloc[-1]
                    rsi_ok = current_rsi > self.rsi_oversold
                    if price_below_emas and rsi_ok and volume_confirmed:
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

    def _get_trend_signal(self, data: Dict[str, Dict[str, pd.DataFrame]], symbol: str) -> str:
        if self.trend_timeframe not in data.get(symbol, {}):
            return 'NEUTRAL'

        trend_df = data[symbol][self.trend_timeframe]
        if trend_df is None or len(trend_df) < self.slow_ema_period + 1:
            return 'NEUTRAL'

        close_series = trend_df['close']
        ema_fast = ema(close_series, period=self.fast_ema_period).iloc[-1]
        ema_slow = ema(close_series, period=self.slow_ema_period).iloc[-1]

        if ema_fast > ema_slow:
            return 'BULLISH'
        if ema_fast < ema_slow:
            return 'BEARISH'
        return 'NEUTRAL'


def create_strategy(symbols=None, timeframes=None, **params):
    if symbols is None or len(symbols) == 0:
        symbols = ['BTCUSDT']
    if timeframes is None or len(timeframes) == 0:
        timeframes = ['1m']

    trend_timeframe = params.get('trend_timeframe', '5m')
    if trend_timeframe not in timeframes:
        timeframes.append(trend_timeframe)

    config = {
        'fast_ema_period': params.get('fast_ema_period', 9),
        'slow_ema_period': params.get('slow_ema_period', 21),
        'rsi_period': params.get('rsi_period', 14),
        'rsi_overbought': params.get('rsi_overbought', 70),
        'rsi_oversold': params.get('rsi_oversold', 30),
        'atr_period': params.get('atr_period', 14),
        'atr_stop_loss': params.get('atr_stop_loss', 1.0),
        'atr_take_profit': params.get('atr_take_profit', 2.0),
        'volume_sma_period': params.get('volume_sma_period', 20),
        'trend_timeframe': trend_timeframe,
        'min_atr_threshold': params.get('min_atr_threshold', 0.1),
        'risk_per_trade': params.get('risk_per_trade', 0.5)
    }
    return ImprovedMultiTimeframeScalpingStrategy(symbols, timeframes, config)
