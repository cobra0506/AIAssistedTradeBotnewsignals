"""
Strategy: Scalping Multi-Indicator (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Build three indicator signals: RSI (overbought/oversold), EMA crossover, MACD crossover.
- Combine them using weighted voting.
- If the weighted signal is strongly bullish, OPEN_LONG.
- If the weighted signal is strongly bearish, OPEN_SHORT.
- Use ATR-based stop loss and take profit to CLOSE positions.
- Also CLOSE if the weighted signal flips against the open position.
- HOLD otherwise.
"""

import os
import sys
import logging
from typing import Dict, List, Any

import pandas as pd

from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.strategies.indicators_library import rsi, ema, macd, atr

# Add parent directories to path for proper imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

STRATEGY_PARAMETERS = {
    'rsi_period': {
        'type': 'int',
        'default': 9,
        'min': 5,
        'max': 21,
        'description': 'RSI calculation period (shorter for scalping)',
        'gui_hint': 'Scalping: Use 5-9 for faster signals'
    },
    'rsi_overbought': {
        'type': 'int',
        'default': 75,
        'min': 65,
        'max': 85,
        'description': 'RSI overbought level (sell signal)',
        'gui_hint': 'Higher = more conservative sells (75-80 recommended)'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 25,
        'min': 15,
        'max': 35,
        'description': 'RSI oversold level (buy signal)',
        'gui_hint': 'Lower = more conservative buys (20-25 recommended)'
    },
    'ema_fast': {
        'type': 'int',
        'default': 5,
        'min': 3,
        'max': 10,
        'description': 'Fast EMA period for trend direction',
        'gui_hint': 'Scalping: Use 3-5 for quick trend changes'
    },
    'ema_slow': {
        'type': 'int',
        'default': 13,
        'min': 8,
        'max': 21,
        'description': 'Slow EMA period for trend confirmation',
        'gui_hint': 'Scalping: Use 9-13 for trend confirmation'
    },
    'macd_fast': {
        'type': 'int',
        'default': 5,
        'min': 3,
        'max': 8,
        'description': 'MACD fast EMA period',
        'gui_hint': 'Scalping: Use 3-5 for faster MACD signals'
    },
    'macd_slow': {
        'type': 'int',
        'default': 13,
        'min': 9,
        'max': 21,
        'description': 'MACD slow EMA period',
        'gui_hint': 'Scalping: Use 9-13 for MACD confirmation'
    },
    'macd_signal': {
        'type': 'int',
        'default': 6,
        'min': 3,
        'max': 9,
        'description': 'MACD signal line period',
        'gui_hint': 'Scalping: Use 3-6 for quick MACD signals'
    },
    'stop_loss_atr': {
        'type': 'float',
        'default': 1.5,
        'min': 0.5,
        'max': 3.0,
        'description': 'Stop loss as ATR multiplier',
        'gui_hint': 'Scalping: Use 0.5-1.5 for tight stops'
    },
    'take_profit_atr': {
        'type': 'float',
        'default': 2.5,
        'min': 1.0,
        'max': 5.0,
        'description': 'Take profit as ATR multiplier',
        'gui_hint': 'Scalping: Use 1.5-3.0 for quick profits'
    },
    'atr_period': {
        'type': 'int',
        'default': 14,
        'min': 7,
        'max': 21,
        'description': 'ATR calculation period',
        'gui_hint': 'Use 7-14 for current volatility'
    },
    'rsi_weight': {
        'type': 'float',
        'default': 0.33,
        'min': 0.1,
        'max': 0.6,
        'description': 'RSI signal weight (0.0-1.0)',
        'gui_hint': 'Higher = more importance to RSI signals'
    },
    'ema_weight': {
        'type': 'float',
        'default': 0.33,
        'min': 0.1,
        'max': 0.6,
        'description': 'EMA crossover signal weight (0.0-1.0)',
        'gui_hint': 'Higher = more importance to trend signals'
    },
    'macd_weight': {
        'type': 'float',
        'default': 0.34,
        'min': 0.1,
        'max': 0.6,
        'description': 'MACD signal weight (0.0-1.0)',
        'gui_hint': 'Higher = more importance to momentum signals'
    },
    'signal_threshold': {
        'type': 'float',
        'default': 0.3,
        'min': 0.1,
        'max': 0.9,
        'description': 'Weighted signal threshold for opening trades',
        'gui_hint': 'Lower = more trades, higher = stronger confirmation'
    }
}


class ScalpingMultiIndicatorStrategy(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Scalping_MultiIndicator",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )

        self.rsi_period = config.get('rsi_period', 9)
        self.rsi_overbought = config.get('rsi_overbought', 75)
        self.rsi_oversold = config.get('rsi_oversold', 25)

        self.ema_fast = config.get('ema_fast', 5)
        self.ema_slow = config.get('ema_slow', 13)

        self.macd_fast = config.get('macd_fast', 5)
        self.macd_slow = config.get('macd_slow', 13)
        self.macd_signal = config.get('macd_signal', 6)

        self.stop_loss_atr = config.get('stop_loss_atr', 1.5)
        self.take_profit_atr = config.get('take_profit_atr', 2.5)
        self.atr_period = config.get('atr_period', 14)

        self.rsi_weight = config.get('rsi_weight', 0.33)
        self.ema_weight = config.get('ema_weight', 0.33)
        self.macd_weight = config.get('macd_weight', 0.34)

        self.signal_threshold = config.get('signal_threshold', 0.3)

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
        min_periods = max(self.ema_slow, self.macd_slow, self.rsi_period, self.atr_period) + 2

        for symbol in data:
            signals[symbol] = {}
            for timeframe, df in data[symbol].items():
                if df is None or len(df) < min_periods:
                    signals[symbol][timeframe] = 'HOLD'
                    continue

                close_prices = df['close']

                rsi_values = rsi(close_prices, self.rsi_period)
                ema_fast_values = ema(close_prices, self.ema_fast)
                ema_slow_values = ema(close_prices, self.ema_slow)
                macd_line, macd_signal_line, _ = macd(
                    close_prices,
                    fast_period=self.macd_fast,
                    slow_period=self.macd_slow,
                    signal_period=self.macd_signal
                )
                atr_values = atr(df['high'], df['low'], close_prices, period=self.atr_period)

                current_rsi = rsi_values.iloc[-1]
                current_close = close_prices.iloc[-1]
                current_atr = atr_values.iloc[-1]

                rsi_signal = 0.0
                if current_rsi < self.rsi_oversold:
                    rsi_signal = 1.0
                elif current_rsi > self.rsi_overbought:
                    rsi_signal = -1.0

                ema_signal = 0.0
                if ema_fast_values.iloc[-1] > ema_slow_values.iloc[-1] and ema_fast_values.iloc[-2] <= ema_slow_values.iloc[-2]:
                    ema_signal = 1.0
                elif ema_fast_values.iloc[-1] < ema_slow_values.iloc[-1] and ema_fast_values.iloc[-2] >= ema_slow_values.iloc[-2]:
                    ema_signal = -1.0

                macd_signal_value = 0.0
                if macd_line.iloc[-1] > macd_signal_line.iloc[-1] and macd_line.iloc[-2] <= macd_signal_line.iloc[-2]:
                    macd_signal_value = 1.0
                elif macd_line.iloc[-1] < macd_signal_line.iloc[-1] and macd_line.iloc[-2] >= macd_signal_line.iloc[-2]:
                    macd_signal_value = -1.0

                weighted_signal = (
                    rsi_signal * self.rsi_weight +
                    ema_signal * self.ema_weight +
                    macd_signal_value * self.macd_weight
                )

                position_key = (symbol, timeframe)
                position = self._position_state.get(position_key)

                if position:
                    if not position.get('is_short', False):
                        if current_close <= position['stop_loss'] or current_close >= position['take_profit']:
                            signals[symbol][timeframe] = self._apply_position_rules(position_key, 'CLOSE_LONG')
                            continue
                        if weighted_signal < 0:
                            signals[symbol][timeframe] = self._apply_position_rules(position_key, 'CLOSE_LONG')
                            continue
                    else:
                        if current_close >= position['stop_loss'] or current_close <= position['take_profit']:
                            signals[symbol][timeframe] = self._apply_position_rules(position_key, 'CLOSE_SHORT')
                            continue
                        if weighted_signal > 0:
                            signals[symbol][timeframe] = self._apply_position_rules(position_key, 'CLOSE_SHORT')
                            continue

                if weighted_signal >= self.signal_threshold:
                    stop_loss = current_close - (current_atr * self.stop_loss_atr)
                    take_profit = current_close + (current_atr * self.take_profit_atr)
                    self._position_state[position_key] = {
                        'is_short': False,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                    signals[symbol][timeframe] = 'OPEN_LONG'
                    continue

                if weighted_signal <= -self.signal_threshold:
                    stop_loss = current_close + (current_atr * self.stop_loss_atr)
                    take_profit = current_close - (current_atr * self.take_profit_atr)
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
        timeframes = ['1m', '3m', '5m']

    config = {
        'rsi_period': params.get('rsi_period', 9),
        'rsi_overbought': params.get('rsi_overbought', 75),
        'rsi_oversold': params.get('rsi_oversold', 25),
        'ema_fast': params.get('ema_fast', 5),
        'ema_slow': params.get('ema_slow', 13),
        'macd_fast': params.get('macd_fast', 5),
        'macd_slow': params.get('macd_slow', 13),
        'macd_signal': params.get('macd_signal', 6),
        'stop_loss_atr': params.get('stop_loss_atr', 1.5),
        'take_profit_atr': params.get('take_profit_atr', 2.5),
        'atr_period': params.get('atr_period', 14),
        'rsi_weight': params.get('rsi_weight', 0.33),
        'ema_weight': params.get('ema_weight', 0.33),
        'macd_weight': params.get('macd_weight', 0.34),
        'signal_threshold': params.get('signal_threshold', 0.3)
    }
    return ScalpingMultiIndicatorStrategy(symbols, timeframes, config)
