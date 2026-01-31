"""
Strategy 2: Mean Reversion with Trend Filter (Updated for OPEN/CLOSE schema)

Trade logic (simple):
- Calculate RSI plus a fast/slow EMA trend filter.
- If EMA fast > EMA slow (uptrend):
  - OPEN_LONG when RSI crosses below the oversold level.
  - CLOSE_LONG when RSI crosses above the overbought level.
- If EMA fast < EMA slow (downtrend) and bidirectional=True:
  - OPEN_SHORT when RSI crosses above the overbought level.
  - CLOSE_SHORT when RSI crosses below the oversold level.
- HOLD otherwise.
"""
import os
import sys
import logging
from typing import Dict, List, Any

import pandas as pd

from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.strategies.indicators_library import rsi, ema

# Add parent directories to path for proper imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

STRATEGY_PARAMETERS = {
    'rsi_period': {
        'type': 'int',
        'default': 14,
        'min': 1,
        'max': 50,
        'description': 'RSI calculation period',
        'gui_hint': 'Standard values: 14, 21. Lower = more sensitive'
    },
    'rsi_overbought': {
        'type': 'int',
        'default': 70,
        'min': 50,
        'max': 90,
        'description': 'RSI overbought level (signal on crossover)',
        'gui_hint': 'Higher values = more conservative signals'
    },
    'rsi_oversold': {
        'type': 'int',
        'default': 30,
        'min': 10,
        'max': 50,
        'description': 'RSI oversold level (signal on crossover)',
        'gui_hint': 'Lower values = more conservative signals'
    },
    'trend_fast_ema': {
        'type': 'int',
        'default': 20,
        'min': 5,
        'max': 50,
        'description': 'Fast EMA period for trend detection',
        'gui_hint': 'Lower values = more responsive trend signals'
    },
    'trend_slow_ema': {
        'type': 'int',
        'default': 50,
        'min': 20,
        'max': 200,
        'description': 'Slow EMA period for trend detection',
        'gui_hint': 'Higher values = smoother trend signals'
    },
    'bidirectional': {
        'type': 'bool',
        'default': True,
        'description': 'Enable bidirectional trading (long and short)',
        'gui_hint': 'When enabled, trades short positions in downtrends'
    }
}


class MeanReversionStrategy(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Mean_Reversion_Bidirectional",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.trend_fast_ema = config.get('trend_fast_ema', 20)
        self.trend_slow_ema = config.get('trend_slow_ema', 50)
        self.bidirectional = config.get('bidirectional', True)
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
        min_periods = max(self.rsi_period, self.trend_slow_ema) + 1

        for symbol in data:
            signals[symbol] = {}
            for timeframe, df in data[symbol].items():
                if df is None or len(df) < min_periods:
                    signals[symbol][timeframe] = 'HOLD'
                    continue

                close_series = df['close']
                rsi_series = rsi(close_series, period=self.rsi_period)
                ema_fast = ema(close_series, period=self.trend_fast_ema)
                ema_slow = ema(close_series, period=self.trend_slow_ema)

                prev_rsi = rsi_series.iloc[-2]
                last_rsi = rsi_series.iloc[-1]
                last_ema_fast = ema_fast.iloc[-1]
                last_ema_slow = ema_slow.iloc[-1]

                uptrend = last_ema_fast > last_ema_slow

                cross_below_oversold = (last_rsi < self.rsi_oversold) and (prev_rsi >= self.rsi_oversold)
                cross_above_overbought = (last_rsi > self.rsi_overbought) and (prev_rsi <= self.rsi_overbought)

                position_key = (symbol, timeframe)
                position = self._position_state.get(position_key)
                is_short = position.get('is_short') if position else None

                if uptrend:
                    if cross_below_oversold:
                        raw_signal = 'OPEN_LONG'
                    elif cross_above_overbought:
                        raw_signal = 'CLOSE_LONG'
                    else:
                        raw_signal = 'HOLD'
                else:
                    if not self.bidirectional:
                        raw_signal = 'HOLD'
                    elif cross_above_overbought:
                        raw_signal = 'OPEN_SHORT'
                    elif cross_below_oversold:
                        raw_signal = 'CLOSE_SHORT'
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
        'trend_fast_ema': params.get('trend_fast_ema', 20),
        'trend_slow_ema': params.get('trend_slow_ema', 50),
        'bidirectional': params.get('bidirectional', True)
    }
    return MeanReversionStrategy(symbols, timeframes, config)


def simple_test():
    try:
        strategy = create_strategy(
            symbols=['BTCUSDT'],
            timeframes=['1m'],
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30,
            trend_fast_ema=20,
            trend_slow_ema=50,
            bidirectional=True
        )
        print(f"✅ Mean Reversion strategy created successfully: {strategy.name}")
        print(f"  - Symbols: {strategy.symbols}")
        print(f"  - Timeframes: {strategy.timeframes}")
        return True
    except Exception as e:
        print(f"❌ Error testing Mean Reversion strategy: {e}")
        return False


if __name__ == "__main__":
    simple_test()
