"""
Strategy 1: Trend Following Strategy (Updated for OPEN/CLOSE schema)
Uses moving average crossovers to identify trend direction and trade accordingly.
Exact trade logic for Strategy_1_Trend_Following
    Indicators: Fast MA and slow MA computed on closing prices (SMA or EMA based on ma_type).
    Crossover detection:
        Cross Up: fast crosses above slow when last_fast > last_slow and prev_fast <= prev_slow.
        Cross Down: fast crosses below slow when last_fast < last_slow and prev_fast >= prev_slow.
    Signal decisions (single signal per bar):
        Cross Up:
            If currently short → CLOSE_SHORT
            If no open position → OPEN_LONG
        Cross Down:
            If currently long → CLOSE_LONG
            If no open position → OPEN_SHORT
        Otherwise → HOLD
    Position gating: Signals are suppressed if they are invalid for the current position state (e.g., trying to open while already in a position, or closing the wrong side).

"""
import os
import sys
import logging
from typing import Dict, List, Any

import pandas as pd

from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.strategies.indicators_library import sma, ema

# Add parent directories to path for proper imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Configure logging
logger = logging.getLogger(__name__)

# STRATEGY_PARAMETERS - GUI Configuration (AT TOP)
# This defines what parameters appear in the GUI for users to configure
STRATEGY_PARAMETERS = {
    'fast_period': {
        'type': 'int',
        'default': 12,
        'min': 5,
        'max': 50,
        'description': 'Fast moving average period',
        'gui_hint': 'Lower values = more sensitive signals'
    },
    'slow_period': {
        'type': 'int',
        'default': 26,
        'min': 10,
        'max': 100,
        'description': 'Slow moving average period',
        'gui_hint': 'Should be 2-3x the fast period'
    },
    'ma_type': {
        'type': 'str',
        'default': 'ema',
        'options': ['sma', 'ema'],
        'description': 'Moving average type',
        'gui_hint': 'EMA reacts faster to price changes'
    }
}


class TrendFollowingStrategy(StrategyBase):
    """
    Trend following strategy using MA crossovers.

    Signal schema:
      - OPEN_LONG  when fast MA crosses above slow MA and no open position.
      - CLOSE_SHORT when fast MA crosses above slow MA and currently short.
      - OPEN_SHORT when fast MA crosses below slow MA and no open position.
      - CLOSE_LONG when fast MA crosses below slow MA and currently long.
      - HOLD otherwise.
    """

    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Trend_Following",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )
        self.fast_period = config.get('fast_period', 12)
        self.slow_period = config.get('slow_period', 26)
        self.ma_type = config.get('ma_type', 'ema')
        self._position_state: Dict[tuple, Dict[str, Any]] = {}

    def _calculate_ma(self, series: pd.Series) -> pd.Series:
        if self.ma_type == 'sma':
            return sma(series, period=self.fast_period), sma(series, period=self.slow_period)
        return ema(series, period=self.fast_period), ema(series, period=self.slow_period)

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
        for symbol in data:
            signals[symbol] = {}
            for timeframe, df in data[symbol].items():
                if df is None or len(df) < self.slow_period + 1:
                    signals[symbol][timeframe] = 'HOLD'
                    continue

                close_series = df['close']
                fast_ma, slow_ma = self._calculate_ma(close_series)

                prev_fast = fast_ma.iloc[-2]
                prev_slow = slow_ma.iloc[-2]
                last_fast = fast_ma.iloc[-1]
                last_slow = slow_ma.iloc[-1]

                position_key = (symbol, timeframe)
                position = self._position_state.get(position_key)
                is_short = position.get('is_short') if position else None

                cross_up = (last_fast > last_slow) and (prev_fast <= prev_slow)
                cross_down = (last_fast < last_slow) and (prev_fast >= prev_slow)

                if cross_up:
                    raw_signal = 'CLOSE_SHORT' if is_short else 'OPEN_LONG'
                elif cross_down:
                    raw_signal = 'CLOSE_LONG' if is_short is False else 'OPEN_SHORT'
                else:
                    raw_signal = 'HOLD'

                signals[symbol][timeframe] = self._apply_position_rules(position_key, raw_signal)

        return signals


def create_strategy(symbols=None, timeframes=None, **params):
    """
    Create Trend Following Strategy
    Uses moving average crossovers to identify trend direction
    """
    if symbols is None or len(symbols) == 0:
        symbols = ['BTCUSDT']
    if timeframes is None or len(timeframes) == 0:
        timeframes = ['1m']

    config = {
        'fast_period': params.get('fast_period', 12),
        'slow_period': params.get('slow_period', 26),
        'ma_type': params.get('ma_type', 'ema')
    }
    return TrendFollowingStrategy(symbols, timeframes, config)


def simple_test():
    """Simple test to verify the strategy works"""
    try:
        strategy = create_strategy(
            symbols=['BTCUSDT'],
            timeframes=['1m'],
            fast_period=12,
            slow_period=26,
            ma_type='ema'
        )
        print(f"✅ Trend Following strategy created successfully: {strategy.name}")
        print(f"  - Symbols: {strategy.symbols}")
        print(f"  - Timeframes: {strategy.timeframes}")
        return True
    except Exception as e:
        print(f"❌ Error testing Trend Following strategy: {e}")
        return False


if __name__ == "__main__":
    simple_test()
