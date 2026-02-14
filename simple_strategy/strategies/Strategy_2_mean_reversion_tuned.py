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
import numpy as np

from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.strategies.indicators_library import rsi, ema, atr

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
    },
    'use_ema_spread_filter': {
        'type': 'bool',
        'default': False,
        'description': 'Require minimum EMA spread before entries',
        'gui_hint': 'Filters weak trends; uses ema_spread_min_pct'
    },
    'ema_spread_min_pct': {
        'type': 'float',
        'default': 0.05,
        'min': 0.0,
        'max': 2.0,
        'description': 'Minimum EMA spread as percent of price',
        'gui_hint': 'Example: 0.05 = 0.05%'
    },
    'use_atr_filter': {
        'type': 'bool',
        'default': False,
        'description': 'Require minimum ATR volatility before entries',
        'gui_hint': 'Filters low-volatility periods'
    },
    'atr_period': {
        'type': 'int',
        'default': 14,
        'min': 2,
        'max': 100,
        'description': 'ATR period',
        'gui_hint': 'Used only if use_atr_filter is enabled'
    },
    'atr_min_pct': {
        'type': 'float',
        'default': 0.2,
        'min': 0.0,
        'max': 5.0,
        'description': 'Minimum ATR as percent of price',
        'gui_hint': 'Example: 0.2 = 0.2%'
    },
    'use_rsi_mid_exit': {
        'type': 'bool',
        'default': False,
        'description': 'Exit at RSI mid-level before extreme levels',
        'gui_hint': 'Uses rsi_exit_level'
    },
    'rsi_exit_level': {
        'type': 'int',
        'default': 50,
        'min': 40,
        'max': 60,
        'description': 'RSI mid-level for early exit',
        'gui_hint': '50 is typical'
    },
    'cooldown_bars': {
        'type': 'int',
        'default': 0,
        'min': 0,
        'max': 50,
        'description': 'Bars to wait after any trade action',
        'gui_hint': 'Prevents rapid re-entries'
    }
}


class MeanReversionStrategy(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Mean_Reversion_Bidirectional_Tuned",
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
        self.use_ema_spread_filter = config.get('use_ema_spread_filter', False)
        self.ema_spread_min_pct = config.get('ema_spread_min_pct', 0.05)
        self.use_atr_filter = config.get('use_atr_filter', False)
        self.atr_period = config.get('atr_period', 14)
        self.atr_min_pct = config.get('atr_min_pct', 0.2)
        self.use_rsi_mid_exit = config.get('use_rsi_mid_exit', False)
        self.rsi_exit_level = config.get('rsi_exit_level', 50)
        self.cooldown_bars = config.get('cooldown_bars', 0)
        self._position_state: Dict[tuple, Dict[str, Any]] = {}
        self._cooldown_state: Dict[tuple, pd.Timestamp] = {}

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
        atr_period = self.atr_period if self.use_atr_filter else 0
        min_periods = max(self.rsi_period, self.trend_slow_ema, atr_period) + 1

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
                atr_series = None
                if self.use_atr_filter:
                    atr_series = atr(df['high'], df['low'], df['close'], period=self.atr_period)

                prev_rsi = rsi_series.iloc[-2]
                last_rsi = rsi_series.iloc[-1]
                last_ema_fast = ema_fast.iloc[-1]
                last_ema_slow = ema_slow.iloc[-1]
                last_close = close_series.iloc[-1]

                uptrend = last_ema_fast > last_ema_slow

                cross_below_oversold = (last_rsi < self.rsi_oversold) and (prev_rsi >= self.rsi_oversold)
                cross_above_overbought = (last_rsi > self.rsi_overbought) and (prev_rsi <= self.rsi_overbought)
                cross_above_exit = (last_rsi > self.rsi_exit_level) and (prev_rsi <= self.rsi_exit_level)
                cross_below_exit = (last_rsi < self.rsi_exit_level) and (prev_rsi >= self.rsi_exit_level)

                position_key = (symbol, timeframe)
                position = self._position_state.get(position_key)
                is_short = position.get('is_short') if position else None

                entry_ok = True
                if last_close == 0 or pd.isna(last_close):
                    entry_ok = False
                if self.use_ema_spread_filter and entry_ok:
                    ema_spread_pct = abs(last_ema_fast - last_ema_slow) / last_close * 100
                    if pd.isna(ema_spread_pct) or ema_spread_pct < self.ema_spread_min_pct:
                        entry_ok = False
                if self.use_atr_filter and entry_ok:
                    last_atr = atr_series.iloc[-1] if atr_series is not None else np.nan
                    atr_pct = (last_atr / last_close * 100) if last_close else 0
                    if pd.isna(atr_pct) or atr_pct < self.atr_min_pct:
                        entry_ok = False

                in_cooldown = False
                if self.cooldown_bars > 0:
                    last_action_time = self._cooldown_state.get(position_key)
                    if last_action_time is not None:
                        try:
                            last_pos = df.index.get_loc(last_action_time)
                            if isinstance(last_pos, slice):
                                last_pos = last_pos.stop - 1
                            elif isinstance(last_pos, (np.ndarray, list)):
                                last_pos = last_pos[-1]
                        except KeyError:
                            last_pos = df.index.searchsorted(last_action_time) - 1
                        if isinstance(last_pos, (int, np.integer)) and last_pos >= 0:
                            bars_since = len(df) - 1 - last_pos
                            in_cooldown = bars_since <= self.cooldown_bars

                if uptrend:
                    if cross_below_oversold:
                        raw_signal = 'OPEN_LONG' if entry_ok else 'HOLD'
                    elif cross_above_overbought:
                        raw_signal = 'CLOSE_LONG'
                    else:
                        raw_signal = 'HOLD'
                else:
                    if not self.bidirectional:
                        raw_signal = 'HOLD'
                    elif cross_above_overbought:
                        raw_signal = 'OPEN_SHORT' if entry_ok else 'HOLD'
                    elif cross_below_oversold:
                        raw_signal = 'CLOSE_SHORT'
                    else:
                        raw_signal = 'HOLD'

                if self.use_rsi_mid_exit and position is not None:
                    if not is_short and cross_above_exit:
                        raw_signal = 'CLOSE_LONG'
                    elif is_short and cross_below_exit:
                        raw_signal = 'CLOSE_SHORT'

                if in_cooldown and raw_signal in ('OPEN_LONG', 'OPEN_SHORT'):
                    raw_signal = 'HOLD'

                applied_signal = self._apply_position_rules(position_key, raw_signal)
                if applied_signal != 'HOLD':
                    self._cooldown_state[position_key] = df.index[-1]
                signals[symbol][timeframe] = applied_signal

        return signals

    def generate_signals_vectorized(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.Series]]:
        signals: Dict[str, Dict[str, pd.Series]] = {}
        atr_period = self.atr_period if self.use_atr_filter else 0
        min_periods = max(self.rsi_period, self.trend_slow_ema, atr_period) + 1

        for symbol in data:
            signals[symbol] = {}
            for timeframe, df in data[symbol].items():
                if df is None or len(df) < min_periods:
                    signals[symbol][timeframe] = pd.Series(['HOLD'] * len(df), index=df.index)
                    continue

                close_series = df['close']
                rsi_series = rsi(close_series, period=self.rsi_period)
                ema_fast = ema(close_series, period=self.trend_fast_ema)
                ema_slow = ema(close_series, period=self.trend_slow_ema)
                atr_series = None
                if self.use_atr_filter:
                    atr_series = atr(df['high'], df['low'], df['close'], period=self.atr_period)

                uptrend = ema_fast > ema_slow
                cross_below_oversold = (rsi_series < self.rsi_oversold) & (rsi_series.shift(1) >= self.rsi_oversold)
                cross_above_overbought = (rsi_series > self.rsi_overbought) & (rsi_series.shift(1) <= self.rsi_overbought)
                cross_above_exit = (rsi_series > self.rsi_exit_level) & (rsi_series.shift(1) <= self.rsi_exit_level)
                cross_below_exit = (rsi_series < self.rsi_exit_level) & (rsi_series.shift(1) >= self.rsi_exit_level)

                entry_allowed = (close_series > 0).to_numpy()
                if self.use_ema_spread_filter:
                    ema_spread_pct = (ema_fast - ema_slow).abs() / close_series * 100
                    entry_allowed &= (ema_spread_pct >= self.ema_spread_min_pct)
                if self.use_atr_filter:
                    atr_pct = atr_series / close_series * 100
                    entry_allowed &= (atr_pct >= self.atr_min_pct)

                raw_up = np.where(cross_below_oversold & entry_allowed, 'OPEN_LONG',
                         np.where(cross_above_overbought, 'CLOSE_LONG', 'HOLD'))

                if self.bidirectional:
                    raw_down = np.where(cross_above_overbought & entry_allowed, 'OPEN_SHORT',
                               np.where(cross_below_oversold, 'CLOSE_SHORT', 'HOLD'))
                else:
                    raw_down = np.full(len(close_series), 'HOLD', dtype=object)

                raw = np.where(uptrend, raw_up, raw_down)

                position_key = (symbol, timeframe)
                position = self._position_state.get(position_key)
                signals_list = []
                cooldown_remaining = 0
                last_action_time = None

                for i, raw_signal in enumerate(raw):
                    in_cooldown = self.cooldown_bars > 0 and cooldown_remaining > 0
                    if self.use_rsi_mid_exit and position is not None:
                        if not position.get('is_short', False) and cross_above_exit.iloc[i]:
                            raw_signal = 'CLOSE_LONG'
                        elif position.get('is_short', False) and cross_below_exit.iloc[i]:
                            raw_signal = 'CLOSE_SHORT'

                    if in_cooldown and raw_signal in ('OPEN_LONG', 'OPEN_SHORT'):
                        raw_signal = 'HOLD'

                    if raw_signal == 'OPEN_LONG':
                        if position is None:
                            position = {'is_short': False}
                            signals_list.append('OPEN_LONG')
                            last_action_time = df.index[i]
                            cooldown_remaining = self.cooldown_bars
                        else:
                            signals_list.append('HOLD')
                    elif raw_signal == 'OPEN_SHORT':
                        if position is None:
                            position = {'is_short': True}
                            signals_list.append('OPEN_SHORT')
                            last_action_time = df.index[i]
                            cooldown_remaining = self.cooldown_bars
                        else:
                            signals_list.append('HOLD')
                    elif raw_signal == 'CLOSE_LONG':
                        if position is not None and not position.get('is_short', False):
                            position = None
                            signals_list.append('CLOSE_LONG')
                            last_action_time = df.index[i]
                            cooldown_remaining = self.cooldown_bars
                        else:
                            signals_list.append('HOLD')
                    elif raw_signal == 'CLOSE_SHORT':
                        if position is not None and position.get('is_short', False):
                            position = None
                            signals_list.append('CLOSE_SHORT')
                            last_action_time = df.index[i]
                            cooldown_remaining = self.cooldown_bars
                        else:
                            signals_list.append('HOLD')
                    else:
                        signals_list.append('HOLD')

                    if cooldown_remaining > 0 and (signals_list[-1] == 'HOLD'):
                        cooldown_remaining -= 1

                if position is None:
                    self._position_state.pop(position_key, None)
                else:
                    self._position_state[position_key] = position
                if last_action_time is not None:
                    self._cooldown_state[position_key] = last_action_time

                signals[symbol][timeframe] = pd.Series(signals_list, index=df.index)

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
        'bidirectional': params.get('bidirectional', True),
        'use_ema_spread_filter': params.get('use_ema_spread_filter', False),
        'ema_spread_min_pct': params.get('ema_spread_min_pct', 0.05),
        'use_atr_filter': params.get('use_atr_filter', False),
        'atr_period': params.get('atr_period', 14),
        'atr_min_pct': params.get('atr_min_pct', 0.2),
        'use_rsi_mid_exit': params.get('use_rsi_mid_exit', False),
        'rsi_exit_level': params.get('rsi_exit_level', 50),
        'cooldown_bars': params.get('cooldown_bars', 0)
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
