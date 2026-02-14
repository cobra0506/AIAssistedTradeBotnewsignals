"""
Strategy: Evolved 1m Bidirectional

Goal:
- Explore different indicator + signal combinations using the shared libraries.
- Keep signal schema: OPEN_LONG, CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT, HOLD.
- Use bidirectional trading by default.
"""
import os
import sys
import logging
from typing import Dict, List, Any

import pandas as pd
import numpy as np

from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.strategies.indicators_library import (
    rsi, ema, macd, bollinger_bands, stochastic, atr, highest, lowest
)
from simple_strategy.strategies.signals_library import (
    rsi_mean_reversion_with_trend,
    macd_signals,
    bollinger_bands_signals,
    stochastic_signals,
    ma_crossover,
    breakout_signals
)

# Add parent directories to path for proper imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

STRUCTURE_DEFS = {
    "rsi_trend": {"signals": ["rsi_trend"], "combine": "single"},
    "ma_cross": {"signals": ["ma_cross"], "combine": "single"},
    "macd": {"signals": ["macd"], "combine": "single"},
    "bollinger": {"signals": ["bollinger"], "combine": "single"},
    "stoch": {"signals": ["stoch"], "combine": "single"},
    "rsi_macd": {"signals": ["rsi_trend", "macd"], "combine": "majority"},
    "rsi_macd_and": {"signals": ["rsi_trend", "macd"], "combine": "and"},
    "rsi_bollinger": {"signals": ["rsi_trend", "bollinger"], "combine": "majority"},
    "rsi_bollinger_and": {"signals": ["rsi_trend", "bollinger"], "combine": "and"},
    "macd_bollinger": {"signals": ["macd", "bollinger"], "combine": "majority"},
    "macd_bollinger_and": {"signals": ["macd", "bollinger"], "combine": "and"},
    "rsi_macd_bollinger": {"signals": ["rsi_trend", "macd", "bollinger"], "combine": "majority"},
    "rsi_stoch": {"signals": ["rsi_trend", "stoch"], "combine": "majority"},
    "rsi_stoch_and": {"signals": ["rsi_trend", "stoch"], "combine": "and"},
    "breakout": {"signals": ["breakout"], "combine": "single"},
    "breakout_rsi": {"signals": ["breakout", "rsi_trend"], "combine": "and"},
}

STRATEGY_PARAMETERS = {
    "structure_id": {
        "type": "categorical",
        "default": "rsi_trend",
        "choices": list(STRUCTURE_DEFS.keys()),
        "description": "Which signal structure to use",
        "gui_hint": "Pick one of the predefined structures"
    },
    "rsi_period": {
        "type": "int",
        "default": 14,
        "min": 5,
        "max": 40,
        "description": "RSI period",
        "gui_hint": "Typical 14"
    },
    "rsi_overbought": {
        "type": "int",
        "default": 70,
        "min": 55,
        "max": 90,
        "description": "RSI overbought level",
        "gui_hint": "Typical 70"
    },
    "rsi_oversold": {
        "type": "int",
        "default": 30,
        "min": 10,
        "max": 45,
        "description": "RSI oversold level",
        "gui_hint": "Typical 30"
    },
    "trend_fast_ema": {
        "type": "int",
        "default": 20,
        "min": 5,
        "max": 60,
        "description": "Fast EMA period",
        "gui_hint": "Lower = faster"
    },
    "trend_slow_ema": {
        "type": "int",
        "default": 50,
        "min": 10,
        "max": 200,
        "description": "Slow EMA period",
        "gui_hint": "Higher = smoother"
    },
    "macd_fast": {
        "type": "int",
        "default": 12,
        "min": 5,
        "max": 30,
        "description": "MACD fast period",
        "gui_hint": "Typical 12"
    },
    "macd_slow": {
        "type": "int",
        "default": 26,
        "min": 10,
        "max": 60,
        "description": "MACD slow period",
        "gui_hint": "Typical 26"
    },
    "macd_signal": {
        "type": "int",
        "default": 9,
        "min": 3,
        "max": 20,
        "description": "MACD signal period",
        "gui_hint": "Typical 9"
    },
    "bb_period": {
        "type": "int",
        "default": 20,
        "min": 10,
        "max": 60,
        "description": "Bollinger period",
        "gui_hint": "Typical 20"
    },
    "bb_std": {
        "type": "float",
        "default": 2.0,
        "min": 1.5,
        "max": 3.5,
        "description": "Bollinger standard deviation",
        "gui_hint": "Typical 2.0"
    },
    "stoch_k": {
        "type": "int",
        "default": 14,
        "min": 5,
        "max": 30,
        "description": "Stochastic K period",
        "gui_hint": "Typical 14"
    },
    "stoch_d": {
        "type": "int",
        "default": 3,
        "min": 2,
        "max": 10,
        "description": "Stochastic D period",
        "gui_hint": "Typical 3"
    },
    "stoch_overbought": {
        "type": "int",
        "default": 80,
        "min": 60,
        "max": 95,
        "description": "Stochastic overbought",
        "gui_hint": "Typical 80"
    },
    "stoch_oversold": {
        "type": "int",
        "default": 20,
        "min": 5,
        "max": 40,
        "description": "Stochastic oversold",
        "gui_hint": "Typical 20"
    },
    "breakout_lookback": {
        "type": "int",
        "default": 20,
        "min": 5,
        "max": 80,
        "description": "Breakout lookback period",
        "gui_hint": "Higher = fewer breakouts"
    },
    "breakout_penetration_pct": {
        "type": "float",
        "default": 0.2,
        "min": 0.05,
        "max": 1.0,
        "description": "Breakout penetration percent",
        "gui_hint": "0.2 = 0.2%"
    },
    "use_atr_filter": {
        "type": "bool",
        "default": True,
        "description": "Use ATR filter for entries",
        "gui_hint": "Filters low volatility"
    },
    "atr_period": {
        "type": "int",
        "default": 14,
        "min": 5,
        "max": 40,
        "description": "ATR period",
        "gui_hint": "Typical 14"
    },
    "atr_min_pct": {
        "type": "float",
        "default": 0.3,
        "min": 0.05,
        "max": 2.0,
        "description": "Minimum ATR percent",
        "gui_hint": "0.3 = 0.3%"
    },
    "use_ema_spread_filter": {
        "type": "bool",
        "default": True,
        "description": "Use EMA spread filter for entries",
        "gui_hint": "Filters weak trends"
    },
    "ema_spread_min_pct": {
        "type": "float",
        "default": 0.1,
        "min": 0.01,
        "max": 1.0,
        "description": "Minimum EMA spread percent",
        "gui_hint": "0.1 = 0.1%"
    },
    "cooldown_bars": {
        "type": "int",
        "default": 5,
        "min": 0,
        "max": 20,
        "description": "Bars to wait after any trade action",
        "gui_hint": "Prevents rapid re-entries"
    },
    "bidirectional": {
        "type": "bool",
        "default": True,
        "description": "Enable long and short trading",
        "gui_hint": "Should be True"
    }
}


class EvolvedStrategy1m(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="Evolved_1m_Bidirectional",
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )

        self.structure_id = config.get("structure_id", "rsi_trend")
        self.rsi_period = int(config.get("rsi_period", 14))
        self.rsi_overbought = int(config.get("rsi_overbought", 70))
        self.rsi_oversold = int(config.get("rsi_oversold", 30))
        self.trend_fast_ema = int(config.get("trend_fast_ema", 20))
        self.trend_slow_ema = int(config.get("trend_slow_ema", 50))

        self.macd_fast = int(config.get("macd_fast", 12))
        self.macd_slow = int(config.get("macd_slow", 26))
        self.macd_signal = int(config.get("macd_signal", 9))

        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))

        self.stoch_k = int(config.get("stoch_k", 14))
        self.stoch_d = int(config.get("stoch_d", 3))
        self.stoch_overbought = int(config.get("stoch_overbought", 80))
        self.stoch_oversold = int(config.get("stoch_oversold", 20))

        self.breakout_lookback = int(config.get("breakout_lookback", 20))
        self.breakout_penetration_pct = float(config.get("breakout_penetration_pct", 0.2))

        self.use_atr_filter = bool(config.get("use_atr_filter", True))
        self.atr_period = int(config.get("atr_period", 14))
        self.atr_min_pct = float(config.get("atr_min_pct", 0.3))

        self.use_ema_spread_filter = bool(config.get("use_ema_spread_filter", True))
        self.ema_spread_min_pct = float(config.get("ema_spread_min_pct", 0.1))

        self.cooldown_bars = int(config.get("cooldown_bars", 5))
        self.bidirectional = bool(config.get("bidirectional", True))

        self._position_state: Dict[tuple, Dict[str, Any]] = {}
        self._cooldown_state: Dict[tuple, pd.Timestamp] = {}

        self._sanitize_params()

    def _sanitize_params(self) -> None:
        if self.rsi_oversold >= self.rsi_overbought:
            self.rsi_oversold = max(10, self.rsi_overbought - 5)

        if self.trend_slow_ema <= self.trend_fast_ema:
            self.trend_slow_ema = self.trend_fast_ema + 1

        if self.macd_slow <= self.macd_fast:
            self.macd_slow = self.macd_fast + 1

        if self.stoch_d >= self.stoch_k:
            self.stoch_d = max(2, self.stoch_k - 1)

        if self.breakout_penetration_pct < 0:
            self.breakout_penetration_pct = 0.0

        if self.bb_std <= 0:
            self.bb_std = 2.0

    def _apply_position_rules(self, position_key: tuple, raw_signal: str) -> str:
        position = self._position_state.get(position_key)

        if raw_signal == "OPEN_LONG":
            if position is not None:
                return "HOLD"
            self._position_state[position_key] = {"is_short": False}
            return raw_signal

        if raw_signal == "OPEN_SHORT":
            if position is not None:
                return "HOLD"
            self._position_state[position_key] = {"is_short": True}
            return raw_signal

        if raw_signal == "CLOSE_LONG":
            if position is None or position.get("is_short", False):
                return "HOLD"
            self._position_state.pop(position_key, None)
            return raw_signal

        if raw_signal == "CLOSE_SHORT":
            if position is None or not position.get("is_short", False):
                return "HOLD"
            self._position_state.pop(position_key, None)
            return raw_signal

        return "HOLD"

    def _combine_signals(self, signals_list: List[pd.Series], index: pd.Index, method: str) -> pd.Series:
        if not signals_list:
            return pd.Series("HOLD", index=index)

        aligned = [s.reindex(index).fillna("HOLD") for s in signals_list]

        if method == "single":
            return aligned[0]

        if method == "and":
            result = pd.Series("HOLD", index=index)
            for action in ["OPEN_LONG", "OPEN_SHORT", "CLOSE_LONG", "CLOSE_SHORT"]:
                mask = pd.Series(True, index=index)
                for series in aligned:
                    mask &= (series == action)
                result[mask] = action
            return result

        # majority vote
        stacked = pd.concat(aligned, axis=1)
        votes = {}
        for action in ["OPEN_LONG", "OPEN_SHORT", "CLOSE_LONG", "CLOSE_SHORT"]:
            votes[action] = (stacked == action).sum(axis=1)
        max_counts = pd.concat(votes.values(), axis=1).max(axis=1)
        result = pd.Series("HOLD", index=index)
        majority_threshold = len(aligned) / 2.0
        for action, counts in votes.items():
            result[(counts > majority_threshold) & (counts == max_counts)] = action
        return result

    def _build_raw_signal_series(self, df: pd.DataFrame) -> pd.Series:
        close_series = df["close"]
        high_series = df["high"]
        low_series = df["low"]

        rsi_series = rsi(close_series, period=self.rsi_period)
        ema_fast = ema(close_series, period=self.trend_fast_ema)
        ema_slow = ema(close_series, period=self.trend_slow_ema)

        macd_line, macd_signal_line, macd_hist = macd(
            close_series,
            fast_period=self.macd_fast,
            slow_period=self.macd_slow,
            signal_period=self.macd_signal
        )

        bb_upper, bb_mid, bb_lower = bollinger_bands(
            close_series,
            period=self.bb_period,
            std_dev=self.bb_std
        )

        stoch_k_series, stoch_d_series = stochastic(
            high_series,
            low_series,
            close_series,
            k_period=self.stoch_k,
            d_period=self.stoch_d
        )

        resistance = highest(close_series, period=self.breakout_lookback)
        support = lowest(close_series, period=self.breakout_lookback)

        atr_series = None
        if self.use_atr_filter:
            atr_series = atr(high_series, low_series, close_series, period=self.atr_period)

        signals_map = {
            "rsi_trend": rsi_mean_reversion_with_trend(
                rsi_series, ema_fast, ema_slow,
                overbought=self.rsi_overbought,
                oversold=self.rsi_oversold
            ),
            "macd": macd_signals(macd_line, macd_signal_line, macd_hist),
            "bollinger": bollinger_bands_signals(close_series, bb_upper, bb_lower, bb_mid),
            "stoch": stochastic_signals(
                stoch_k_series, stoch_d_series,
                overbought=self.stoch_overbought,
                oversold=self.stoch_oversold
            ),
            "ma_cross": ma_crossover(ema_fast, ema_slow),
            "breakout": breakout_signals(
                close_series, resistance, support,
                penetration_pct=self.breakout_penetration_pct / 100.0
            )
        }

        structure = STRUCTURE_DEFS.get(self.structure_id, STRUCTURE_DEFS["rsi_trend"])
        selected = [signals_map[name] for name in structure["signals"] if name in signals_map]
        combined = self._combine_signals(selected, df.index, structure["combine"])

        entry_ok = pd.Series(True, index=df.index)
        if self.use_ema_spread_filter:
            ema_spread_pct = (ema_fast - ema_slow).abs() / close_series * 100.0
            entry_ok &= (ema_spread_pct >= self.ema_spread_min_pct)
        if self.use_atr_filter and atr_series is not None:
            atr_pct = atr_series / close_series * 100.0
            entry_ok &= (atr_pct >= self.atr_min_pct)

        open_mask = combined.isin(["OPEN_LONG", "OPEN_SHORT"])
        combined[open_mask & ~entry_ok] = "HOLD"

        if not self.bidirectional:
            combined[combined == "OPEN_SHORT"] = "HOLD"
            combined[combined == "CLOSE_SHORT"] = "HOLD"

        return combined

    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        signals: Dict[str, Dict[str, str]] = {}
        min_periods = max(self.rsi_period, self.trend_slow_ema, self.bb_period, self.atr_period) + 1

        for symbol in data:
            signals[symbol] = {}
            for timeframe, df in data[symbol].items():
                if df is None or len(df) < min_periods:
                    signals[symbol][timeframe] = "HOLD"
                    continue

                raw_series = self._build_raw_signal_series(df)
                raw_signal = raw_series.iloc[-1] if len(raw_series) else "HOLD"

                position_key = (symbol, timeframe)

                if self.cooldown_bars > 0 and raw_signal in ("OPEN_LONG", "OPEN_SHORT"):
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
                            if bars_since <= self.cooldown_bars:
                                raw_signal = "HOLD"

                applied_signal = self._apply_position_rules(position_key, raw_signal)
                if applied_signal != "HOLD":
                    self._cooldown_state[position_key] = df.index[-1]
                signals[symbol][timeframe] = applied_signal

        return signals

    def generate_signals_vectorized(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.Series]]:
        signals: Dict[str, Dict[str, pd.Series]] = {}
        min_periods = max(self.rsi_period, self.trend_slow_ema, self.bb_period, self.atr_period) + 1

        for symbol in data:
            signals[symbol] = {}
            for timeframe, df in data[symbol].items():
                if df is None or len(df) < min_periods:
                    signals[symbol][timeframe] = pd.Series(["HOLD"] * len(df), index=df.index)
                    continue

                raw = self._build_raw_signal_series(df)
                position_key = (symbol, timeframe)
                position = self._position_state.get(position_key)
                signals_list = []
                cooldown_remaining = 0
                last_action_time = None

                for i, raw_signal in enumerate(raw):
                    in_cooldown = self.cooldown_bars > 0 and cooldown_remaining > 0
                    if in_cooldown and raw_signal in ("OPEN_LONG", "OPEN_SHORT"):
                        raw_signal = "HOLD"

                    if raw_signal == "OPEN_LONG":
                        if position is None:
                            position = {"is_short": False}
                            signals_list.append("OPEN_LONG")
                            last_action_time = df.index[i]
                            cooldown_remaining = self.cooldown_bars
                        else:
                            signals_list.append("HOLD")
                    elif raw_signal == "OPEN_SHORT":
                        if position is None:
                            position = {"is_short": True}
                            signals_list.append("OPEN_SHORT")
                            last_action_time = df.index[i]
                            cooldown_remaining = self.cooldown_bars
                        else:
                            signals_list.append("HOLD")
                    elif raw_signal == "CLOSE_LONG":
                        if position is not None and not position.get("is_short", False):
                            position = None
                            signals_list.append("CLOSE_LONG")
                            last_action_time = df.index[i]
                            cooldown_remaining = self.cooldown_bars
                        else:
                            signals_list.append("HOLD")
                    elif raw_signal == "CLOSE_SHORT":
                        if position is not None and position.get("is_short", False):
                            position = None
                            signals_list.append("CLOSE_SHORT")
                            last_action_time = df.index[i]
                            cooldown_remaining = self.cooldown_bars
                        else:
                            signals_list.append("HOLD")
                    else:
                        signals_list.append("HOLD")

                    if cooldown_remaining > 0 and signals_list[-1] == "HOLD":
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
        symbols = ["BTCUSDT"]
    if timeframes is None or len(timeframes) == 0:
        timeframes = ["1m"]

    config = {
        "structure_id": params.get("structure_id", "rsi_trend"),
        "rsi_period": params.get("rsi_period", 14),
        "rsi_overbought": params.get("rsi_overbought", 70),
        "rsi_oversold": params.get("rsi_oversold", 30),
        "trend_fast_ema": params.get("trend_fast_ema", 20),
        "trend_slow_ema": params.get("trend_slow_ema", 50),
        "macd_fast": params.get("macd_fast", 12),
        "macd_slow": params.get("macd_slow", 26),
        "macd_signal": params.get("macd_signal", 9),
        "bb_period": params.get("bb_period", 20),
        "bb_std": params.get("bb_std", 2.0),
        "stoch_k": params.get("stoch_k", 14),
        "stoch_d": params.get("stoch_d", 3),
        "stoch_overbought": params.get("stoch_overbought", 80),
        "stoch_oversold": params.get("stoch_oversold", 20),
        "breakout_lookback": params.get("breakout_lookback", 20),
        "breakout_penetration_pct": params.get("breakout_penetration_pct", 0.2),
        "use_atr_filter": params.get("use_atr_filter", True),
        "atr_period": params.get("atr_period", 14),
        "atr_min_pct": params.get("atr_min_pct", 0.3),
        "use_ema_spread_filter": params.get("use_ema_spread_filter", True),
        "ema_spread_min_pct": params.get("ema_spread_min_pct", 0.1),
        "cooldown_bars": params.get("cooldown_bars", 5),
        "bidirectional": True
    }
    return EvolvedStrategy1m(symbols, timeframes, config)


def simple_test():
    try:
        strategy = create_strategy(
            symbols=["BTCUSDT"],
            timeframes=["1m"]
        )
        print(f"Strategy created: {strategy.name}")
        return True
    except Exception as e:
        print(f"Error testing strategy: {e}")
        return False


if __name__ == "__main__":
    simple_test()
