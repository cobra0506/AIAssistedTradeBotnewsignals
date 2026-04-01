from __future__ import annotations

from typing import Dict

import pandas as pd

from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.strategies.imported_nateemma_direct_batch1_helper import VALID_SIGNALS

STRATEGY_PARAMETERS = {
    "min_candle_pct": {"type": "float", "default": 2.0, "gui_hint": "Trigger candle range %"},
    "target_fraction": {"type": "float", "default": 0.5, "gui_hint": "0.5 half, 1.0 full candle"},
    "max_hold_bars": {"type": "int", "default": 0, "gui_hint": "0 disables timed exit"},
    "min_body_ratio": {"type": "float", "default": 0.0, "gui_hint": "0 disables body filter"},
    "volume_spike_multiplier": {"type": "float", "default": 0.0, "gui_hint": "0 disables volume filter"},
    "volume_lookback": {"type": "int", "default": 20, "gui_hint": "Bars for average volume"},
    "cooldown_bars": {"type": "int", "default": 0, "gui_hint": "Bars to wait after a close"},
}


class BigCandleHalfTargetStrategy(StrategyBase):
    def __init__(self, symbols=None, timeframes=None, config=None):
        strategy_config = config or {}
        super().__init__(
            name="Strategy_Custom_BigCandle_HalfTarget",
            symbols=list(symbols or ["BNBUSDT"]),
            timeframes=list(timeframes or ["5m", "15m"]),
            config=strategy_config,
        )
        self.entry_timeframe = "5m"
        self.min_candle_pct = float(strategy_config.get("min_candle_pct", 2.0))
        self.target_fraction = max(0.1, float(strategy_config.get("target_fraction", 0.5)))
        self.max_hold_bars = max(0, int(strategy_config.get("max_hold_bars", 0)))
        self.min_body_ratio = max(0.0, min(1.0, float(strategy_config.get("min_body_ratio", 0.0))))
        self.volume_spike_multiplier = max(0.0, float(strategy_config.get("volume_spike_multiplier", 0.0)))
        self.volume_lookback = max(2, int(strategy_config.get("volume_lookback", 20)))
        self.cooldown_bars = max(0, int(strategy_config.get("cooldown_bars", 0)))
        self.targets: Dict[tuple[str, str], dict] = {}
        self.cooldowns: Dict[tuple[str, str], int] = {}

    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        result: Dict[str, Dict[str, str]] = {}
        for symbol, tf_map in data.items():
            result[symbol] = {timeframe: "HOLD" for timeframe in tf_map}
            entry_df = tf_map.get(self.entry_timeframe)
            if entry_df is None or len(entry_df) < 2:
                continue

            key = (symbol, self.entry_timeframe)
            signal = "HOLD"
            current_close = float(entry_df["close"].iloc[-1])
            current_index = len(entry_df) - 1

            state = self.targets.get(key)
            if state is not None:
                bars_open = max(0, current_index - int(state.get("entry_index", current_index)))
                if self.max_hold_bars > 0 and bars_open >= self.max_hold_bars:
                    signal = "CLOSE_SHORT" if state["is_short"] else "CLOSE_LONG"
                    self.targets.pop(key, None)
                    self.cooldowns[key] = current_index + self.cooldown_bars
                elif not state["is_short"] and current_close >= state["target_price"]:
                    signal = "CLOSE_LONG"
                    self.targets.pop(key, None)
                    self.cooldowns[key] = current_index + self.cooldown_bars
                elif state["is_short"] and current_close <= state["target_price"]:
                    signal = "CLOSE_SHORT"
                    self.targets.pop(key, None)
                    self.cooldowns[key] = current_index + self.cooldown_bars

            if signal == "HOLD" and key not in self.targets:
                if current_index < int(self.cooldowns.get(key, -1)):
                    result[symbol][self.entry_timeframe] = "HOLD"
                    continue
                candle_open = float(entry_df["open"].iloc[-1])
                candle_close = float(entry_df["close"].iloc[-1])
                candle_high = float(entry_df["high"].iloc[-1])
                candle_low = float(entry_df["low"].iloc[-1])
                candle_range = candle_high - candle_low
                candle_range_pct = (candle_range / max(candle_open, 1e-9)) * 100.0
                candle_body_ratio = abs(candle_close - candle_open) / max(candle_range, 1e-9)
                volume_ok = True
                if self.volume_spike_multiplier > 0.0:
                    prior_volume = entry_df["volume"].iloc[:-1].tail(self.volume_lookback)
                    avg_volume = float(prior_volume.mean()) if len(prior_volume) >= self.volume_lookback else 0.0
                    current_volume = float(entry_df["volume"].iloc[-1])
                    volume_ok = avg_volume > 0.0 and current_volume >= avg_volume * self.volume_spike_multiplier
                if (
                    candle_range_pct >= self.min_candle_pct
                    and candle_body_ratio >= self.min_body_ratio
                    and volume_ok
                ):
                    if candle_close > candle_open:
                        signal = "OPEN_SHORT"
                        self.targets[key] = {
                            "is_short": True,
                            "target_price": candle_close - (candle_range * self.target_fraction),
                            "entry_index": current_index,
                        }
                    elif candle_close < candle_open:
                        signal = "OPEN_LONG"
                        self.targets[key] = {
                            "is_short": False,
                            "target_price": candle_close + (candle_range * self.target_fraction),
                            "entry_index": current_index,
                        }

            result[symbol][self.entry_timeframe] = signal if signal in VALID_SIGNALS else "HOLD"
        return result


def create_strategy(symbols=None, timeframes=None, **params):
    return BigCandleHalfTargetStrategy(symbols=symbols, timeframes=timeframes, config=params)
