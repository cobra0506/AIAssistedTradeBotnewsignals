"""Exact-rule ports for the second direct nateemma archived batch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.strategies.imported_nateemma_direct_batch1_helper import (
    VALID_SIGNALS,
    _adx_components,
    _bollinger_bands,
    _bool_last,
    _crossed_above,
    _crossed_below,
    _ema,
    _fisher_rsi,
    _linear_regression_slope,
    _mfi,
    _mirror_threshold,
    _parabolic_sar,
    _regime,
    _room_to_lower,
    _room_to_upper,
    _series_last,
    _sma,
    _typical_price,
    _weighted_bollinger_bands,
    _macd,
)


def _donchian(df: pd.DataFrame, period: int) -> pd.DataFrame:
    upper = df["high"].rolling(period, min_periods=period).max()
    lower = df["low"].rolling(period, min_periods=period).min()
    mid = (upper + lower) / 2.0
    dist = upper - lower
    return pd.DataFrame(
        {
            "upper": upper,
            "lower": lower,
            "mid": mid,
            "gain": (upper - df["close"]) / df["close"].replace(0.0, np.nan),
            "hf": upper - dist * 0.236,
            "lf": upper - dist * 0.764,
        },
        index=df.index,
    )


@dataclass
class DirectBatch2Config:
    name: str
    entry_timeframe: str = "5m"
    trend_timeframe: str = "15m"
    trend_ma_period: int = 50
    trend_deadband_pct: float = 0.15
    ema003_short_period: int = 7
    ema003_long_period: int = 25
    ema50_period: int = 50


class ImportedNateemmaDirectBatch2Strategy(StrategyBase):
    def __init__(self, variant: str, symbols: List[str], timeframes: List[str], config: Dict):
        super().__init__(
            name=f"Strategy_Import_Nateemma_Direct_{variant}",
            symbols=symbols,
            timeframes=timeframes,
            config=config,
        )
        self.variant = variant
        self.variant_config = DirectBatch2Config(
            name=variant,
            trend_ma_period=max(2, int(config.get("trend_ma_period", 50))),
            trend_deadband_pct=float(config.get("trend_deadband_pct", 0.15)),
            ema003_short_period=max(2, int(config.get("ema003_short_period", 7))),
            ema003_long_period=max(3, int(config.get("ema003_long_period", 25))),
            ema50_period=max(2, int(config.get("ema50_period", 50))),
        )
        if self.variant_config.ema003_long_period <= self.variant_config.ema003_short_period:
            self.variant_config.ema003_long_period = self.variant_config.ema003_short_period + 1
        self._position_state = {}

    def _apply_position_rules(self, key, signal: str) -> str:
        position = self._position_state.get(key)
        if signal == "OPEN_LONG":
            if position is not None:
                return "HOLD"
            self._position_state[key] = {"is_short": False}
            return signal
        if signal == "OPEN_SHORT":
            if position is not None:
                return "HOLD"
            self._position_state[key] = {"is_short": True}
            return signal
        if signal == "CLOSE_LONG":
            if position is None or position.get("is_short", False):
                return "HOLD"
            self._position_state.pop(key, None)
            return signal
        if signal == "CLOSE_SHORT":
            if position is None or not position.get("is_short", False):
                return "HOLD"
            self._position_state.pop(key, None)
            return signal
        return "HOLD"

    def _signal_bollinger_bounce(self, df: pd.DataFrame) -> str:
        mfi = _mfi(df)
        fisher = _fisher_rsi(df["close"])
        bands = _weighted_bollinger_bands(_typical_price(df), window=20, stds=2.0)
        bb_gain = _room_to_upper(df["close"], bands["upper"])
        bb_drop = _room_to_lower(df["close"], bands["lower"])

        long_entry = (
            _series_last(fisher, 0.0) < -0.81
            and _series_last(bb_gain, 0.0) >= 0.04
            and float(df["close"].iloc[-1]) > float(df["open"].iloc[-1])
            and float(df["open"].iloc[-1]) < _series_last(bands["lower"], float(df["open"].iloc[-1]))
            and float(df["close"].iloc[-1]) >= _series_last(bands["lower"], float(df["close"].iloc[-1]))
            and float(df["volume"].iloc[-1]) > 0.0
        )
        short_entry = (
            _series_last(fisher, 0.0) > 0.81
            and _series_last(bb_drop, 0.0) >= 0.04
            and float(df["close"].iloc[-1]) < float(df["open"].iloc[-1])
            and float(df["open"].iloc[-1]) > _series_last(bands["upper"], float(df["open"].iloc[-1]))
            and float(df["close"].iloc[-1]) <= _series_last(bands["upper"], float(df["close"].iloc[-1]))
            and float(df["volume"].iloc[-1]) > 0.0
        )
        if long_entry:
            return "OPEN_LONG"
        if short_entry:
            return "OPEN_SHORT"
        return "HOLD"

    def _signal_buy_dips(self, df: pd.DataFrame) -> str:
        mfi = _mfi(df)
        macd_line, macd_signal, _ = _macd(df["close"])
        bands = _bollinger_bands(_typical_price(df), window=20, stds=2.0)
        bb_gain = _room_to_upper(df["close"], bands["upper"])
        bb_drop = _room_to_lower(df["close"], bands["lower"])
        long_entry = (
            _series_last(mfi, 0.0) >= 54.0
            and _series_last(bb_gain, 0.0) >= 0.07
            and _bool_last(_crossed_above(macd_line.fillna(0.0), macd_signal.fillna(0.0)))
            and float(df["volume"].iloc[-1]) > 0.0
        )
        short_entry = (
            _series_last(mfi, 100.0) <= 46.0
            and _series_last(bb_drop, 0.0) >= 0.07
            and _bool_last(_crossed_below(macd_line.fillna(0.0), macd_signal.fillna(0.0)))
            and float(df["volume"].iloc[-1]) > 0.0
        )
        if long_entry:
            return "OPEN_LONG"
        if short_entry:
            return "OPEN_SHORT"
        return "HOLD"

    def _signal_dcbb_bounce(self, df: pd.DataFrame) -> str:
        bands = _weighted_bollinger_bands(_typical_price(df), window=20, stds=2.0)
        dc = _donchian(df, period=52)
        dcbb_diff_lower = dc["lower"] - bands["lower"]
        dcbb_diff_upper = dc["upper"] - bands["upper"]
        adx, plus_dm, minus_dm, _, _, _, _ = _adx_components(df)
        sar = _parabolic_sar(df)
        long_entry = (
            bool(dcbb_diff_lower.notnull().iloc[-1])
            and float(df["close"].iloc[-1]) >= float(df["open"].iloc[-1])
            and _bool_last(_crossed_above(dcbb_diff_lower.fillna(0.0), 0.0))
            and float(df["close"].iloc[-1]) < _series_last(sar, float(df["close"].iloc[-1]))
            and _series_last(adx, 0.0) > 25.0
            and _series_last(plus_dm, 0.0) >= _series_last(minus_dm, 0.0)
        )
        short_entry = (
            bool(dcbb_diff_upper.notnull().iloc[-1])
            and float(df["close"].iloc[-1]) <= float(df["open"].iloc[-1])
            and _bool_last(_crossed_below(dcbb_diff_upper.fillna(0.0), 0.0))
            and float(df["close"].iloc[-1]) > _series_last(sar, float(df["close"].iloc[-1]))
            and _series_last(adx, 0.0) > 25.0
            and _series_last(minus_dm, 0.0) >= _series_last(plus_dm, 0.0)
        )
        if long_entry:
            return "OPEN_LONG"
        if short_entry:
            return "OPEN_SHORT"
        return "HOLD"

    def _signal_donchian_bounce(self, df: pd.DataFrame) -> str:
        dc = _donchian(df, period=60)
        adx, _, _, _, _, _, _ = _adx_components(df)
        ema50 = _ema(df["close"], 50)
        sma200 = _sma(df["close"], 200)
        upper_drop = (df["close"] - dc["lower"]) / df["close"].replace(0.0, np.nan)

        long_entry = (
            bool(dc["hf"].notnull().iloc[-1])
            and float(df["close"].iloc[-1]) > _series_last(sma200, float(df["close"].iloc[-1]))
            and _series_last(dc["gain"], 0.0) >= 0.05
            and (
                _bool_last(_crossed_above(df["close"], dc["lower"]))
                or (
                    float(df["close"].iloc[-1]) >= _series_last(dc["lower"], float(df["close"].iloc[-1]))
                    and float(df["close"].iloc[-2]) < _series_last(dc["lower"].shift(1), float(df["close"].iloc[-2]))
                )
            )
        )
        short_entry = (
            bool(dc["hf"].notnull().iloc[-1])
            and float(df["close"].iloc[-1]) < _series_last(sma200, float(df["close"].iloc[-1]))
            and _series_last(upper_drop, 0.0) >= 0.05
            and (
                _bool_last(_crossed_below(df["close"], dc["upper"]))
                or (
                    float(df["close"].iloc[-1]) <= _series_last(dc["upper"], float(df["close"].iloc[-1]))
                    and float(df["close"].iloc[-2]) > _series_last(dc["upper"].shift(1), float(df["close"].iloc[-2]))
                )
            )
        )
        if long_entry:
            return "OPEN_LONG"
        if short_entry:
            return "OPEN_SHORT"
        return "HOLD"

    def _signal_donchian_channel(self, df: pd.DataFrame) -> str:
        dc = _donchian(df, period=13)
        adx, plus_dm, minus_dm, _, _, dm_delta, _ = _adx_components(df)
        mfi = _mfi(df)
        macd_line, macd_signal, _ = _macd(df["close"])
        fisher = _fisher_rsi(df["close"])
        sma200 = _sma(df["close"], 200)

        long_entry = (
            float(df["close"].iloc[-1]) > _series_last(sma200, float(df["close"].iloc[-1]))
            and _series_last(mfi, 0.0) >= 5.0
            and _series_last(macd_line, 0.0) > _series_last(macd_signal, 0.0)
            and _series_last(fisher, 0.0) < 0.06
            and float(df["close"].iloc[-1]) >= _series_last(dc["upper"], float(df["close"].iloc[-1]))
        )
        short_entry = (
            float(df["close"].iloc[-1]) < _series_last(sma200, float(df["close"].iloc[-1]))
            and _series_last(mfi, 100.0) <= 95.0
            and _series_last(macd_line, 0.0) < _series_last(macd_signal, 0.0)
            and _series_last(fisher, 0.0) > _mirror_threshold(0.06)
            and float(df["close"].iloc[-1]) <= _series_last(dc["lower"], float(df["close"].iloc[-1]))
        )
        if long_entry:
            return "OPEN_LONG"
        if short_entry:
            return "OPEN_SHORT"
        return "HOLD"

    def _signal_ema003(self, df: pd.DataFrame) -> str:
        cfg = self.variant_config
        ema_short = _ema(df["close"], cfg.ema003_short_period)
        ema_long = _ema(df["close"], cfg.ema003_long_period)
        long_entry = _bool_last(_crossed_above(ema_short.fillna(0.0), ema_long.fillna(0.0)))
        short_entry = _bool_last(_crossed_below(ema_short.fillna(0.0), ema_long.fillna(0.0)))
        if long_entry:
            return "OPEN_LONG"
        if short_entry:
            return "OPEN_SHORT"
        return "HOLD"

    def _signal_ema50(self, df: pd.DataFrame) -> str:
        ema_line = _ema(df["close"], self.variant_config.ema50_period)
        long_entry = _bool_last(_crossed_above(df["close"], ema_line))
        short_entry = _bool_last(_crossed_below(df["close"], ema_line))
        if long_entry:
            return "OPEN_LONG"
        if short_entry:
            return "OPEN_SHORT"
        return "HOLD"

    def _signal_ema_bounce(self, df: pd.DataFrame) -> str:
        ema = _ema(df["close"], 50)
        ema_short = _ema(df["close"], 10)
        ema_angle = _linear_regression_slope(ema_short, period=3) / (2.0 * np.pi)
        ema_diff = ((ema - df["close"]) / ema.replace(0.0, np.nan)) - 0.065
        short_diff = ((df["close"] - ema) / ema.replace(0.0, np.nan)) - 0.065
        long_entry = (
            float(df["volume"].iloc[-1]) > 0.0
            and _bool_last(_crossed_above(ema_angle.fillna(0.0), 0.0))
            and _series_last(ema_diff, 0.0) > 0.0
        )
        short_entry = (
            float(df["volume"].iloc[-1]) > 0.0
            and _bool_last(_crossed_below(ema_angle.fillna(0.0), 0.0))
            and _series_last(short_diff, 0.0) > 0.0
        )
        if long_entry:
            return "OPEN_LONG"
        if short_entry:
            return "OPEN_SHORT"
        return "HOLD"

    def _signal_ema_breakout(self, df: pd.DataFrame) -> str:
        ema = _ema(df["close"], 90)
        macd_line, macd_signal, hist = _macd(df["close"])
        long_entry = (
            float(df["volume"].iloc[-1]) > 0.0
            and _series_last(hist, 0.0) >= 0.0
            and _bool_last(_crossed_above(df["close"], ema))
        )
        long_exit = _bool_last(_crossed_below(df["close"], ema))
        short_entry = (
            float(df["volume"].iloc[-1]) > 0.0
            and _series_last(hist, 0.0) <= 0.0
            and _bool_last(_crossed_below(df["close"], ema))
        )
        short_exit = _bool_last(_crossed_above(df["close"], ema))
        if long_entry:
            return "OPEN_LONG"
        if long_exit:
            return "CLOSE_LONG"
        if short_entry:
            return "OPEN_SHORT"
        if short_exit:
            return "CLOSE_SHORT"
        return "HOLD"

    def _signal_ema_cross(self, df: pd.DataFrame) -> str:
        ema_short = _ema(df["close"], 5)
        ema_long = _ema(df["close"], 10)
        if float(df["volume"].iloc[-1]) <= 0.0:
            return "HOLD"
        long_entry = _bool_last(_crossed_above(ema_short, ema_long))
        long_exit = _bool_last(_crossed_above(ema_long, ema_short))
        short_entry = _bool_last(_crossed_below(ema_short, ema_long))
        short_exit = _bool_last(_crossed_below(ema_long, ema_short))
        if long_entry:
            return "OPEN_LONG"
        if long_exit:
            return "CLOSE_LONG"
        if short_entry:
            return "OPEN_SHORT"
        if short_exit:
            return "CLOSE_SHORT"
        return "HOLD"

    def _compute_variant_signal(self, variant: str, entry_df: pd.DataFrame) -> str:
        if len(entry_df) < 220:
            return "HOLD"
        if variant == "BollingerBounce":
            return self._signal_bollinger_bounce(entry_df)
        if variant == "BuyDips":
            return self._signal_buy_dips(entry_df)
        if variant == "DCBBBounce":
            return self._signal_dcbb_bounce(entry_df)
        if variant == "DonchianBounce":
            return self._signal_donchian_bounce(entry_df)
        if variant == "DonchianChannel":
            return self._signal_donchian_channel(entry_df)
        if variant == "EMA003":
            return self._signal_ema003(entry_df)
        if variant == "EMA50":
            return self._signal_ema50(entry_df)
        if variant == "EMABounce":
            return self._signal_ema_bounce(entry_df)
        if variant == "EMABreakout":
            return self._signal_ema_breakout(entry_df)
        if variant == "EMACross":
            return self._signal_ema_cross(entry_df)
        return "HOLD"

    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        cfg = self.variant_config
        result: Dict[str, Dict[str, str]] = {}
        for symbol, tf_map in data.items():
            result[symbol] = {timeframe: "HOLD" for timeframe in tf_map}
            entry_df = tf_map.get(cfg.entry_timeframe)
            trend_df = tf_map.get(cfg.trend_timeframe)
            if entry_df is None or trend_df is None or entry_df.empty or trend_df.empty:
                continue
            raw_signal = self._compute_variant_signal(self.variant, entry_df.copy())
            market_regime = _regime(trend_df.copy(), cfg.trend_ma_period, cfg.trend_deadband_pct)
            if raw_signal == "OPEN_LONG" and market_regime != "uptrend":
                raw_signal = "HOLD"
            elif raw_signal == "OPEN_SHORT" and market_regime != "downtrend":
                raw_signal = "HOLD"
            position_key = (symbol, cfg.entry_timeframe)
            final_signal = self._apply_position_rules(position_key, raw_signal)
            result[symbol][cfg.entry_timeframe] = final_signal if final_signal in VALID_SIGNALS else "HOLD"
        return result


def create_direct_batch2_strategy(variant: str, symbols=None, timeframes=None, **config):
    symbols = list(symbols or ["BNBUSDT"])
    resolved_timeframes = list(timeframes or ["5m", "15m"])
    for required in ("5m", "15m"):
        if required not in resolved_timeframes:
            resolved_timeframes.append(required)
    return ImportedNateemmaDirectBatch2Strategy(
        variant=variant,
        symbols=symbols,
        timeframes=resolved_timeframes,
        config=config,
    )
