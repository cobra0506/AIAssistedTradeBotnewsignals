"""Exact-rule ports for the first direct nateemma archived strategies.

These ports rebuild the real rule logic from the source strategies and then
adapt it to this project in one explicit way:
- uptrend: allow only new longs
- downtrend: allow only new shorts
- sideways: no new entries

Close signals stay allowed so positions can still unwind cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from simple_strategy.shared.strategy_base import StrategyBase

VALID_SIGNALS = {"OPEN_LONG", "CLOSE_LONG", "OPEN_SHORT", "CLOSE_SHORT", "HOLD"}


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=max(1, int(period)), adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=max(1, int(period)), min_periods=1).mean()


def _typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _true_range(df).rolling(window=max(1, int(period)), min_periods=1).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _fisher_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    rsi = _rsi(close, period=period)
    normalized = 0.1 * (rsi - 50.0)
    return (np.exp(2.0 * normalized) - 1.0) / (np.exp(2.0 * normalized) + 1.0)


def _mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = _typical_price(df)
    flow = tp * df["volume"]
    delta = tp.diff()
    positive = flow.where(delta > 0.0, 0.0).rolling(period, min_periods=period).sum()
    negative = flow.where(delta < 0.0, 0.0).abs().rolling(period, min_periods=period).sum()
    ratio = positive / negative.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + ratio))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_line = _ema(close, fast)
    slow_line = _ema(close, slow)
    macd_line = fast_line - slow_line
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _adx_components(df: pd.DataFrame, period: int = 14):
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=df.index,
    )
    atr = _atr(df, period=period)
    plus_di = 100.0 * plus_dm.rolling(period, min_periods=period).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.rolling(period, min_periods=period).mean() / atr.replace(0.0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)) * 100.0
    adx = dx.rolling(period, min_periods=period).mean()
    dm_delta = plus_dm - minus_dm
    adx_slope = adx.diff(3) / 3.0
    return adx, plus_dm, minus_dm, plus_di, minus_di, dm_delta, adx_slope


def _linear_regression_slope(series: pd.Series, period: int = 3) -> pd.Series:
    period = max(2, int(period))
    x = np.arange(period, dtype=float)

    def _slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        slope, _ = np.polyfit(x, values, 1)
        return float(slope)

    return series.rolling(period, min_periods=period).apply(_slope, raw=True)


def _weighted_bollinger_bands(series: pd.Series, window: int = 20, stds: float = 2.0) -> pd.DataFrame:
    mid = _ema(series, window)
    std = series.rolling(window, min_periods=window).std()
    return pd.DataFrame({"upper": mid + std * stds, "mid": mid, "lower": mid - std * stds}, index=series.index)


def _bollinger_bands(series: pd.Series, window: int = 20, stds: float = 2.0) -> pd.DataFrame:
    mid = _sma(series, window)
    std = series.rolling(window, min_periods=window).std()
    return pd.DataFrame({"upper": mid + std * stds, "mid": mid, "lower": mid - std * stds}, index=series.index)


def _keltner_channel(df: pd.DataFrame, window: int = 14, atrs: float = 2.0) -> pd.DataFrame:
    mid = _typical_price(df).rolling(window, min_periods=window).mean()
    atr = _atr(df, period=window) * float(atrs)
    return pd.DataFrame({"upper": mid + atr, "mid": mid, "lower": mid - atr}, index=df.index)


def _parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    result = np.full(len(df), np.nan, dtype=float)
    if len(df) < 2:
        return pd.Series(result, index=df.index)

    long_trend = True
    af = step
    ep = high[0]
    sar = low[0]
    result[0] = sar

    for i in range(1, len(df)):
        prev_sar = sar
        if long_trend:
            sar = prev_sar + af * (ep - prev_sar)
            sar = min(sar, low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if low[i] < sar:
                long_trend = False
                sar = ep
                ep = low[i]
                af = step
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_step)
        else:
            sar = prev_sar + af * (ep - prev_sar)
            sar = max(sar, high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if high[i] > sar:
                long_trend = True
                sar = ep
                ep = high[i]
                af = step
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_step)
        result[i] = sar

    return pd.Series(result, index=df.index)


def _crossed_above(series1: pd.Series, series2) -> pd.Series:
    if isinstance(series2, (float, int)):
        series2 = pd.Series(series2, index=series1.index)
    return pd.Series((series1 > series2) & (series1.shift(1) <= series2.shift(1)), index=series1.index)


def _crossed_below(series1: pd.Series, series2) -> pd.Series:
    if isinstance(series2, (float, int)):
        series2 = pd.Series(series2, index=series1.index)
    return pd.Series((series1 < series2) & (series1.shift(1) >= series2.shift(1)), index=series1.index)


def _room_to_upper(close: pd.Series, upper: pd.Series) -> pd.Series:
    return (upper - close) / close.replace(0.0, np.nan)


def _room_to_lower(close: pd.Series, lower: pd.Series) -> pd.Series:
    return (close - lower) / close.replace(0.0, np.nan)


def _regime(trend_df: pd.DataFrame, ma_period: int, deadband_pct: float) -> str:
    trend_line = _ema(trend_df["close"], ma_period)
    if trend_line.empty or pd.isna(trend_line.iloc[-1]):
        return "sideways"
    close_value = float(trend_df["close"].iloc[-1])
    ma_value = float(trend_line.iloc[-1])
    if ma_value <= 0.0:
        return "sideways"
    distance_pct = ((close_value - ma_value) / ma_value) * 100.0
    if distance_pct >= deadband_pct:
        return "uptrend"
    if distance_pct <= -deadband_pct:
        return "downtrend"
    return "sideways"


def _bool_last(series: pd.Series) -> bool:
    if series is None or len(series) == 0:
        return False
    value = series.iloc[-1]
    return bool(False if pd.isna(value) else value)


def _series_last(series: pd.Series, default: float = 0.0) -> float:
    if series is None or len(series) == 0:
        return float(default)
    value = series.iloc[-1]
    return float(default if pd.isna(value) else value)


def _mirror_threshold(value: float) -> float:
    return max(-1.0, min(1.0, -float(value)))


@dataclass
class DirectVariantConfig:
    name: str
    entry_timeframe: str = "5m"
    trend_timeframe: str = "15m"
    trend_ma_period: int = 50
    trend_deadband_pct: float = 0.15
    benchmark_symbol: str = "BTCUSDT"
    buy_adx: float = 60.0


class ImportedNateemmaDirectBatch1Strategy(StrategyBase):
    def __init__(self, variant: str, symbols: List[str], timeframes: List[str], config: Dict):
        super().__init__(
            name=f"Strategy_Import_Nateemma_Direct_{variant}",
            symbols=symbols,
            timeframes=timeframes,
            config=config,
        )
        self.variant = variant
        self.variant_config = DirectVariantConfig(
            name=variant,
            benchmark_symbol=str(config.get("benchmark_symbol", "BTCUSDT")),
            trend_ma_period=max(2, int(config.get("trend_ma_period", 50))),
            trend_deadband_pct=float(config.get("trend_deadband_pct", 0.15)),
            buy_adx=float(config.get("buy_adx", 60.0)),
        )
        self._position_state = {}

    def _benchmark_df(self, symbol: str, data: Dict[str, Dict[str, pd.DataFrame]]) -> Optional[pd.DataFrame]:
        benchmark_symbol = self.variant_config.benchmark_symbol
        if symbol == benchmark_symbol:
            return None
        benchmark_map = data.get(benchmark_symbol, {})
        return benchmark_map.get(self.variant_config.entry_timeframe)

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

    def _signal_adxdm(self, df: pd.DataFrame) -> str:
        cfg = self.variant_config
        adx, plus_dm, minus_dm, plus_di, minus_di, dm_delta, adx_slope = _adx_components(df)
        mfi = _mfi(df)
        bands = _bollinger_bands(_typical_price(df), window=20, stds=2.0)
        bb_gain = _room_to_upper(df["close"], bands["upper"])
        long_entry = (
            bool(adx.notnull().iloc[-1])
            and _series_last(mfi, 100.0) <= _series_last(adx, 0.0)
            and _series_last(adx, 0.0) > cfg.buy_adx
            and _bool_last(_crossed_below(adx_slope.fillna(0.0), 0.0))
            and _series_last(dm_delta, 0.0) < 0.0
        )
        long_exit = False
        short_entry = (
            bool(adx.notnull().iloc[-1])
            and _series_last(mfi, 0.0) >= 100.0 - min(cfg.buy_adx, _series_last(adx, 0.0))
            and _series_last(adx, 0.0) > cfg.buy_adx
            and _bool_last(_crossed_below(adx_slope.fillna(0.0), 0.0))
            and _series_last(dm_delta, 0.0) > 0.0
        )
        short_exit = False
        if long_entry:
            return "OPEN_LONG"
        if long_exit:
            return "CLOSE_LONG"
        if short_entry:
            return "OPEN_SHORT"
        if short_exit:
            return "CLOSE_SHORT"
        return "HOLD"

    def _signal_bbbhold(self, df: pd.DataFrame) -> str:
        mfi = _mfi(df)
        fisher = _fisher_rsi(df["close"])
        bands = _weighted_bollinger_bands(_typical_price(df), window=20, stds=2.0)
        bb_gain = _room_to_upper(df["close"], bands["upper"])
        bb_drop = _room_to_lower(df["close"], bands["lower"])

        long_entry = (
            _series_last(fisher, 0.0) < -0.53
            and _series_last(bb_gain, 0.0) >= 0.06
            and float(df["close"].iloc[-1]) > float(df["open"].iloc[-1])
            and float(df["open"].iloc[-1]) < _series_last(bands["lower"], float(df["open"].iloc[-1]))
            and float(df["close"].iloc[-1]) >= _series_last(bands["lower"], float(df["close"].iloc[-1]))
            and float(df["volume"].iloc[-1]) > 0.0
        )
        short_entry = (
            _series_last(fisher, 0.0) > 0.53
            and _series_last(bb_drop, 0.0) >= 0.06
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

    def _signal_bbkcbounce(self, df: pd.DataFrame) -> str:
        macd_line, macd_signal, _ = _macd(df["close"])
        mfi = _mfi(df)
        adx, _, _, _, _, dm_delta, _ = _adx_components(df)
        sar = _parabolic_sar(df)
        fisher = _fisher_rsi(df["close"])
        bb = _bollinger_bands(_typical_price(df), window=20, stds=2.0)
        kc = _keltner_channel(df, window=14, atrs=2.0)
        kc_gain = _room_to_upper(df["close"], kc["upper"])
        kc_drop = _room_to_lower(df["close"], kc["lower"])

        long_entry = (
            _series_last(mfi, 0.0) > 6.0
            and float(df["close"].iloc[-1]) < _series_last(sar, float(df["close"].iloc[-1]))
            and _series_last(fisher, 0.0) < -0.41
            and float(df["close"].iloc[-1]) > float(df["open"].iloc[-1])
            and _series_last(kc_gain, 0.0) >= 0.05
            and (
                _bool_last(_crossed_above(df["close"], bb["lower"]))
                or _bool_last(_crossed_above(df["close"], kc["lower"]))
            )
            and float(df["close"].iloc[-1]) >= _series_last(bb["lower"], float(df["close"].iloc[-1]))
            and float(df["close"].iloc[-1]) >= _series_last(kc["lower"], float(df["close"].iloc[-1]))
        )
        short_entry = (
            _series_last(mfi, 100.0) < 94.0
            and float(df["close"].iloc[-1]) > _series_last(sar, float(df["close"].iloc[-1]))
            and _series_last(fisher, 0.0) > 0.41
            and float(df["close"].iloc[-1]) < float(df["open"].iloc[-1])
            and _series_last(kc_drop, 0.0) >= 0.05
            and (
                _bool_last(_crossed_below(df["close"], bb["upper"]))
                or _bool_last(_crossed_below(df["close"], kc["upper"]))
            )
            and float(df["close"].iloc[-1]) <= _series_last(bb["upper"], float(df["close"].iloc[-1]))
            and float(df["close"].iloc[-1]) <= _series_last(kc["upper"], float(df["close"].iloc[-1]))
        )
        if long_entry:
            return "OPEN_LONG"
        if short_entry:
            return "OPEN_SHORT"
        return "HOLD"

    def _signal_btcbigdrop(self, df: pd.DataFrame, benchmark_df: pd.DataFrame) -> str:
        fisher = _fisher_rsi(df["close"])
        mfi = _mfi(df)
        bands = _weighted_bollinger_bands(_typical_price(df), window=20, stds=2.0)
        btc_red = [float(benchmark_df["close"].iloc[-1 - i]) <= float(benchmark_df["open"].iloc[-1 - i]) for i in range(3)]
        btc_green = [float(benchmark_df["close"].iloc[-1 - i]) >= float(benchmark_df["open"].iloc[-1 - i]) for i in range(3)]
        btc_drop = (float(benchmark_df["open"].iloc[-3]) - float(benchmark_df["close"].iloc[-1])) / max(float(benchmark_df["open"].iloc[-3]), 1e-9)
        btc_rise = (float(benchmark_df["close"].iloc[-1]) - float(benchmark_df["open"].iloc[-3])) / max(float(benchmark_df["open"].iloc[-3]), 1e-9)
        long_entry = (
            _series_last(fisher, 0.0) < -0.02
            and all(btc_red)
            and btc_drop >= 0.014
        )
        short_entry = (
            _series_last(fisher, 0.0) > 0.02
            and all(btc_green)
            and btc_rise >= 0.014
        )
        if long_entry:
            return "OPEN_LONG"
        if short_entry:
            return "OPEN_SHORT"
        return "HOLD"

    def _signal_btcemabounce(self, df: pd.DataFrame, benchmark_df: pd.DataFrame) -> str:
        btc_ema_short = _ema(benchmark_df["close"], 10)
        btc_angle = _linear_regression_slope(btc_ema_short, period=3) / (2.0 * np.pi)
        ema_long = _ema(df["close"], 50)
        macd_line, macd_signal, macd_hist = _macd(df["close"])
        ema_diff = ((ema_long - df["close"]) / ema_long.replace(0.0, np.nan)) - 0.065
        short_ema_diff = ((df["close"] - ema_long) / ema_long.replace(0.0, np.nan)) - 0.065
        long_entry = (
            float(df["volume"].iloc[-1]) > 0.0
            and _bool_last(_crossed_above(btc_angle.fillna(0.0), 0.0))
            and _series_last(ema_diff, 0.0) > 0.0
        )
        short_entry = (
            float(df["volume"].iloc[-1]) > 0.0
            and _bool_last(_crossed_below(btc_angle.fillna(0.0), 0.0))
            and _series_last(short_ema_diff, 0.0) > 0.0
        )
        long_exit = (
            _bool_last(_crossed_below(btc_angle.fillna(0.0), 0.0))
            and float(df["close"].iloc[-1]) > _series_last(ema_long, float(df["close"].iloc[-1]))
            and _series_last(ema_diff, 0.0) >= -0.057
        )
        short_exit = (
            _bool_last(_crossed_above(btc_angle.fillna(0.0), 0.0))
            and float(df["close"].iloc[-1]) < _series_last(ema_long, float(df["close"].iloc[-1]))
            and _series_last(short_ema_diff, 0.0) >= -0.057
        )
        if long_entry:
            return "OPEN_LONG"
        if long_exit:
            return "CLOSE_LONG"
        if short_entry:
            return "OPEN_SHORT"
        if short_exit:
            return "CLOSE_SHORT"
        return "HOLD"

    def _signal_btcjump(self, df: pd.DataFrame, benchmark_df: pd.DataFrame) -> str:
        fisher = _fisher_rsi(df["close"])
        bands = _weighted_bollinger_bands(_typical_price(df), window=20, stds=2.0)
        bb_gain = _room_to_upper(df["close"], bands["upper"])
        bb_drop = _room_to_lower(df["close"], bands["lower"])
        btc_gain = (benchmark_df["close"] - benchmark_df["open"]) / benchmark_df["open"].replace(0.0, np.nan)
        btc_zgain = btc_gain - 0.009
        btc_zdrop = btc_gain + 0.009
        long_entry = (
            _series_last(fisher, 0.0) <= -0.12
            and _series_last(bb_gain, 0.0) >= 0.09
            and _bool_last(_crossed_above(btc_zgain.fillna(0.0), 0.0))
        )
        short_entry = (
            _series_last(fisher, 0.0) >= 0.12
            and _series_last(bb_drop, 0.0) >= 0.09
            and _bool_last(_crossed_below(btc_zdrop.fillna(0.0), 0.0))
        )
        if long_entry:
            return "OPEN_LONG"
        if short_entry:
            return "OPEN_SHORT"
        return "HOLD"

    def _signal_btcmacdcross(self, df: pd.DataFrame, benchmark_df: pd.DataFrame) -> str:
        adx, _, _, _, _, dm_delta, _ = _adx_components(df)
        fisher = _fisher_rsi(df["close"])
        macd_line, macd_signal, _ = _macd(df["close"])
        mfi = _mfi(df)
        sar = _parabolic_sar(df)
        bb = _bollinger_bands(_typical_price(df), window=20, stds=2.0)
        bb_gain = _room_to_upper(df["close"], bb["upper"])
        bb_drop = _room_to_lower(df["close"], bb["lower"])
        btc_macd, btc_signal, _ = _macd(benchmark_df["close"])

        long_entry = (
            _series_last(dm_delta, 0.0) > 0.0
            and float(df["close"].iloc[-1]) < _series_last(sar, float(df["close"].iloc[-1]))
            and _series_last(fisher, 0.0) < 0.18
            and _series_last(macd_line, 0.0) < 0.0
            and _series_last(bb_gain, 0.0) >= 0.04
            and _bool_last(_crossed_above(btc_macd.fillna(0.0), btc_signal.fillna(0.0)))
            and float(df["volume"].iloc[-1]) > 0.0
        )
        long_exit = _bool_last(_crossed_below(macd_line.fillna(0.0), macd_signal.fillna(0.0))) and _series_last(macd_line, 0.0) > 0.0
        short_entry = (
            _series_last(dm_delta, 0.0) < 0.0
            and float(df["close"].iloc[-1]) > _series_last(sar, float(df["close"].iloc[-1]))
            and _series_last(fisher, 0.0) > _mirror_threshold(0.18)
            and _series_last(macd_line, 0.0) > 0.0
            and _series_last(bb_drop, 0.0) >= 0.04
            and _bool_last(_crossed_below(btc_macd.fillna(0.0), btc_signal.fillna(0.0)))
            and float(df["volume"].iloc[-1]) > 0.0
        )
        short_exit = _bool_last(_crossed_above(macd_line.fillna(0.0), macd_signal.fillna(0.0))) and _series_last(macd_line, 0.0) < 0.0
        if long_entry:
            return "OPEN_LONG"
        if long_exit:
            return "CLOSE_LONG"
        if short_entry:
            return "OPEN_SHORT"
        if short_exit:
            return "CLOSE_SHORT"
        return "HOLD"

    def _signal_btcn_drop(self, df: pd.DataFrame, benchmark_df: pd.DataFrame) -> str:
        fisher = _fisher_rsi(df["close"])
        bands = _weighted_bollinger_bands(_typical_price(df), window=20, stds=2.0)
        n = 3
        btc_red = [float(benchmark_df["close"].iloc[-1 - i]) <= float(benchmark_df["open"].iloc[-1 - i]) for i in range(n)]
        btc_green = [float(benchmark_df["close"].iloc[-1 - i]) >= float(benchmark_df["open"].iloc[-1 - i]) for i in range(n)]
        btc_drop = (float(benchmark_df["open"].iloc[-n]) - float(benchmark_df["close"].iloc[-1])) / max(float(benchmark_df["open"].iloc[-n]), 1e-9)
        btc_rise = (float(benchmark_df["close"].iloc[-1]) - float(benchmark_df["open"].iloc[-n])) / max(float(benchmark_df["open"].iloc[-n]), 1e-9)
        long_entry = all(btc_red) and btc_drop >= 0.014
        short_entry = all(btc_green) and btc_rise >= 0.014
        if long_entry:
            return "OPEN_LONG"
        if short_entry:
            return "OPEN_SHORT"
        return "HOLD"

    def _signal_btcn_seq(self, df: pd.DataFrame, benchmark_df: pd.DataFrame) -> str:
        fisher = _fisher_rsi(df["close"])
        bands = _weighted_bollinger_bands(_typical_price(df), window=20, stds=2.0)
        bb_gain = _room_to_upper(df["close"], bands["upper"])
        bb_drop = _room_to_lower(df["close"], bands["lower"])
        n = 5
        prev_red = [float(benchmark_df["close"].iloc[-2 - i]) <= float(benchmark_df["open"].iloc[-2 - i]) for i in range(n)]
        prev_green = [float(benchmark_df["close"].iloc[-2 - i]) >= float(benchmark_df["open"].iloc[-2 - i]) for i in range(n)]
        btc_drop = (float(benchmark_df["open"].iloc[-n]) - float(benchmark_df["close"].iloc[-1])) / max(float(benchmark_df["open"].iloc[-n]), 1e-9)
        btc_rise = (float(benchmark_df["close"].iloc[-1]) - float(benchmark_df["open"].iloc[-n])) / max(float(benchmark_df["open"].iloc[-n]), 1e-9)
        long_entry = (
            _series_last(fisher, 0.0) <= -0.02
            and _series_last(bb_gain, 0.0) >= 0.07
            and float(benchmark_df["close"].iloc[-1]) > float(benchmark_df["open"].iloc[-1])
            and all(prev_red)
            and btc_drop >= 0.018
        )
        short_entry = (
            _series_last(fisher, 0.0) >= 0.02
            and _series_last(bb_drop, 0.0) >= 0.07
            and float(benchmark_df["close"].iloc[-1]) < float(benchmark_df["open"].iloc[-1])
            and all(prev_green)
            and btc_rise >= 0.018
        )
        if long_entry:
            return "OPEN_LONG"
        if short_entry:
            return "OPEN_SHORT"
        return "HOLD"

    def _signal_bigdrop(self, df: pd.DataFrame) -> str:
        fisher = _fisher_rsi(df["close"])
        mfi = _mfi(df)
        bands = _weighted_bollinger_bands(_typical_price(df), window=20, stds=2.0)
        n = 9
        long_entry = (
            _series_last(fisher, 0.0) < -0.23
            and (float(df["open"].iloc[-n]) - float(df["close"].iloc[-1])) / max(float(df["open"].iloc[-n]), 1e-9) >= 0.06
        )
        short_entry = (
            _series_last(fisher, 0.0) > 0.23
            and (float(df["close"].iloc[-1]) - float(df["open"].iloc[-n])) / max(float(df["open"].iloc[-n]), 1e-9) >= 0.06
        )
        if long_entry:
            return "OPEN_LONG"
        if short_entry:
            return "OPEN_SHORT"
        return "HOLD"

    def _compute_variant_signal(
        self,
        variant: str,
        entry_df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame],
    ) -> str:
        if len(entry_df) < 60:
            return "HOLD"
        if variant == "ADXDM":
            return self._signal_adxdm(entry_df)
        if variant == "BBBHold":
            return self._signal_bbbhold(entry_df)
        if variant == "BBKCBounce":
            return self._signal_bbkcbounce(entry_df)
        if variant == "BigDrop":
            return self._signal_bigdrop(entry_df)
        if benchmark_df is None or len(benchmark_df) < 60:
            return "HOLD"
        if variant == "BTCBigDrop":
            return self._signal_btcbigdrop(entry_df, benchmark_df)
        if variant == "BTCEMABounce":
            return self._signal_btcemabounce(entry_df, benchmark_df)
        if variant == "BTCJump":
            return self._signal_btcjump(entry_df, benchmark_df)
        if variant == "BTCMACDCross":
            return self._signal_btcmacdcross(entry_df, benchmark_df)
        if variant == "BTCNDrop":
            return self._signal_btcn_drop(entry_df, benchmark_df)
        if variant == "BTCNSeq":
            return self._signal_btcn_seq(entry_df, benchmark_df)
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

            benchmark_df = self._benchmark_df(symbol, data)
            raw_signal = self._compute_variant_signal(self.variant, entry_df.copy(), None if benchmark_df is None else benchmark_df.copy())
            market_regime = _regime(trend_df.copy(), cfg.trend_ma_period, cfg.trend_deadband_pct)

            if raw_signal == "OPEN_LONG" and market_regime != "uptrend":
                raw_signal = "HOLD"
            elif raw_signal == "OPEN_SHORT" and market_regime != "downtrend":
                raw_signal = "HOLD"

            position_key = (symbol, cfg.entry_timeframe)
            final_signal = self._apply_position_rules(position_key, raw_signal)
            result[symbol][cfg.entry_timeframe] = final_signal if final_signal in VALID_SIGNALS else "HOLD"

        return result


def create_direct_strategy(variant: str, symbols=None, timeframes=None, **config):
    symbols = list(symbols or ["BNBUSDT"])
    resolved_timeframes = list(timeframes or ["5m", "15m"])
    for required in ("5m", "15m"):
        if required not in resolved_timeframes:
            resolved_timeframes.append(required)
    benchmark_symbol = str(config.get("benchmark_symbol", "BTCUSDT"))
    if benchmark_symbol not in symbols:
        symbols.append(benchmark_symbol)
    return ImportedNateemmaDirectBatch1Strategy(
        variant=variant,
        symbols=symbols,
        timeframes=resolved_timeframes,
        config=config,
    )
