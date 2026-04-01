"""Rule-based ports of the first 10 nateemma Anomaly training-signal variants.

These are not full ML/anomaly-model ports. They recreate the public training
signal logic as lightweight, backtester-friendly strategies using this
project's signal schema:
OPEN_LONG, CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT, HOLD
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.strategies.indicators_library import (
    bollinger_bands,
    ema,
    highest,
    lowest,
    macd,
    rsi,
    williams_r,
)

VALID_SIGNALS = {"OPEN_LONG", "CLOSE_LONG", "OPEN_SHORT", "CLOSE_SHORT", "HOLD"}


def _money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    money_flow = typical_price * df["volume"]
    direction = typical_price.diff()
    positive = money_flow.where(direction > 0.0, 0.0).rolling(period).sum()
    negative = money_flow.where(direction < 0.0, 0.0).rolling(period).sum().abs()
    ratio = positive / negative.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + ratio))


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
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100.0 * plus_dm.rolling(period).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.rolling(period).mean() / atr.replace(0.0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)) * 100.0
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di


def _aroon(df: pd.DataFrame, period: int = 25):
    def _up(values: np.ndarray) -> float:
        return float(((period - 1 - np.argmax(values)) / float(period - 1)) * 100.0)

    def _down(values: np.ndarray) -> float:
        return float(((period - 1 - np.argmin(values)) / float(period - 1)) * 100.0)

    aroon_up = df["high"].rolling(period).apply(_up, raw=True)
    aroon_down = df["low"].rolling(period).apply(_down, raw=True)
    return aroon_up, aroon_down


def _fisher_wr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    wr = williams_r(df["high"], df["low"], df["close"], period=period)
    scaled = (wr + 50.0) / 50.0
    return scaled.clip(-1.0, 1.0)


def _local_peak(series: pd.Series) -> bool:
    if len(series) < 3:
        return False
    return bool(series.iloc[-2] > series.iloc[-3] and series.iloc[-2] > series.iloc[-1])


def _local_valley(series: pd.Series) -> bool:
    if len(series) < 3:
        return False
    return bool(series.iloc[-2] < series.iloc[-3] and series.iloc[-2] < series.iloc[-1])


def _regime(trend_df: pd.DataFrame, ma_period: int, deadband_pct: float) -> str:
    trend_line = ema(trend_df["close"], period=ma_period)
    if trend_line.empty or pd.isna(trend_line.iloc[-1]):
        return "sideways"
    last_close = float(trend_df["close"].iloc[-1])
    ma_value = float(trend_line.iloc[-1])
    if ma_value <= 0.0:
        return "sideways"
    distance_pct = ((last_close - ma_value) / ma_value) * 100.0
    if distance_pct >= deadband_pct:
        return "uptrend"
    if distance_pct <= -deadband_pct:
        return "downtrend"
    return "sideways"


@dataclass
class ImportedVariantConfig:
    name: str
    entry_timeframe: str = "5m"
    trend_timeframe: str = "15m"
    trend_ma_period: int = 50
    trend_deadband_pct: float = 0.15
    rsi_period: int = 14
    williams_period: int = 14
    bb_period: int = 20
    bb_std_dev: float = 2.0
    adx_period: int = 14
    aroon_period: int = 25
    dwt_proxy_period: int = 34
    lookback_period: int = 20
    jump_zscore: float = 2.0


class ImportedNateemmaBatch1Strategy(StrategyBase):
    def __init__(self, variant: str, symbols: List[str], timeframes: List[str], config: Dict):
        super().__init__(name=f"Strategy_Import_Nateemma_{variant}", symbols=symbols, timeframes=timeframes, config=config)
        self.variant = variant
        self.variant_config = ImportedVariantConfig(name=variant)
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

    def _compute_variant_signal(self, df: pd.DataFrame) -> str:
        cfg = self.variant_config
        if len(df) < 60:
            return "HOLD"

        fisher = _fisher_wr(df, period=cfg.williams_period)
        adx, plus_di, minus_di = _adx_components(df, period=cfg.adx_period)
        aroon_up, aroon_down = _aroon(df, period=cfg.aroon_period)
        upper_band, mid_band, lower_band = bollinger_bands(
            df["close"], period=cfg.bb_period, std_dev=cfg.bb_std_dev
        )
        bb_width = (upper_band - lower_band) / mid_band.replace(0.0, np.nan)
        mfi = _money_flow_index(df)
        macd_line, macd_signal, macd_hist = macd(df["close"])
        recent_low = lowest(df["close"], period=cfg.lookback_period)
        recent_high = highest(df["close"], period=cfg.lookback_period)
        ema_mid = ema(df["close"], period=cfg.dwt_proxy_period)
        dwt_proxy = df["close"] - ema_mid
        dwt_proxy_prev = dwt_proxy.shift(1)
        gain = df["close"].pct_change()
        rolling_gain = gain.rolling(cfg.lookback_period)
        gain_mean = rolling_gain.mean()
        gain_std = rolling_gain.std().replace(0.0, np.nan)

        last_fisher = float(fisher.iloc[-1]) if not pd.isna(fisher.iloc[-1]) else 0.0
        prev_fisher = float(fisher.iloc[-2]) if not pd.isna(fisher.iloc[-2]) else 0.0
        adx_last = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
        plus_di_last = float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0.0
        minus_di_last = float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0.0
        aroon_up_last = float(aroon_up.iloc[-1]) if not pd.isna(aroon_up.iloc[-1]) else 0.0
        aroon_down_last = float(aroon_down.iloc[-1]) if not pd.isna(aroon_down.iloc[-1]) else 0.0
        macd_hist_last = float(macd_hist.iloc[-1]) if not pd.isna(macd_hist.iloc[-1]) else 0.0
        macd_hist_prev = float(macd_hist.iloc[-2]) if not pd.isna(macd_hist.iloc[-2]) else 0.0
        mfi_last = float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50.0
        close_last = float(df["close"].iloc[-1])
        recent_low_last = float(recent_low.iloc[-1]) if not pd.isna(recent_low.iloc[-1]) else close_last
        recent_high_last = float(recent_high.iloc[-1]) if not pd.isna(recent_high.iloc[-1]) else close_last
        upper_last = float(upper_band.iloc[-1]) if not pd.isna(upper_band.iloc[-1]) else close_last
        lower_last = float(lower_band.iloc[-1]) if not pd.isna(lower_band.iloc[-1]) else close_last
        bb_gain = (upper_last - close_last) / close_last if close_last else 0.0
        bb_loss = (lower_last - close_last) / close_last if close_last else 0.0
        jump_down = bool(
            not pd.isna(gain.iloc[-2])
            and not pd.isna(gain_mean.iloc[-1])
            and not pd.isna(gain_std.iloc[-1])
            and gain.iloc[-2] <= gain_mean.iloc[-1] - cfg.jump_zscore * abs(gain_std.iloc[-1])
        )
        jump_up = bool(
            not pd.isna(gain.iloc[-2])
            and not pd.isna(gain_mean.iloc[-1])
            and not pd.isna(gain_std.iloc[-1])
            and gain.iloc[-2] >= gain_mean.iloc[-1] + cfg.jump_zscore * abs(gain_std.iloc[-1])
        )

        oversold = last_fisher <= -0.8
        overbought = last_fisher >= 0.8

        open_long = False
        close_long = False
        open_short = False
        close_short = False

        if self.variant == "Anomaly_adx":
            open_long = last_fisher < -0.5 and adx_last > 25.0 and minus_di_last > plus_di_last
            close_long = prev_fisher < -0.2 and last_fisher >= -0.2
            open_short = last_fisher > 0.5 and adx_last > 25.0 and plus_di_last > minus_di_last
            close_short = prev_fisher > 0.2 and last_fisher <= 0.2
        elif self.variant == "Anomaly_adx3":
            di_delta = plus_di_last - minus_di_last
            open_long = adx_last > 20.0 and di_delta <= -10.0 and last_fisher < -0.5
            close_long = prev_fisher < -0.2 and last_fisher >= -0.1
            open_short = adx_last > 20.0 and di_delta >= 10.0 and last_fisher > 0.5
            close_short = prev_fisher > 0.2 and last_fisher <= 0.1
        elif self.variant == "Anomaly_all":
            score_long = sum(
                [
                    last_fisher < -0.5,
                    macd_hist_prev < 0.0 and macd_hist_last >= 0.0,
                    close_last <= recent_low_last * 1.01,
                    _local_valley(bb_width.fillna(0.0)),
                ]
            )
            score_short = sum(
                [
                    last_fisher > 0.5,
                    macd_hist_prev > 0.0 and macd_hist_last <= 0.0,
                    close_last >= recent_high_last * 0.99,
                    _local_peak(bb_width.fillna(0.0)),
                ]
            )
            open_long = score_long >= 2
            close_long = prev_fisher < -0.2 and last_fisher >= -0.1
            open_short = score_short >= 2
            close_short = prev_fisher > 0.2 and last_fisher <= 0.1
        elif self.variant == "Anomaly_aroon":
            open_long = aroon_up_last > aroon_down_last and aroon_up_last > 90.0 and aroon_down_last < 10.0
            close_long = aroon_up_last < 60.0
            open_short = aroon_up_last < aroon_down_last and aroon_up_last < 10.0 and aroon_down_last > 90.0
            close_short = aroon_down_last < 60.0
        elif self.variant == "Anomaly_bbw":
            open_long = _local_peak(bb_width.fillna(0.0)) and last_fisher < -0.5
            close_long = prev_fisher < -0.2 and last_fisher >= -0.1
            open_short = _local_valley(bb_width.fillna(0.0)) and last_fisher > 0.5
            close_short = prev_fisher > 0.2 and last_fisher <= 0.1
        elif self.variant == "Anomaly_dwt":
            open_long = last_fisher < -0.5 and dwt_proxy_prev.iloc[-1] < 0.0 and dwt_proxy.iloc[-1] > dwt_proxy_prev.iloc[-1]
            close_long = dwt_proxy.iloc[-1] >= 0.0
            open_short = last_fisher > 0.5 and dwt_proxy_prev.iloc[-1] > 0.0 and dwt_proxy.iloc[-1] < dwt_proxy_prev.iloc[-1]
            close_short = dwt_proxy.iloc[-1] <= 0.0
        elif self.variant == "Anomaly_dwt2":
            dwt_proxy_series = ema_mid.fillna(df["close"])
            open_long = _local_valley(dwt_proxy_series) and last_fisher < -0.1
            close_long = _local_peak(dwt_proxy_series)
            open_short = _local_peak(dwt_proxy_series) and last_fisher > 0.1
            close_short = _local_valley(dwt_proxy_series)
        elif self.variant == "Anomaly_fbb":
            open_long = mfi_last < 50.0 and last_fisher < -0.8 and bb_gain >= 0.01
            close_long = prev_fisher < -0.2 and last_fisher >= -0.1
            open_short = mfi_last > 50.0 and last_fisher > 0.8 and bb_loss <= -0.01
            close_short = prev_fisher > 0.2 and last_fisher <= 0.1
        elif self.variant == "Anomaly_fwr":
            open_long = oversold
            close_long = prev_fisher < -0.2 and last_fisher >= -0.1
            open_short = overbought
            close_short = prev_fisher > 0.2 and last_fisher <= 0.1
        elif self.variant == "Anomaly_highlow":
            open_long = close_last <= recent_low_last * 1.002 and last_fisher < -0.5
            close_long = close_last >= ema_mid.iloc[-1] if not pd.isna(ema_mid.iloc[-1]) else False
            open_short = close_last >= recent_high_last * 0.998 and last_fisher > 0.5
            close_short = close_last <= ema_mid.iloc[-1] if not pd.isna(ema_mid.iloc[-1]) else False
        elif self.variant == "Anomaly_jump":
            open_long = jump_down and close_last <= recent_low_last * 1.02
            close_long = gain.iloc[-1] > 0.0 if not pd.isna(gain.iloc[-1]) else False
            open_short = jump_up and close_last >= recent_high_last * 0.98
            close_short = gain.iloc[-1] < 0.0 if not pd.isna(gain.iloc[-1]) else False
        elif self.variant == "Anomaly_macd":
            open_long = macd_hist_prev < 0.0 and macd_hist_last >= 0.0
            close_long = macd_hist_prev > 0.0 and macd_hist_last <= 0.0
            open_short = macd_hist_prev > 0.0 and macd_hist_last <= 0.0
            close_short = macd_hist_prev < 0.0 and macd_hist_last >= 0.0
        elif self.variant == "Anomaly_macd2":
            macd_hist_series = macd_hist.fillna(0.0)
            open_long = _local_valley(macd_hist_series) and macd_hist_last < 0.0
            close_long = _local_peak(macd_hist_series)
            open_short = _local_peak(macd_hist_series) and macd_hist_last > 0.0
            close_short = _local_valley(macd_hist_series)
        elif self.variant == "Anomaly_macd3":
            macd_hist_series = macd_hist.fillna(0.0)
            macd_neg_mean = float(macd_hist_series.clip(upper=0.0).rolling(30).mean().iloc[-1]) if len(macd_hist_series) >= 30 else 0.0
            macd_pos_mean = float(macd_hist_series.clip(lower=0.0).rolling(30).mean().iloc[-1]) if len(macd_hist_series) >= 30 else 0.0
            open_long = macd_hist_last < macd_neg_mean and mfi_last < 50.0
            close_long = macd_hist_last > 0.0
            open_short = macd_hist_last > macd_pos_mean and mfi_last > 50.0
            close_short = macd_hist_last < 0.0
        elif self.variant == "Anomaly_mfi":
            open_long = mfi_last <= 15.0
            close_long = mfi_last >= 60.0
            open_short = mfi_last >= 85.0
            close_short = mfi_last <= 40.0
        elif self.variant == "Anomaly_minmax":
            open_long = close_last <= recent_low_last * 1.001 and last_fisher < -0.5
            close_long = close_last >= recent_high_last * 0.995
            open_short = close_last >= recent_high_last * 0.999 and last_fisher > 0.5
            close_short = close_last <= recent_low_last * 1.005
        elif self.variant == "Anomaly_nseq":
            dn_seq = int((df["close"].diff().lt(0)).astype(int).groupby(df["close"].diff().ge(0).astype(int).cumsum()).cumsum().iloc[-1])
            up_seq = int((df["close"].diff().gt(0)).astype(int).groupby(df["close"].diff().le(0).astype(int).cumsum()).cumsum().iloc[-1])
            open_long = dn_seq >= 3 and last_fisher < -0.3
            close_long = up_seq >= 2
            open_short = up_seq >= 3 and last_fisher > 0.3
            close_short = dn_seq >= 2
        elif self.variant == "Anomaly_over":
            rsi_last = float(rsi(df["close"], period=cfg.rsi_period).iloc[-1])
            open_long = rsi_last < 40.0 and mfi_last < 40.0 and last_fisher < -0.4
            close_long = rsi_last > 55.0 or last_fisher > 0.2
            open_short = rsi_last > 60.0 and mfi_last > 60.0 and last_fisher > 0.6
            close_short = rsi_last < 45.0 or last_fisher < -0.2
        elif self.variant == "Anomaly_profit":
            open_long = last_fisher < -0.5 and bb_gain >= 0.02
            close_long = last_fisher > 0.2
            open_short = last_fisher > 0.5 and bb_loss <= -0.02
            close_short = last_fisher < -0.2
        elif self.variant == "Anomaly_pv":
            open_long = _local_valley(df["close"]) and close_last <= recent_low_last * 1.01
            close_long = _local_peak(df["close"])
            open_short = _local_peak(df["close"]) and close_last >= recent_high_last * 0.99
            close_short = _local_valley(df["close"])
        elif self.variant == "Anomaly_slope":
            slope_now = float(ema_mid.diff().iloc[-1]) if not pd.isna(ema_mid.diff().iloc[-1]) else 0.0
            slope_prev = float(ema_mid.diff().iloc[-3:].mean()) if len(ema_mid) >= 3 else slope_now
            open_long = slope_prev < 0.0 and slope_now > slope_prev and last_fisher < -0.5
            close_long = slope_now <= 0.0
            open_short = slope_prev > 0.0 and slope_now < slope_prev and last_fisher > 0.5
            close_short = slope_now >= 0.0
        elif self.variant == "Anomaly_smooth":
            smooth = ((df["high"] + df["low"]) / 2.0).ewm(span=7).mean()
            open_long = _local_valley(smooth) and last_fisher < -0.1
            close_long = _local_peak(smooth)
            open_short = _local_peak(smooth) and last_fisher > 0.1
            close_short = _local_valley(smooth)
        elif self.variant == "Anomaly_stochastic":
            stoch_k_base = ((df["close"] - lowest(df["low"], period=14)) / (highest(df["high"], period=14) - lowest(df["low"], period=14)).replace(0.0, np.nan)) * 100.0
            stoch_d = stoch_k_base.rolling(3).mean()
            fast_diff = stoch_k_base - stoch_d
            open_long = float(fast_diff.iloc[-1]) > 0.0 and float(fast_diff.iloc[-2]) <= 0.0
            close_long = float(fast_diff.iloc[-1]) < 0.0
            open_short = float(fast_diff.iloc[-1]) < 0.0 and float(fast_diff.iloc[-2]) >= 0.0
            close_short = float(fast_diff.iloc[-1]) > 0.0
        elif self.variant == "Anomaly_swing":
            open_long = _local_valley(ema_mid.fillna(df["close"])) and close_last <= recent_low_last * 1.01
            close_long = _local_peak(ema_mid.fillna(df["close"]))
            open_short = _local_peak(ema_mid.fillna(df["close"])) and close_last >= recent_high_last * 0.99
            close_short = _local_valley(ema_mid.fillna(df["close"]))

        if open_long:
            return "OPEN_LONG"
        if close_long:
            return "CLOSE_LONG"
        if open_short:
            return "OPEN_SHORT"
        if close_short:
            return "CLOSE_SHORT"
        return "HOLD"

    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        result: Dict[str, Dict[str, str]] = {}
        cfg = self.variant_config

        for symbol, tf_map in data.items():
            result[symbol] = {timeframe: "HOLD" for timeframe in tf_map}
            entry_df = tf_map.get(cfg.entry_timeframe)
            trend_df = tf_map.get(cfg.trend_timeframe)
            if entry_df is None or trend_df is None or entry_df.empty or trend_df.empty:
                continue

            raw_signal = self._compute_variant_signal(entry_df.copy())
            market_regime = _regime(
                trend_df.copy(),
                ma_period=cfg.trend_ma_period,
                deadband_pct=cfg.trend_deadband_pct,
            )

            if raw_signal == "OPEN_LONG" and market_regime != "uptrend":
                raw_signal = "HOLD"
            elif raw_signal == "OPEN_SHORT" and market_regime != "downtrend":
                raw_signal = "HOLD"

            position_key = (symbol, cfg.entry_timeframe)
            final_signal = self._apply_position_rules(position_key, raw_signal)
            result[symbol][cfg.entry_timeframe] = final_signal if final_signal in VALID_SIGNALS else "HOLD"

        return result


def create_imported_strategy(variant: str, symbols=None, timeframes=None, **config):
    symbols = list(symbols or ["BNBUSDT"])
    resolved_timeframes = list(timeframes or ["5m", "15m"])
    for required in ("5m", "15m"):
        if required not in resolved_timeframes:
            resolved_timeframes.append(required)
    return ImportedNateemmaBatch1Strategy(variant=variant, symbols=symbols, timeframes=resolved_timeframes, config=config)
