"""Generated strategy: Strategy_FAST_20260221_090208_01"""
import pandas as pd

from simple_strategy.strategies.strategy_builder import StrategyBuilder
from simple_strategy.strategies.indicators_library import atr, bollinger_bands, ema, rsi
from simple_strategy.strategies.signals_library import bollinger_bands_signals, rsi_mean_reversion_with_trend

STRATEGY_PARAMETERS = {
    "entry_timeframe": {
        "type": "str",
        "default": "1m",
        "description": "Primary timeframe used for entry and exit signals",
        "gui_hint": "Use minutes like 1m, 5m, 15m. Default is 1m.",
    },
    "use_15m_trend_filter": {
        "type": "bool",
        "default": True,
        "description": "Use higher-timeframe trend to block bad longs",
        "gui_hint": "If ON, long entries are filtered by trend timeframe EMA",
    },
    "trend_timeframe": {
        "type": "str",
        "default": "15m",
        "description": "Timeframe used as trend filter",
        "gui_hint": "Default 15m",
    },
    "force_close_bad_longs": {
        "type": "bool",
        "default": True,
        "description": "Force CLOSE_LONG when trend timeframe turns down",
        "gui_hint": "If OFF, only blocks new OPEN_LONG",
    },
    "use_atr_volatility_filter": {
        "type": "bool",
        "default": True,
        "description": "Trade only the most volatile symbols using ATR/price ranking",
        "gui_hint": "If ON, blocks entries outside the top volatility bucket",
    },
    "atr_volatility_top_pct": {
        "type": "float",
        "default": 0.5,
        "description": "Top volatility fraction to allow (0.3=top 30%, 0.5=top 50%)",
        "gui_hint": "Use 0.3 to 0.5",
    },
    "atr_volatility_period": {
        "type": "int",
        "default": 14,
        "description": "ATR period used for volatility ranking",
        "gui_hint": "Default 14",
    },
}


def _normalize_timeframe(timeframe):
    text = str(timeframe).strip().lower()
    if not text:
        return "1m"

    try:
        if text.endswith("m"):
            minutes = int(text[:-1])
        elif text.endswith("h"):
            minutes = int(text[:-1]) * 60
        elif text.endswith("d"):
            minutes = int(text[:-1]) * 1440
        else:
            minutes = int(text)
    except (TypeError, ValueError):
        return "1m"

    if minutes <= 0:
        return "1m"
    return f"{minutes}m"


def _to_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _to_positive_int(value, default=14, minimum=2):
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = int(default)
    return max(int(minimum), parsed)


def _normalize_top_fraction(value, default=0.5):
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        fraction = float(default)

    if fraction > 1.0:
        fraction = fraction / 100.0
    return min(1.0, max(0.05, fraction))


def _resolve_timeframe_key(mapping, preferred):
    if not isinstance(mapping, dict) or not mapping:
        return None
    target = _normalize_timeframe(preferred)
    for key in mapping.keys():
        if _normalize_timeframe(key) == target:
            return key
    return None


def _calculate_atr_pct(df, period):
    if df is None or df.empty:
        return None
    if any(col not in df.columns for col in ("high", "low", "close")):
        return None
    if len(df) < (period + 1):
        return None

    atr_series = atr(df["high"], df["low"], df["close"], period=period)
    if atr_series is None or atr_series.empty:
        return None

    atr_series = atr_series.dropna()
    if atr_series.empty:
        return None

    try:
        atr_value = float(atr_series.iloc[-1])
        close_value = float(df["close"].iloc[-1])
    except (TypeError, ValueError):
        return None

    if atr_value <= 0.0 or close_value <= 0.0:
        return None
    return atr_value / close_value


def _extract_latest_timestamp(df):
    if df is None or df.empty:
        return None

    value = None
    if "datetime" in df.columns:
        value = df["datetime"].iloc[-1]
    elif "timestamp" in df.columns:
        value = df["timestamp"].iloc[-1]
    elif len(df.index) > 0:
        value = df.index[-1]

    if value is None:
        return None

    try:
        parsed = pd.to_datetime(value, unit="ms", errors="coerce")
        if pd.isna(parsed):
            parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        if getattr(parsed, "tzinfo", None) is not None:
            parsed = parsed.tz_localize(None)
        return parsed.to_pydatetime()
    except Exception:
        return None


def _timeframe_to_seconds(timeframe):
    normalized = _normalize_timeframe(timeframe)
    try:
        minutes = int(normalized[:-1])
    except (TypeError, ValueError):
        minutes = 1
    return max(60, minutes * 60)


def _apply_15m_trend_filter(
    signals,
    data,
    entry_timeframe,
    trend_timeframe,
    force_close_bad_longs,
):
    if not isinstance(signals, dict):
        return signals

    for symbol, tf_signals in signals.items():
        if symbol not in data or not isinstance(tf_signals, dict):
            continue
        symbol_data = data.get(symbol, {})
        if not isinstance(symbol_data, dict):
            continue

        entry_key = _resolve_timeframe_key(tf_signals, entry_timeframe)
        trend_key = _resolve_timeframe_key(symbol_data, trend_timeframe)
        if entry_key is None or trend_key is None:
            continue

        trend_df = symbol_data.get(trend_key)
        if trend_df is None or trend_df.empty or len(trend_df) < 56:
            continue

        ema_fast_15 = ema(trend_df["close"], period=26)
        ema_slow_15 = ema(trend_df["close"], period=56)
        trend_is_up = float(ema_fast_15.iloc[-1]) >= float(ema_slow_15.iloc[-1])

        current_signal = tf_signals.get(entry_key, "HOLD")
        if trend_is_up:
            continue

        if current_signal == "OPEN_LONG":
            tf_signals[entry_key] = "HOLD"
        elif current_signal == "HOLD" and force_close_bad_longs:
            tf_signals[entry_key] = "CLOSE_LONG"

    return signals


def _apply_atr_volatility_filter(
    signals,
    data,
    entry_timeframe,
    atr_period,
    top_fraction,
    volatility_cache,
):
    if not isinstance(signals, dict) or not isinstance(data, dict):
        return signals

    refreshed_timestamps = []
    for symbol, symbol_data in data.items():
        if not isinstance(symbol_data, dict):
            continue
        tf_key = _resolve_timeframe_key(symbol_data, entry_timeframe)
        if tf_key is None:
            continue
        tf_df = symbol_data.get(tf_key)
        atr_pct = _calculate_atr_pct(tf_df, period=atr_period)
        if atr_pct is not None:
            latest_ts = _extract_latest_timestamp(tf_df)
            refreshed_timestamps.append(latest_ts)
            volatility_cache[symbol] = {
                "atr_pct": float(atr_pct),
                "timestamp": latest_ts,
            }

    reference_ts = next((ts for ts in reversed(refreshed_timestamps) if ts is not None), None)
    if reference_ts is not None and volatility_cache:
        max_age_seconds = _timeframe_to_seconds(entry_timeframe) * 3
        for cached_symbol, cached_data in list(volatility_cache.items()):
            if not isinstance(cached_data, dict):
                volatility_cache.pop(cached_symbol, None)
                continue
            cached_ts = cached_data.get("timestamp")
            if cached_ts is None:
                continue
            try:
                age_seconds = abs((reference_ts - cached_ts).total_seconds())
            except Exception:
                volatility_cache.pop(cached_symbol, None)
                continue
            if age_seconds > max_age_seconds:
                volatility_cache.pop(cached_symbol, None)

    if len(volatility_cache) < 3:
        return signals

    sorted_vols = sorted(
        float(item["atr_pct"])
        for item in volatility_cache.values()
        if isinstance(item, dict) and item.get("atr_pct") is not None
    )
    if not sorted_vols:
        return signals

    keep_count = max(1, int(len(sorted_vols) * top_fraction))
    cutoff_index = max(0, len(sorted_vols) - keep_count)
    cutoff = sorted_vols[cutoff_index]

    for symbol, tf_signals in signals.items():
        if not isinstance(tf_signals, dict):
            continue
        entry_key = _resolve_timeframe_key(tf_signals, entry_timeframe)
        if entry_key is None:
            continue
        symbol_cache = volatility_cache.get(symbol, {})
        if not isinstance(symbol_cache, dict):
            continue
        symbol_vol = symbol_cache.get("atr_pct")
        if symbol_vol is None or float(symbol_vol) >= cutoff:
            continue

        current_signal = tf_signals.get(entry_key, "HOLD")
        if current_signal in ("OPEN_LONG", "OPEN_SHORT"):
            tf_signals[entry_key] = "HOLD"

    return signals


def create_strategy(symbols=None, timeframes=None, **params):
    symbols = symbols or ["BNBUSDT", "XRPUSDT", "ADAUSDT", "DOTUSDT", "NEARUSDT"]

    entry_timeframe = _normalize_timeframe(params.get("entry_timeframe", "1m"))

    requested_timeframes = []
    if isinstance(timeframes, (list, tuple)):
        requested_timeframes = [
            _normalize_timeframe(tf)
            for tf in timeframes
            if str(tf).strip()
        ]

    if not requested_timeframes:
        requested_timeframes = [entry_timeframe]

    if entry_timeframe not in requested_timeframes:
        requested_timeframes.append(entry_timeframe)

    use_15m_trend_filter = _to_bool(params.get("use_15m_trend_filter", True), default=True)
    trend_timeframe = _normalize_timeframe(params.get("trend_timeframe", "15m"))
    force_close_bad_longs = _to_bool(params.get("force_close_bad_longs", True), default=True)
    use_atr_volatility_filter = _to_bool(params.get("use_atr_volatility_filter", True), default=True)
    atr_volatility_top_pct = _normalize_top_fraction(params.get("atr_volatility_top_pct", 0.5), default=0.5)
    atr_volatility_period = _to_positive_int(params.get("atr_volatility_period", 14), default=14, minimum=2)

    if use_15m_trend_filter and trend_timeframe not in requested_timeframes:
        requested_timeframes.append(trend_timeframe)

    builder = StrategyBuilder(symbols=symbols, timeframes=requested_timeframes)
    builder.add_indicator("bb_main", bollinger_bands, period=28, std_dev=2.3)
    builder.add_indicator("ema_fast", ema, period=26)
    builder.add_indicator("ema_slow", ema, period=56)
    builder.add_indicator("rsi_main", rsi, period=14)
    builder.add_signal_rule(
        "rsi_trend_combo",
        rsi_mean_reversion_with_trend,
        rsi="rsi_main",
        ema_fast="ema_fast",
        ema_slow="ema_slow",
        overbought=84,
        oversold=33,
    )
    builder.add_signal_rule(
        "bb_reversion",
        bollinger_bands_signals,
        price="price",
        upper_band="upper_band",
        lower_band="lower_band",
    )
    builder.set_signal_combination("and_signals")
    builder.set_strategy_info("Strategy_FAST_20260221_090208_01", "1.0.0")
    strategy = builder.build()
    strategy.entry_timeframe = entry_timeframe
    strategy.use_15m_trend_filter = use_15m_trend_filter
    strategy.trend_timeframe = trend_timeframe
    strategy.force_close_bad_longs = force_close_bad_longs
    strategy.use_atr_volatility_filter = use_atr_volatility_filter
    strategy.atr_volatility_top_pct = atr_volatility_top_pct
    strategy.atr_volatility_period = atr_volatility_period
    strategy.params = dict(params or {})
    strategy.params["entry_timeframe"] = entry_timeframe
    strategy.params["use_15m_trend_filter"] = use_15m_trend_filter
    strategy.params["trend_timeframe"] = trend_timeframe
    strategy.params["force_close_bad_longs"] = force_close_bad_longs
    strategy.params["use_atr_volatility_filter"] = use_atr_volatility_filter
    strategy.params["atr_volatility_top_pct"] = atr_volatility_top_pct
    strategy.params["atr_volatility_period"] = atr_volatility_period

    if (use_15m_trend_filter or use_atr_volatility_filter) and hasattr(strategy, "generate_signals"):
        original_generate_signals = strategy.generate_signals
        volatility_cache = {}

        def _generate_signals_with_trend_filter(data):
            base_signals = original_generate_signals(data)
            filtered_signals = base_signals
            if use_15m_trend_filter:
                filtered_signals = _apply_15m_trend_filter(
                    filtered_signals,
                    data,
                    entry_timeframe=entry_timeframe,
                    trend_timeframe=trend_timeframe,
                    force_close_bad_longs=force_close_bad_longs,
                )
            if use_atr_volatility_filter:
                filtered_signals = _apply_atr_volatility_filter(
                    filtered_signals,
                    data,
                    entry_timeframe=entry_timeframe,
                    atr_period=atr_volatility_period,
                    top_fraction=atr_volatility_top_pct,
                    volatility_cache=volatility_cache,
                )
            return filtered_signals

        strategy.generate_signals = _generate_signals_with_trend_filter
        strategy.atr_volatility_cache = volatility_cache
        # Backtester should use strategy.generate_signals for this strategy,
        # otherwise builder internal shortcuts bypass cross-timeframe filters.
        strategy.force_strategy_generate_signals = True
    return strategy
