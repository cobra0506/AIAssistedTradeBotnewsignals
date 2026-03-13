import importlib
import logging
from typing import Any, Dict, List

import pandas as pd

from rl_ai.shared.feature_builder import AccountSnapshot, FeatureBuilder
from rl_ai.shared.feature_schema import FeatureSchema
from rl_ai.shared.signal_adapter import SignalAdapter
from simple_strategy.strategies.indicators_library import (
    atr,
    bollinger_bands,
    ema,
    highest,
    lowest,
    rsi,
    sma,
    stochastic,
    volume_sma,
)
from simple_strategy.strategies.signals_library import (
    breakout_signals,
    bollinger_bands_signals,
    ma_crossover,
    overbought_oversold,
    overbought_oversold_crossover,
    rsi_mean_reversion_with_trend,
    stochastic_signals,
)
from simple_strategy.strategies.strategy_builder import StrategyBuilder

logger = logging.getLogger(__name__)

VALID_SIGNALS = {"OPEN_LONG", "CLOSE_LONG", "OPEN_SHORT", "CLOSE_SHORT", "HOLD"}


def _ensure_list(value: List[str], default: List[str]) -> List[str]:
    cleaned = [str(item).strip() for item in (value or []) if str(item).strip()]
    return cleaned or list(default)


def _ensure_timeframes(
    timeframes: List[str],
    entry_timeframe: str = "",
    trend_timeframe: str = "",
    default: List[str] = None,
) -> List[str]:
    resolved = _ensure_list(timeframes, default or ["1m"])
    for timeframe in (entry_timeframe, trend_timeframe):
        if timeframe and timeframe not in resolved:
            resolved.append(timeframe)
    return resolved


def _normalize_timeframe(tf: str) -> str:
    return str(tf).rstrip("m")


def _parse_indicator_columns(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _clamp_ratio(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    return max(0.0, min(1.0, parsed))


def price_direction_signal(price: pd.Series) -> pd.Series:
    signals = pd.Series("HOLD", index=price.index)
    rising = price > price.shift(1)
    falling = price < price.shift(1)
    signals[rising] = "OPEN_LONG"
    signals[falling] = "CLOSE_LONG"
    return signals


def entry_timeframe_filter(
    signals: Dict[str, Dict[str, str]],
    data: Dict[str, Dict[str, pd.DataFrame]],
    *,
    entry_timeframe: str,
    **_: Any,
) -> Dict[str, Dict[str, str]]:
    filtered = {}
    for symbol, tf_map in data.items():
        filtered[symbol] = {}
        for timeframe in tf_map:
            filtered[symbol][timeframe] = signals.get(symbol, {}).get(timeframe, "HOLD")
            if timeframe != entry_timeframe:
                filtered[symbol][timeframe] = "HOLD"
    return filtered


def long_only_filter(
    signals: Dict[str, Dict[str, str]],
    data: Dict[str, Dict[str, pd.DataFrame]],
    **_: Any,
) -> Dict[str, Dict[str, str]]:
    filtered = {symbol: dict(tf_map) for symbol, tf_map in signals.items()}
    for symbol, tf_map in filtered.items():
        for timeframe, signal in tf_map.items():
            if signal in {"OPEN_SHORT", "CLOSE_SHORT"}:
                filtered[symbol][timeframe] = "HOLD"
    return filtered


def trend_timeframe_filter(
    signals: Dict[str, Dict[str, str]],
    data: Dict[str, Dict[str, pd.DataFrame]],
    *,
    entry_timeframe: str,
    trend_timeframe: str,
    ma_period: int,
    ma_kind: str = "ema",
    **_: Any,
) -> Dict[str, Dict[str, str]]:
    ma_func = sma if str(ma_kind).lower() == "sma" else ema
    filtered = {symbol: dict(tf_map) for symbol, tf_map in signals.items()}

    for symbol, tf_map in data.items():
        trend_df = tf_map.get(trend_timeframe)
        if trend_df is None or len(trend_df) < max(2, ma_period):
            continue

        trend_ma = ma_func(trend_df["close"], period=ma_period)
        if trend_ma.empty or pd.isna(trend_ma.iloc[-1]):
            continue

        bullish = bool(trend_df["close"].iloc[-1] > trend_ma.iloc[-1])
        current_signal = filtered.get(symbol, {}).get(entry_timeframe, "HOLD")

        if current_signal == "OPEN_LONG" and not bullish:
            filtered[symbol][entry_timeframe] = "HOLD"
        elif current_signal == "OPEN_SHORT" and bullish:
            filtered[symbol][entry_timeframe] = "HOLD"

    return filtered


def rsi_gate_filter(
    signals: Dict[str, Dict[str, str]],
    data: Dict[str, Dict[str, pd.DataFrame]],
    *,
    entry_timeframe: str,
    rsi_period: int,
    bullish_threshold: float,
    bearish_threshold: float,
    **_: Any,
) -> Dict[str, Dict[str, str]]:
    filtered = {symbol: dict(tf_map) for symbol, tf_map in signals.items()}

    for symbol, tf_map in data.items():
        df = tf_map.get(entry_timeframe)
        if df is None or len(df) < max(2, rsi_period):
            continue

        rsi_series = rsi(df["close"], period=rsi_period)
        if rsi_series.empty or pd.isna(rsi_series.iloc[-1]):
            continue

        current_rsi = float(rsi_series.iloc[-1])
        current_signal = filtered.get(symbol, {}).get(entry_timeframe, "HOLD")

        if current_signal == "OPEN_LONG" and current_rsi <= bullish_threshold:
            filtered[symbol][entry_timeframe] = "HOLD"
        elif current_signal == "OPEN_SHORT" and current_rsi >= bearish_threshold:
            filtered[symbol][entry_timeframe] = "HOLD"

    return filtered


def volume_filter(
    signals: Dict[str, Dict[str, str]],
    data: Dict[str, Dict[str, pd.DataFrame]],
    *,
    entry_timeframe: str,
    volume_sma_period: int,
    volume_multiplier: float,
    **_: Any,
) -> Dict[str, Dict[str, str]]:
    filtered = {symbol: dict(tf_map) for symbol, tf_map in signals.items()}

    for symbol, tf_map in data.items():
        df = tf_map.get(entry_timeframe)
        if df is None or len(df) < max(2, volume_sma_period):
            continue

        volume_avg = volume_sma(df["volume"], period=volume_sma_period)
        if volume_avg.empty or pd.isna(volume_avg.iloc[-1]):
            continue

        confirmed = bool(df["volume"].iloc[-1] > volume_avg.iloc[-1] * volume_multiplier)
        current_signal = filtered.get(symbol, {}).get(entry_timeframe, "HOLD")
        if current_signal in {"OPEN_LONG", "OPEN_SHORT"} and not confirmed:
            filtered[symbol][entry_timeframe] = "HOLD"

    return filtered


def trend_and_volume_filter(
    signals: Dict[str, Dict[str, str]],
    data: Dict[str, Dict[str, pd.DataFrame]],
    *,
    entry_timeframe: str,
    trend_sma_period: int,
    volume_sma_period: int,
    volume_multiplier: float,
    **_: Any,
) -> Dict[str, Dict[str, str]]:
    filtered = {symbol: dict(tf_map) for symbol, tf_map in signals.items()}

    for symbol, tf_map in data.items():
        df = tf_map.get(entry_timeframe)
        if df is None or len(df) < max(trend_sma_period, volume_sma_period, 2):
            continue

        trend_line = sma(df["close"], period=trend_sma_period)
        volume_avg = volume_sma(df["volume"], period=volume_sma_period)
        if trend_line.empty or volume_avg.empty:
            continue
        if pd.isna(trend_line.iloc[-1]) or pd.isna(volume_avg.iloc[-1]):
            continue

        close_price = float(df["close"].iloc[-1])
        volume_ok = bool(df["volume"].iloc[-1] > volume_avg.iloc[-1] * volume_multiplier)
        current_signal = filtered.get(symbol, {}).get(entry_timeframe, "HOLD")

        if current_signal == "OPEN_LONG" and (close_price <= trend_line.iloc[-1] or not volume_ok):
            filtered[symbol][entry_timeframe] = "HOLD"
        elif current_signal == "OPEN_SHORT" and (close_price >= trend_line.iloc[-1] or not volume_ok):
            filtered[symbol][entry_timeframe] = "HOLD"

    return filtered


def atr_threshold_filter(
    signals: Dict[str, Dict[str, str]],
    data: Dict[str, Dict[str, pd.DataFrame]],
    *,
    entry_timeframe: str,
    atr_period: int,
    min_atr_threshold: float,
    threshold_is_percent: bool = True,
    **_: Any,
) -> Dict[str, Dict[str, str]]:
    filtered = {symbol: dict(tf_map) for symbol, tf_map in signals.items()}

    for symbol, tf_map in data.items():
        df = tf_map.get(entry_timeframe)
        if df is None or len(df) < max(atr_period, 2):
            continue

        atr_series = atr(df["high"], df["low"], df["close"], period=atr_period)
        if atr_series.empty or pd.isna(atr_series.iloc[-1]):
            continue

        value = float(atr_series.iloc[-1])
        if threshold_is_percent and float(df["close"].iloc[-1]) > 0:
            value = (value / float(df["close"].iloc[-1])) * 100.0

        current_signal = filtered.get(symbol, {}).get(entry_timeframe, "HOLD")
        if current_signal in {"OPEN_LONG", "OPEN_SHORT"} and value < float(min_atr_threshold):
            filtered[symbol][entry_timeframe] = "HOLD"

    return filtered


def price_change_bias_filter(
    signals: Dict[str, Dict[str, str]],
    data: Dict[str, Dict[str, pd.DataFrame]],
    *,
    entry_timeframe: str,
    price_change_threshold: float,
    **_: Any,
) -> Dict[str, Dict[str, str]]:
    filtered = {symbol: dict(tf_map) for symbol, tf_map in signals.items()}

    for symbol, tf_map in data.items():
        df = tf_map.get(entry_timeframe)
        if df is None or len(df) < 2:
            continue

        prev_close = float(df["close"].iloc[-2])
        last_close = float(df["close"].iloc[-1])
        pct_change = 0.0 if prev_close == 0 else (last_close - prev_close) / prev_close
        current_signal = filtered.get(symbol, {}).get(entry_timeframe, "HOLD")

        if current_signal == "OPEN_LONG" and pct_change <= -abs(price_change_threshold):
            filtered[symbol][entry_timeframe] = "HOLD"
        elif current_signal == "OPEN_SHORT" and pct_change >= abs(price_change_threshold):
            filtered[symbol][entry_timeframe] = "HOLD"
        elif current_signal == "HOLD":
            if pct_change > abs(price_change_threshold):
                filtered[symbol][entry_timeframe] = "OPEN_LONG"
            elif pct_change < -abs(price_change_threshold):
                filtered[symbol][entry_timeframe] = "OPEN_SHORT"

    return filtered


def toggle_test_filter(
    signals: Dict[str, Dict[str, str]],
    data: Dict[str, Dict[str, pd.DataFrame]],
    *,
    strategy=None,
    test_mode: bool = True,
    entry_timeframe: str = "",
    **_: Any,
) -> Dict[str, Dict[str, str]]:
    if strategy is None or not test_mode:
        return signals

    toggle_state = getattr(strategy, "_toggle_state", {})
    filtered = {symbol: dict(tf_map) for symbol, tf_map in signals.items()}

    for symbol, tf_map in data.items():
        for timeframe, df in tf_map.items():
            if entry_timeframe and timeframe != entry_timeframe:
                filtered.setdefault(symbol, {})[timeframe] = "HOLD"
                continue
            if df is None or len(df) < 2:
                filtered.setdefault(symbol, {})[timeframe] = "HOLD"
                continue

            key = (symbol, timeframe)
            toggle = bool(toggle_state.get(key, False))
            position = strategy._position_state.get(key)
            if position is None:
                signal = "OPEN_LONG"
            elif not position.get("is_short", False):
                signal = "CLOSE_LONG" if not toggle else "OPEN_SHORT"
            else:
                signal = "CLOSE_SHORT" if not toggle else "OPEN_LONG"
            toggle_state[key] = not toggle
            filtered.setdefault(symbol, {})[timeframe] = signal

    strategy._toggle_state = toggle_state
    return filtered


def rl_signal_filter(
    signals: Dict[str, Dict[str, str]],
    data: Dict[str, Dict[str, pd.DataFrame]],
    *,
    strategy=None,
    entry_timeframe: str,
    rl_mode: str,
    model_path: str,
    deterministic: bool,
    confirm_strategy_name: str,
    indicator_columns: Any,
    min_bars: int,
    initial_virtual_balance: float,
    position_size_pct: float,
    max_position_size_pct: float,
    max_drawdown_pct: float,
    fee_pct: float,
    **_: Any,
) -> Dict[str, Dict[str, str]]:
    if strategy is None:
        return signals

    if not hasattr(strategy, "_rl_state"):
        state: Dict[str, Any] = {
            "policy_model": None,
            "confirm_strategy": None,
            "feature_builder": FeatureBuilder(
                schema=FeatureSchema(indicator_features=_parse_indicator_columns(indicator_columns))
            ),
            "virtual_balance": max(100.0, float(initial_virtual_balance)),
            "virtual_peak_equity": max(100.0, float(initial_virtual_balance)),
            "virtual_drawdown_pct": 0.0,
            "drawdown_lock": False,
            "effective_position_size_pct": min(
                _clamp_ratio(position_size_pct, 0.05),
                _clamp_ratio(max_position_size_pct, 0.10),
            ),
            "fee_pct": max(0.0, float(fee_pct)),
            "initial_virtual_balance": max(100.0, float(initial_virtual_balance)),
            "max_drawdown_pct": _clamp_ratio(max_drawdown_pct, 0.15),
            "latest_price_by_position_key": {},
        }
        try:
            model_io_module = importlib.import_module("rl_ai.01_library_based.model_io")
            state["policy_model"], _ = model_io_module.load_policy_model(str(model_path))
        except Exception as exc:
            logger.warning("RL preset could not load model %s: %s", model_path, exc)

        if rl_mode == "rl_confirm" and confirm_strategy_name:
            for module_name in (
                f"simple_strategy.strategies.{confirm_strategy_name}",
                f"strategies.{confirm_strategy_name}",
            ):
                try:
                    module = importlib.import_module(module_name)
                    if hasattr(module, "create_strategy"):
                        state["confirm_strategy"] = module.create_strategy(
                            symbols=strategy.symbols,
                            timeframes=strategy.timeframes,
                        )
                        break
                except Exception:
                    continue
        strategy._rl_state = state

    state = strategy._rl_state
    filtered = {symbol: dict(tf_map) for symbol, tf_map in signals.items()}

    def _estimate_unrealized(position: Dict[str, Any], current_price: float) -> float:
        entry_price = float(position.get("entry_price", 0.0) or 0.0)
        notional = float(position.get("notional", 0.0) or 0.0)
        if entry_price <= 0.0 or current_price <= 0.0 or notional <= 0.0:
            return 0.0
        if bool(position.get("is_short", False)):
            return notional * ((entry_price - current_price) / entry_price)
        return notional * ((current_price - entry_price) / entry_price)

    def _refresh_risk_state() -> None:
        equity = float(state["virtual_balance"])
        for position_key, position in strategy._position_state.items():
            current_price = float(state["latest_price_by_position_key"].get(position_key, 0.0) or 0.0)
            equity += _estimate_unrealized(position, current_price)
        if equity > state["virtual_peak_equity"]:
            state["virtual_peak_equity"] = float(equity)
        peak = float(state["virtual_peak_equity"])
        state["virtual_drawdown_pct"] = 0.0 if peak <= 0 else max(0.0, (peak - equity) / peak)
        if state["virtual_drawdown_pct"] >= state["max_drawdown_pct"]:
            state["drawdown_lock"] = True

    def _apply_position_rules(position_key: tuple, raw_signal: str, current_price: float) -> str:
        position = strategy._position_state.get(position_key)
        if raw_signal == "OPEN_LONG":
            if position is not None or state["drawdown_lock"] or current_price <= 0.0:
                return "HOLD"
            notional = float(state["virtual_balance"]) * float(state["effective_position_size_pct"])
            fee = notional * float(state["fee_pct"])
            if notional <= 0.0 or fee >= float(state["virtual_balance"]):
                return "HOLD"
            state["virtual_balance"] -= fee
            strategy._position_state[position_key] = {"is_short": False, "entry_price": current_price, "notional": notional}
            _refresh_risk_state()
            return "OPEN_LONG"
        if raw_signal == "OPEN_SHORT":
            if position is not None or state["drawdown_lock"] or current_price <= 0.0:
                return "HOLD"
            notional = float(state["virtual_balance"]) * float(state["effective_position_size_pct"])
            fee = notional * float(state["fee_pct"])
            if notional <= 0.0 or fee >= float(state["virtual_balance"]):
                return "HOLD"
            state["virtual_balance"] -= fee
            strategy._position_state[position_key] = {"is_short": True, "entry_price": current_price, "notional": notional}
            _refresh_risk_state()
            return "OPEN_SHORT"
        if raw_signal == "CLOSE_LONG":
            if position is None or position.get("is_short", False):
                return "HOLD"
            pnl = _estimate_unrealized(position, current_price)
            state["virtual_balance"] += pnl - (max(0.0, float(position.get("notional", 0.0)) + pnl) * float(state["fee_pct"]))
            strategy._position_state.pop(position_key, None)
            _refresh_risk_state()
            return "CLOSE_LONG"
        if raw_signal == "CLOSE_SHORT":
            if position is None or not position.get("is_short", False):
                return "HOLD"
            pnl = _estimate_unrealized(position, current_price)
            state["virtual_balance"] += pnl - (max(0.0, float(position.get("notional", 0.0)) + pnl) * float(state["fee_pct"]))
            strategy._position_state.pop(position_key, None)
            _refresh_risk_state()
            return "CLOSE_SHORT"
        return "HOLD"

    for symbol, tf_map in data.items():
        for timeframe, df in tf_map.items():
            filtered.setdefault(symbol, {})[timeframe] = "HOLD"
            if timeframe != entry_timeframe or df is None or len(df) < max(10, int(min_bars)):
                continue
            position_key = (symbol, timeframe)
            current_price = float(df["close"].iloc[-1])
            state["latest_price_by_position_key"][position_key] = current_price
            _refresh_risk_state()
            if state["drawdown_lock"]:
                position = strategy._position_state.get(position_key)
                raw_signal = "HOLD" if position is None else ("CLOSE_SHORT" if position.get("is_short", False) else "CLOSE_LONG")
                filtered[symbol][timeframe] = _apply_position_rules(position_key, raw_signal, current_price)
                continue
            model = state.get("policy_model")
            raw_signal = "HOLD"
            if model is not None:
                try:
                    account = AccountSnapshot(
                        balance=float(state["virtual_balance"]),
                        initial_balance=float(state["initial_virtual_balance"]),
                        has_position=position_key in strategy._position_state,
                        is_short=bool(strategy._position_state.get(position_key, {}).get("is_short", False)),
                        entry_price=strategy._position_state.get(position_key, {}).get("entry_price"),
                        current_price=current_price,
                        drawdown_pct=float(state["virtual_drawdown_pct"]),
                    )
                    obs = state["feature_builder"].build_vector(df, row_index=len(df) - 1, account=account)
                    if int(getattr(model, "observation_size", -1)) == int(len(obs)):
                        action = int(model.greedy_action(obs) if deterministic else model.sample_action(obs).action)
                        raw_signal = SignalAdapter.action_to_signal(
                            action,
                            has_position=position_key in strategy._position_state,
                            is_short=bool(strategy._position_state.get(position_key, {}).get("is_short", False)),
                        )
                except Exception:
                    raw_signal = "HOLD"
            if rl_mode == "rl_confirm" and state.get("confirm_strategy") is not None:
                try:
                    confirm_signal = state["confirm_strategy"].generate_signals(data).get(symbol, {}).get(timeframe, "HOLD")
                except Exception:
                    confirm_signal = "HOLD"
                if raw_signal != confirm_signal:
                    raw_signal = "HOLD"
            filtered[symbol][timeframe] = _apply_position_rules(position_key, raw_signal, current_price)

    return filtered


def _finalize_builder(builder: StrategyBuilder, strategy_name: str, version: str = "1.0.0"):
    builder.set_strategy_info(strategy_name, version)
    strategy = builder.build()
    strategy.force_strategy_generate_signals = True
    return strategy


def build_strategy_preset(strategy_name: str, symbols=None, timeframes=None, **params):
    symbols = _ensure_list(symbols, ["BTCUSDT"])

    if strategy_name == "Strategy_1_Trend_Following":
        ma_type = str(params.get("ma_type", "ema")).lower()
        ma_func = sma if ma_type == "sma" else ema
        timeframes = _ensure_timeframes(timeframes, default=["5m"])
        builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
        builder.add_indicator("fast_ma", ma_func, period=params.get("fast_period", 12))
        builder.add_indicator("slow_ma", ma_func, period=params.get("slow_period", 26))
        builder.add_signal_rule("ma_crossover", ma_crossover, fast_ma="fast_ma", slow_ma="slow_ma")
        return _finalize_builder(builder, strategy_name)

    if strategy_name in {
        "Strategy_2_mean_reversion",
        "Strategy_Mean_Reversion",
        "Strategy_RSI_EMA_Trend_Filter",
        "Strategy_Simple_EMA_RSI_Scalping",
    }:
        entry_timeframe = str(params.get("entry_timeframe", (timeframes or ["5m"])[0]))
        timeframes = _ensure_timeframes(timeframes, entry_timeframe=entry_timeframe, default=["5m"])
        builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
        builder.add_indicator("rsi_main", rsi, period=params.get("rsi_period", 14))
        builder.add_indicator("ema_fast", ema, period=params.get("fast_ema_period", params.get("trend_fast_ema", 20)))
        builder.add_indicator("ema_slow", ema, period=params.get("slow_ema_period", params.get("trend_slow_ema", 50)))
        builder.add_signal_rule(
            "rsi_trend_combo",
            rsi_mean_reversion_with_trend,
            rsi="rsi_main",
            ema_fast="ema_fast",
            ema_slow="ema_slow",
            overbought=params.get("rsi_overbought", 70),
            oversold=params.get("rsi_oversold", 30),
        )
        builder.add_filter_rule("entry_timeframe", entry_timeframe_filter, entry_timeframe=entry_timeframe)
        if not bool(params.get("bidirectional", True)):
            builder.add_filter_rule("long_only", long_only_filter)
        return _finalize_builder(builder, strategy_name)

    if strategy_name in {"Strategy_NEW__MAcrossover_RSIconfirm", "Strategy_Simple_MA_Crossover"}:
        entry_timeframe = str(params.get("entry_timeframe", "1m"))
        trend_timeframe = str(params.get("trend_timeframe", "5m"))
        timeframes = _ensure_timeframes(
            timeframes,
            entry_timeframe=entry_timeframe,
            trend_timeframe=trend_timeframe,
            default=["1m", "5m"],
        )
        builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
        builder.add_indicator("fast_ma", ema, period=params.get("fast_ma_period", params.get("ma_fast_period", 5)))
        builder.add_indicator("slow_ma", ema, period=params.get("slow_ma_period", params.get("ma_slow_period", 15)))
        builder.add_indicator("rsi_main", rsi, period=params.get("rsi_period", 14))
        builder.add_signal_rule("ma_crossover", ma_crossover, fast_ma="fast_ma", slow_ma="slow_ma")
        builder.add_filter_rule("entry_timeframe", entry_timeframe_filter, entry_timeframe=entry_timeframe)
        builder.add_filter_rule(
            "rsi_gate",
            rsi_gate_filter,
            entry_timeframe=entry_timeframe,
            rsi_period=params.get("rsi_period", 14),
            bullish_threshold=params.get("rsi_bullish_threshold", params.get("rsi_buy_threshold", 55)),
            bearish_threshold=params.get("rsi_bearish_threshold", params.get("rsi_sell_threshold", 45)),
        )
        if strategy_name == "Strategy_Simple_MA_Crossover":
            builder.add_filter_rule(
                "trend_tf",
                trend_timeframe_filter,
                entry_timeframe=entry_timeframe,
                trend_timeframe=trend_timeframe,
                ma_period=params.get("trend_ma_period", 50),
                ma_kind="ema",
            )
        return _finalize_builder(builder, strategy_name)

    if strategy_name == "Strategy_Simple_RSI_Extremes":
        entry_timeframe = str(params.get("entry_timeframe", (timeframes or ["5m"])[0]))
        timeframes = _ensure_timeframes(timeframes, entry_timeframe=entry_timeframe, default=["5m"])
        builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
        builder.add_indicator("rsi_main", rsi, period=params.get("rsi_period", 8))
        builder.add_signal_rule(
            "rsi_extremes",
            overbought_oversold_crossover,
            indicator="rsi_main",
            overbought=params.get("rsi_overbought", 77),
            oversold=params.get("rsi_oversold", 26),
        )
        builder.add_filter_rule(
            "trend_and_volume",
            trend_and_volume_filter,
            entry_timeframe=entry_timeframe,
            trend_sma_period=params.get("trend_sma_period", 233),
            volume_sma_period=params.get("volume_sma_period", 34),
            volume_multiplier=params.get("volume_multiplier", 1.79),
        )
        return _finalize_builder(builder, strategy_name)

    if strategy_name == "Strategy_Simple_Frequent_Trading":
        entry_timeframe = str(params.get("entry_timeframe", (timeframes or ["1m"])[0]))
        timeframes = _ensure_timeframes(timeframes, entry_timeframe=entry_timeframe, default=["1m"])
        builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
        builder.add_indicator("fast_ma", sma, period=params.get("fast_ma_period", 3))
        builder.add_indicator("slow_ma", sma, period=params.get("slow_ma_period", 8))
        builder.add_indicator("rsi_main", rsi, period=params.get("rsi_period", 7))
        builder.add_signal_rule("ma_signal", ma_crossover, fast_ma="fast_ma", slow_ma="slow_ma")
        builder.add_signal_rule(
            "rsi_signal",
            overbought_oversold,
            indicator="rsi_main",
            overbought=params.get("rsi_overbought", 60),
            oversold=params.get("rsi_oversold", 40),
        )
        builder.set_signal_combination("majority_vote")
        builder.add_filter_rule(
            "price_bias",
            price_change_bias_filter,
            entry_timeframe=entry_timeframe,
            price_change_threshold=params.get("price_change_threshold", 0.001),
        )
        return _finalize_builder(builder, strategy_name)

    if strategy_name == "Strategy_Fast_Test_Scalper":
        entry_timeframe = str(params.get("entry_timeframe", (timeframes or ["1m"])[0]))
        timeframes = _ensure_timeframes(timeframes, entry_timeframe=entry_timeframe, default=["1m"])
        builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
        builder.add_indicator("seed_ema", ema, period=max(2, int(params.get("min_periods", 2))))
        builder.add_signal_rule("price_direction", price_direction_signal, price="price")
        builder.add_filter_rule("entry_timeframe", entry_timeframe_filter, entry_timeframe=entry_timeframe)
        return _finalize_builder(builder, strategy_name)

    if strategy_name == "Strategy_test_bidirectional_strategy":
        entry_timeframe = str(params.get("entry_timeframe", (timeframes or ["1m"])[0]))
        timeframes = _ensure_timeframes(timeframes, entry_timeframe=entry_timeframe, default=["1m"])
        builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
        builder.add_indicator("seed_ema", ema, period=2)
        builder.add_signal_rule("seed_signal", price_direction_signal, price="price")
        builder.add_filter_rule(
            "toggle_test",
            toggle_test_filter,
            entry_timeframe=entry_timeframe,
            test_mode=bool(params.get("test_mode", True)),
        )
        return _finalize_builder(builder, strategy_name)

    if strategy_name == "Strategy_Breakout_Scalping":
        entry_timeframe = str(params.get("entry_timeframe", (timeframes or ["5m"])[0]))
        timeframes = _ensure_timeframes(timeframes, entry_timeframe=entry_timeframe, default=["5m"])
        builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
        builder.add_indicator("resistance", highest, period=params.get("range_period", 20))
        builder.add_indicator("support", lowest, period=params.get("range_period", 20))
        builder.add_signal_rule(
            "breakout",
            breakout_signals,
            price="price",
            resistance="resistance",
            support="support",
            penetration_pct=float(params.get("breakout_threshold", 0.1)) / 100.0,
        )
        builder.add_filter_rule(
            "volume",
            volume_filter,
            entry_timeframe=entry_timeframe,
            volume_sma_period=params.get("volume_sma_period", 20),
            volume_multiplier=params.get("volume_multiplier", 2.0),
        )
        builder.add_filter_rule(
            "atr_gate",
            atr_threshold_filter,
            entry_timeframe=entry_timeframe,
            atr_period=params.get("atr_period", 14),
            min_atr_threshold=params.get("atr_threshold", 0.15),
        )
        return _finalize_builder(builder, strategy_name)

    if strategy_name == "Strategy_Mean_Reversion_Scalping":
        entry_timeframe = str(params.get("entry_timeframe", (timeframes or ["1m"])[0]))
        timeframes = _ensure_timeframes(timeframes, entry_timeframe=entry_timeframe, default=["1m"])
        builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
        builder.add_indicator("rsi_main", rsi, period=params.get("rsi_period", 14))
        builder.add_indicator(
            "bb_main",
            bollinger_bands,
            period=params.get("bb_period", 20),
            std_dev=params.get("bb_std_dev", 2.0),
        )
        builder.add_signal_rule(
            "rsi_signal",
            overbought_oversold_crossover,
            indicator="rsi_main",
            overbought=params.get("rsi_overbought", 75),
            oversold=params.get("rsi_oversold", 25),
        )
        builder.add_signal_rule(
            "bb_signal",
            bollinger_bands_signals,
            price="price",
            upper_band="upper_band",
            lower_band="lower_band",
            middle_band="middle_band",
        )
        builder.set_signal_combination("majority_vote")
        builder.add_filter_rule(
            "volume",
            volume_filter,
            entry_timeframe=entry_timeframe,
            volume_sma_period=params.get("volume_sma_period", 20),
            volume_multiplier=params.get("volume_multiplier", 1.2),
        )
        return _finalize_builder(builder, strategy_name)

    if strategy_name in {
        "Strategy_Multi_Timeframe_Scalping",
        "Strategy_Improved_Multi_Timeframe_Scalping",
        "Strategy_Scalping_MultiIndicator",
        "Strategy_Simple_multi_timeframe",
    }:
        entry_timeframe = str(params.get("entry_timeframe", "1m"))
        trend_timeframe = str(params.get("trend_timeframe", "5m"))
        timeframes = _ensure_timeframes(
            timeframes,
            entry_timeframe=entry_timeframe,
            trend_timeframe=trend_timeframe,
            default=["1m", "5m"],
        )
        builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
        builder.add_indicator("ema_fast", ema, period=params.get("fast_ema_period", params.get("fast_ma_period", 10)))
        builder.add_indicator("ema_slow", ema, period=params.get("slow_ema_period", params.get("slow_ma_period", 30)))
        builder.add_indicator("rsi_main", rsi, period=params.get("rsi_period", 14))
        builder.add_indicator(
            "bb_main",
            bollinger_bands,
            period=params.get("bb_period", 20),
            std_dev=params.get("bb_std_dev", 2.0),
        )
        builder.add_signal_rule("ema_cross", ma_crossover, fast_ma="ema_fast", slow_ma="ema_slow")
        builder.add_signal_rule(
            "rsi_signal",
            overbought_oversold,
            indicator="rsi_main",
            overbought=params.get("rsi_overbought", 70),
            oversold=params.get("rsi_oversold", 30),
        )
        builder.add_signal_rule(
            "bb_signal",
            bollinger_bands_signals,
            price="price",
            upper_band="upper_band",
            lower_band="lower_band",
            middle_band="middle_band",
        )
        builder.set_signal_combination("majority_vote")
        builder.add_filter_rule("entry_timeframe", entry_timeframe_filter, entry_timeframe=entry_timeframe)
        builder.add_filter_rule(
            "trend_tf",
            trend_timeframe_filter,
            entry_timeframe=entry_timeframe,
            trend_timeframe=trend_timeframe,
            ma_period=params.get("trend_ma_period", params.get("slow_ema_period", 50)),
            ma_kind="ema",
        )
        builder.add_filter_rule(
            "atr_gate",
            atr_threshold_filter,
            entry_timeframe=entry_timeframe,
            atr_period=params.get("atr_period", 14),
            min_atr_threshold=params.get("min_atr_threshold", 0.1),
        )
        if "volume_sma_period" in params or "volume_multiplier" in params:
            builder.add_filter_rule(
                "volume",
                volume_filter,
                entry_timeframe=entry_timeframe,
                volume_sma_period=params.get("volume_sma_period", 20),
                volume_multiplier=params.get("volume_multiplier", 1.2),
            )
        return _finalize_builder(builder, strategy_name)

    if strategy_name == "Strategy_multi_timeframe_srsi":
        entry_timeframe = str(params.get("entry_timeframe", "1m"))
        trend_timeframe = str(params.get("trend_timeframe", "5m"))
        timeframes = _ensure_timeframes(
            timeframes,
            entry_timeframe=entry_timeframe,
            trend_timeframe=trend_timeframe,
            default=["1m", "5m"],
        )
        builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
        builder.add_indicator(
            "stoch_main",
            stochastic,
            k_period=params.get("srsi_period", 14),
            d_period=params.get("signal_period", 3),
        )
        builder.add_signal_rule(
            "stoch_signal",
            stochastic_signals,
            k_percent="k_percent",
            d_percent="d_percent",
            overbought=params.get("overbought_level", 80),
            oversold=params.get("oversold_level", 20),
        )
        builder.add_filter_rule("entry_timeframe", entry_timeframe_filter, entry_timeframe=entry_timeframe)
        builder.add_filter_rule(
            "trend_tf",
            trend_timeframe_filter,
            entry_timeframe=entry_timeframe,
            trend_timeframe=trend_timeframe,
            ma_period=params.get("trend_period", 50),
            ma_kind="sma",
        )
        return _finalize_builder(builder, strategy_name)

    if strategy_name == "Strategy_RL_Hybrid":
        entry_timeframe = str(params.get("entry_timeframe", "1m"))
        timeframes = _ensure_timeframes(timeframes, entry_timeframe=entry_timeframe, default=["1m"])
        builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)
        builder.add_indicator("seed_ema", ema, period=2)
        builder.add_signal_rule("seed_signal", price_direction_signal, price="price")
        builder.add_filter_rule(
            "rl_signal",
            rl_signal_filter,
            entry_timeframe=entry_timeframe,
            rl_mode=params.get("rl_mode", "rl_only"),
            model_path=params.get("model_path", "rl_ai/models/phase2_policy.npz"),
            deterministic=bool(params.get("deterministic", True)),
            confirm_strategy_name=params.get("confirm_strategy_name", "Strategy_1_Trend_Following"),
            indicator_columns=params.get("indicator_columns", ""),
            min_bars=params.get("min_bars", 100),
            initial_virtual_balance=params.get("initial_virtual_balance", 10_000.0),
            position_size_pct=params.get("position_size_pct", 0.05),
            max_position_size_pct=params.get("max_position_size_pct", 0.10),
            max_drawdown_pct=params.get("max_drawdown_pct", 0.15),
            fee_pct=params.get("fee_pct", 0.0006),
        )
        return _finalize_builder(builder, strategy_name)

    raise ValueError(f"Unknown builder preset strategy: {strategy_name}")
