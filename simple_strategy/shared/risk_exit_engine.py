"""Shared risk-exit helpers for backtest, paper, and live execution paths."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def normalize_risk_exit_mode(mode: Any) -> str:
    text = str(mode or "").strip().lower()
    if not text:
        return "none"
    if text in {"none", "fixed", "trailing", "atr"}:
        return text
    return "none"


def normalize_risk_pct(value: Any, default: float = 0.02) -> float:
    try:
        pct = float(value)
    except (TypeError, ValueError):
        pct = float(default)
    if pct > 1.0:
        pct = pct / 100.0
    return max(0.0001, min(0.99, pct))


def calculate_atr_from_dataframe(historical_data: Optional[pd.DataFrame], period: int) -> Optional[float]:
    try:
        if historical_data is None or historical_data.empty:
            return None
        p = max(2, int(period))
        if len(historical_data) < (p + 1):
            return None

        high = pd.to_numeric(historical_data["high"], errors="coerce")
        low = pd.to_numeric(historical_data["low"], errors="coerce")
        close = pd.to_numeric(historical_data["close"], errors="coerce")
        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_series = true_range.rolling(window=p, min_periods=p).mean()
        atr_value = float(atr_series.iloc[-1])
        if atr_value <= 0.0:
            return None
        return atr_value
    except Exception:
        return None


def build_position_risk_config(
    *,
    entry_price: float,
    position_type: str,
    risk_exit_mode: Any = "none",
    risk_sl_pct: Any = 0.02,
    risk_atr_period: int = 14,
    risk_atr_sl_multiplier: Any = 1.3,
    risk_atr_tp_multiplier: Any = 1.7,
    atr_value: Optional[float] = None,
) -> Dict[str, Any]:
    position_side = str(position_type).upper()
    requested_mode = normalize_risk_exit_mode(risk_exit_mode)
    mode = requested_mode
    sl_pct = normalize_risk_pct(risk_sl_pct, default=0.02)

    try:
        entry = float(entry_price)
    except (TypeError, ValueError):
        entry = 0.0

    config: Dict[str, Any] = {
        "risk_exit_mode_requested": requested_mode,
        "risk_exit_mode": mode,
        "risk_sl_pct": float(sl_pct),
        "risk_atr_period": max(2, int(risk_atr_period)),
        "risk_atr_sl_multiplier": max(0.1, float(risk_atr_sl_multiplier)),
        "risk_atr_tp_multiplier": max(0.1, float(risk_atr_tp_multiplier)),
    }

    if entry <= 0.0 or position_side not in {"LONG", "SHORT"}:
        config["risk_exit_mode"] = "none"
        return config

    if mode == "none":
        return config

    if mode in {"fixed", "trailing"}:
        if position_side == "LONG":
            config["risk_stop_price"] = entry * (1.0 - sl_pct)
            if mode == "trailing":
                config["risk_high_watermark"] = entry
        else:
            config["risk_stop_price"] = entry * (1.0 + sl_pct)
            if mode == "trailing":
                config["risk_low_watermark"] = entry
        config["stop_loss_pct"] = float(sl_pct)
        return config

    if mode == "atr":
        try:
            atr_numeric = float(atr_value) if atr_value is not None else 0.0
        except (TypeError, ValueError):
            atr_numeric = 0.0

        if atr_numeric <= 0.0:
            mode = "fixed"
            config["risk_exit_mode"] = mode
            if position_side == "LONG":
                config["risk_stop_price"] = entry * (1.0 - sl_pct)
            else:
                config["risk_stop_price"] = entry * (1.0 + sl_pct)
            config["stop_loss_pct"] = float(sl_pct)
            return config

        sl_dist = atr_numeric * float(config["risk_atr_sl_multiplier"])
        tp_dist = atr_numeric * float(config["risk_atr_tp_multiplier"])
        config["risk_atr_value"] = atr_numeric
        if position_side == "LONG":
            config["risk_stop_price"] = entry - sl_dist
            config["risk_take_profit_price"] = entry + tp_dist
        else:
            config["risk_stop_price"] = entry + sl_dist
            config["risk_take_profit_price"] = entry - tp_dist
        stop_loss_pct = abs(entry - float(config["risk_stop_price"])) / max(entry, 1e-12)
        config["stop_loss_pct"] = max(0.0001, stop_loss_pct)
        return config

    config["risk_exit_mode"] = "none"
    return config


def evaluate_risk_exit(
    position: Dict[str, Any],
    *,
    high: Any,
    low: Any,
    current_price: Any = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(position, dict):
        return None

    mode = normalize_risk_exit_mode(position.get("risk_exit_mode", "none"))
    if mode not in {"fixed", "trailing", "atr"}:
        return None

    position_type = str(position.get("type", "")).upper()
    if position_type not in {"LONG", "SHORT"}:
        position_type = "SHORT" if bool(position.get("is_short", False)) else "LONG"

    try:
        high_value = float(high)
    except (TypeError, ValueError):
        high_value = 0.0
    try:
        low_value = float(low)
    except (TypeError, ValueError):
        low_value = 0.0
    try:
        current_value = float(current_price if current_price is not None else 0.0)
    except (TypeError, ValueError):
        current_value = 0.0

    if high_value <= 0.0 and low_value <= 0.0 and current_value <= 0.0:
        return None
    if high_value <= 0.0:
        high_value = current_value if current_value > 0.0 else low_value
    if low_value <= 0.0:
        low_value = current_value if current_value > 0.0 else high_value
    if current_value <= 0.0:
        current_value = low_value if position_type == "LONG" else high_value

    if mode == "trailing":
        sl_pct = normalize_risk_pct(position.get("risk_sl_pct", 0.02), default=0.02)
        if position_type == "LONG":
            high_watermark = max(float(position.get("risk_high_watermark", 0.0) or 0.0), high_value)
            position["risk_high_watermark"] = high_watermark
            stop_price = high_watermark * (1.0 - sl_pct)
            position["risk_stop_price"] = stop_price
            if low_value <= stop_price:
                return {"reason": "trailing_stop_hit", "trigger_price": stop_price}
        else:
            seed = float(position.get("risk_low_watermark", current_value) or current_value)
            low_watermark = min(seed, low_value)
            position["risk_low_watermark"] = low_watermark
            stop_price = low_watermark * (1.0 + sl_pct)
            position["risk_stop_price"] = stop_price
            if high_value >= stop_price:
                return {"reason": "trailing_stop_hit", "trigger_price": stop_price}
        return None

    stop_price = float(position.get("risk_stop_price", 0.0) or 0.0)
    take_profit_price = float(position.get("risk_take_profit_price", 0.0) or 0.0)

    if stop_price > 0.0:
        if position_type == "LONG" and low_value <= stop_price:
            return {
                "reason": "atr_stop_loss_hit" if mode == "atr" else "fixed_stop_loss_hit",
                "trigger_price": stop_price,
            }
        if position_type == "SHORT" and high_value >= stop_price:
            return {
                "reason": "atr_stop_loss_hit" if mode == "atr" else "fixed_stop_loss_hit",
                "trigger_price": stop_price,
            }

    if mode == "atr" and take_profit_price > 0.0:
        if position_type == "LONG" and high_value >= take_profit_price:
            return {"reason": "atr_take_profit_hit", "trigger_price": take_profit_price}
        if position_type == "SHORT" and low_value <= take_profit_price:
            return {"reason": "atr_take_profit_hit", "trigger_price": take_profit_price}

    return None
