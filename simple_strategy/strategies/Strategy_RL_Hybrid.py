"""
Strategy: RL Hybrid

Modes:
- rl_only: use RL signal directly.
- rl_confirm: RL signal is allowed only when a second strategy confirms it.

This is additive integration for Phase 3:
- Keeps strict signal schema.
- Does not change backtester/paper-trader core paths.
"""

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from simple_strategy.shared.strategy_base import StrategyBase
from rl_ai.shared.feature_builder import AccountSnapshot, FeatureBuilder
from rl_ai.shared.feature_schema import FeatureSchema
from rl_ai.shared.signal_adapter import SignalAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

VALID_SIGNALS = {"OPEN_LONG", "CLOSE_LONG", "OPEN_SHORT", "CLOSE_SHORT", "HOLD"}


STRATEGY_PARAMETERS = {
    "rl_mode": {
        "type": "str",
        "default": "rl_only",
        "options": ["rl_only", "rl_confirm"],
        "description": "RL signal mode",
        "gui_hint": "rl_only = RL standalone, rl_confirm = RL must match confirm strategy",
    },
    "model_path": {
        "type": "str",
        "default": "rl_ai/models/phase2_policy.npz",
        "description": "Path to trained RL model (.npz or .zip)",
        "gui_hint": "Use .npz for linear model, .zip for deep SB3 model",
    },
    "deterministic": {
        "type": "bool",
        "default": True,
        "description": "Use greedy RL action (no sampling)",
        "gui_hint": "Recommended True for stable behavior",
    },
    "entry_timeframe": {
        "type": "str",
        "default": "1m",
        "options": ["1m", "3m", "5m", "15m"],
        "description": "Timeframe where RL emits signals",
        "gui_hint": "Use 1m first for paper testing",
    },
    "confirm_strategy_name": {
        "type": "str",
        "default": "Strategy_1_Trend_Following",
        "description": "Strategy file name used for confirmation mode",
        "gui_hint": "Only used when rl_mode=rl_confirm",
    },
    "indicator_columns": {
        "type": "str",
        "default": "",
        "description": "Comma-separated indicator column names for RL features",
        "gui_hint": "Example: rsi_1,ema_fast_1,ema_slow_1",
    },
    "min_bars": {
        "type": "int",
        "default": 100,
        "min": 10,
        "max": 5000,
        "description": "Minimum candles before strategy can emit non-HOLD",
        "gui_hint": "Use larger values if indicators need warmup",
    },
    "initial_virtual_balance": {
        "type": "float",
        "default": 10000.0,
        "description": "Virtual balance used by internal RL risk guard",
        "gui_hint": "Does not change exchange order sizing",
    },
    "position_size_pct": {
        "type": "float",
        "default": 0.05,
        "min": 0.0,
        "max": 1.0,
        "description": "Virtual position size fraction used by risk guard",
        "gui_hint": "0.05 = 5% per open signal",
    },
    "max_position_size_pct": {
        "type": "float",
        "default": 0.10,
        "min": 0.0,
        "max": 1.0,
        "description": "Hard cap for virtual position size",
        "gui_hint": "Open signals are blocked if effective size is 0",
    },
    "max_drawdown_pct": {
        "type": "float",
        "default": 0.15,
        "min": 0.0,
        "max": 1.0,
        "description": "Hard drawdown stop (fraction)",
        "gui_hint": "0.15 = stop after 15% drawdown",
    },
    "fee_pct": {
        "type": "float",
        "default": 0.0006,
        "min": 0.0,
        "max": 0.01,
        "description": "Virtual per-side fee for risk tracking",
        "gui_hint": "Used only for internal guard accounting",
    },
}


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
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


class RLHybridStrategy(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__(
            name="RL_Hybrid_Strategy",
            symbols=symbols,
            timeframes=timeframes,
            config=config,
        )

        self.rl_mode = str(config.get("rl_mode", "rl_only")).strip().lower()
        if self.rl_mode not in ("rl_only", "rl_confirm"):
            self.rl_mode = "rl_only"

        self.model_path = str(config.get("model_path", "rl_ai/models/phase2_policy.npz")).strip()
        self.deterministic = bool(config.get("deterministic", True))
        self.entry_timeframe = str(config.get("entry_timeframe", "1m")).strip()
        self.confirm_strategy_name = str(config.get("confirm_strategy_name", "")).strip()
        self.indicator_columns = _parse_indicator_columns(config.get("indicator_columns", ""))
        self.min_bars = max(10, int(config.get("min_bars", 100)))
        self.initial_virtual_balance = max(
            100.0, float(config.get("initial_virtual_balance", 10_000.0))
        )
        self.position_size_pct = _clamp_ratio(config.get("position_size_pct", 0.05), default=0.05)
        self.max_position_size_pct = _clamp_ratio(
            config.get("max_position_size_pct", 0.10), default=0.10
        )
        self.max_drawdown_pct = _clamp_ratio(config.get("max_drawdown_pct", 0.15), default=0.15)
        self.fee_pct = max(0.0, float(config.get("fee_pct", 0.0006)))

        self._effective_position_size_pct = min(self.position_size_pct, self.max_position_size_pct)
        if self.position_size_pct > self.max_position_size_pct:
            logger.warning(
                "position_size_pct %.4f exceeds max_position_size_pct %.4f, clamped.",
                self.position_size_pct,
                self.max_position_size_pct,
            )

        self._position_state: Dict[tuple, Dict[str, Any]] = {}
        self._confirm_strategy = None
        self._policy_model = None
        self._policy_metadata: Dict[str, Any] = {}
        self._latest_price_by_position_key: Dict[tuple, float] = {}
        self._latest_decision_by_position_key: Dict[tuple, Dict[str, Any]] = {}

        self._virtual_balance = float(self.initial_virtual_balance)
        self._virtual_peak_equity = float(self.initial_virtual_balance)
        self._virtual_drawdown_pct = 0.0
        self._drawdown_lock = False

        self._feature_schema = FeatureSchema(indicator_features=self.indicator_columns)
        self._feature_builder = FeatureBuilder(schema=self._feature_schema)

        self._load_rl_model()
        self._load_confirmation_strategy()

    def _resolve_model_path(self) -> Path:
        path = Path(self.model_path)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def _load_rl_model(self) -> None:
        try:
            model_file = self._resolve_model_path()
            if not model_file.exists():
                logger.warning("RL model not found at %s. Strategy will emit HOLD unless patched.", model_file)
                return

            model_io_module = importlib.import_module("rl_ai.01_library_based.model_io")
            load_policy_model = model_io_module.load_policy_model
            self._policy_model, self._policy_metadata = load_policy_model(str(model_file))
        except Exception as exc:
            logger.error("Failed to load RL model: %s", exc)
            self._policy_model = None
            self._policy_metadata = {}

    def _load_confirmation_strategy(self) -> None:
        if self.rl_mode != "rl_confirm":
            return

        if not self.confirm_strategy_name:
            return

        if self.confirm_strategy_name == "Strategy_RL_Hybrid":
            logger.warning("confirm_strategy_name points to self; disabling confirmation strategy.")
            return

        module = None
        for module_name in (
            f"simple_strategy.strategies.{self.confirm_strategy_name}",
            f"strategies.{self.confirm_strategy_name}",
        ):
            try:
                module = importlib.import_module(module_name)
                break
            except Exception:
                module = None

        if module is None:
            logger.warning("Confirmation strategy module not found: %s", self.confirm_strategy_name)
            return

        if not hasattr(module, "create_strategy"):
            logger.warning("Confirmation strategy missing create_strategy: %s", self.confirm_strategy_name)
            return

        try:
            self._confirm_strategy = module.create_strategy(
                symbols=self.symbols,
                timeframes=self.timeframes,
            )
        except Exception as exc:
            logger.warning("Failed to initialize confirmation strategy %s: %s", self.confirm_strategy_name, exc)
            self._confirm_strategy = None

    def _estimate_unrealized_pnl(self, position: Dict[str, Any], current_price: float) -> float:
        entry_price = float(position.get("entry_price", 0.0) or 0.0)
        notional = float(position.get("notional", 0.0) or 0.0)
        if entry_price <= 0.0 or current_price <= 0.0 or notional <= 0.0:
            return 0.0

        is_short = bool(position.get("is_short", False))
        if is_short:
            pnl_ratio = (entry_price - current_price) / entry_price
        else:
            pnl_ratio = (current_price - entry_price) / entry_price
        return notional * pnl_ratio

    def _current_equity(self) -> float:
        equity = float(self._virtual_balance)
        for position_key, position in self._position_state.items():
            current_price = float(self._latest_price_by_position_key.get(position_key, 0.0) or 0.0)
            equity += self._estimate_unrealized_pnl(position, current_price)
        return float(equity)

    def _refresh_risk_state(self) -> None:
        equity = self._current_equity()
        if equity > self._virtual_peak_equity:
            self._virtual_peak_equity = float(equity)

        if self._virtual_peak_equity > 0:
            self._virtual_drawdown_pct = max(
                0.0, (self._virtual_peak_equity - equity) / self._virtual_peak_equity
            )
        else:
            self._virtual_drawdown_pct = 0.0

        if self._virtual_drawdown_pct >= self.max_drawdown_pct:
            self._drawdown_lock = True

    def _risk_guard_signal(self, position_key: tuple) -> Optional[str]:
        self._refresh_risk_state()

        if not self._drawdown_lock:
            return None

        position = self._position_state.get(position_key)
        if position is None:
            return "HOLD"
        return "CLOSE_SHORT" if bool(position.get("is_short", False)) else "CLOSE_LONG"

    def _apply_position_rules(self, position_key: tuple, raw_signal: str, current_price: float) -> str:
        if raw_signal not in VALID_SIGNALS:
            return "HOLD"

        position = self._position_state.get(position_key)

        if raw_signal == "OPEN_LONG":
            if position is not None:
                return "HOLD"
            if self._drawdown_lock:
                return "HOLD"
            if current_price <= 0.0:
                return "HOLD"
            if self._effective_position_size_pct <= 0.0:
                return "HOLD"

            notional = self._virtual_balance * self._effective_position_size_pct
            if notional <= 0.0:
                return "HOLD"

            open_fee = notional * self.fee_pct
            if open_fee >= self._virtual_balance:
                return "HOLD"

            self._virtual_balance -= open_fee
            self._position_state[position_key] = {
                "is_short": False,
                "entry_price": float(current_price),
                "notional": float(notional),
                "size_pct": float(self._effective_position_size_pct),
            }
            self._refresh_risk_state()
            return raw_signal

        if raw_signal == "OPEN_SHORT":
            if position is not None:
                return "HOLD"
            if self._drawdown_lock:
                return "HOLD"
            if current_price <= 0.0:
                return "HOLD"
            if self._effective_position_size_pct <= 0.0:
                return "HOLD"

            notional = self._virtual_balance * self._effective_position_size_pct
            if notional <= 0.0:
                return "HOLD"

            open_fee = notional * self.fee_pct
            if open_fee >= self._virtual_balance:
                return "HOLD"

            self._virtual_balance -= open_fee
            self._position_state[position_key] = {
                "is_short": True,
                "entry_price": float(current_price),
                "notional": float(notional),
                "size_pct": float(self._effective_position_size_pct),
            }
            self._refresh_risk_state()
            return raw_signal

        if raw_signal == "CLOSE_LONG":
            if position is None or position.get("is_short", False):
                return "HOLD"
            realized_pnl = self._estimate_unrealized_pnl(position, current_price)
            close_notional = max(0.0, float(position.get("notional", 0.0)) + realized_pnl)
            close_fee = close_notional * self.fee_pct
            self._virtual_balance += realized_pnl - close_fee
            if self._virtual_balance < 0.0:
                self._virtual_balance = 0.0
            self._position_state.pop(position_key, None)
            self._refresh_risk_state()
            return raw_signal

        if raw_signal == "CLOSE_SHORT":
            if position is None or not position.get("is_short", False):
                return "HOLD"
            realized_pnl = self._estimate_unrealized_pnl(position, current_price)
            close_notional = max(0.0, float(position.get("notional", 0.0)) + realized_pnl)
            close_fee = close_notional * self.fee_pct
            self._virtual_balance += realized_pnl - close_fee
            if self._virtual_balance < 0.0:
                self._virtual_balance = 0.0
            self._position_state.pop(position_key, None)
            self._refresh_risk_state()
            return raw_signal

        return "HOLD"

    def _predict_rl_signal(self, df: pd.DataFrame, position_key: tuple) -> str:
        if self._policy_model is None:
            return "HOLD"

        position = self._position_state.get(position_key)
        has_position = position is not None
        is_short = bool(position and position.get("is_short", False))

        close_price = float(df["close"].iloc[-1]) if len(df) else 0.0
        entry_price = None
        if position is not None and position.get("entry_price") is not None:
            entry_price = float(position.get("entry_price"))

        self._refresh_risk_state()

        account = AccountSnapshot(
            balance=float(self._virtual_balance),
            initial_balance=float(self.initial_virtual_balance),
            has_position=has_position,
            is_short=is_short,
            entry_price=entry_price,
            current_price=close_price,
            drawdown_pct=float(self._virtual_drawdown_pct),
        )

        try:
            observation = self._feature_builder.build_vector(df, row_index=len(df) - 1, account=account)
        except Exception:
            return "HOLD"

        if int(getattr(self._policy_model, "observation_size", -1)) != int(len(observation)):
            return "HOLD"

        try:
            if self.deterministic:
                action = int(self._policy_model.greedy_action(observation))
            else:
                action = int(self._policy_model.sample_action(observation).action)
        except Exception:
            return "HOLD"

        return SignalAdapter.action_to_signal(action, has_position=has_position, is_short=is_short)

    def _get_confirmation_signal(
        self,
        symbol: str,
        timeframe: str,
        data: Dict[str, Dict[str, pd.DataFrame]],
    ) -> str:
        if self._confirm_strategy is None:
            return "HOLD"

        try:
            confirm_signals = self._confirm_strategy.generate_signals(data)
            signal = confirm_signals.get(symbol, {}).get(timeframe, "HOLD")
            return signal if signal in VALID_SIGNALS else "HOLD"
        except Exception:
            return "HOLD"

    def _merge_mode_signal(self, rl_signal: str, confirm_signal: str) -> str:
        if self.rl_mode == "rl_only":
            return rl_signal
        if self.rl_mode == "rl_confirm":
            return rl_signal if rl_signal == confirm_signal else "HOLD"
        return "HOLD"

    def _record_decision(
        self,
        position_key: tuple,
        timeframe: str,
        current_price: float,
        rl_signal: str,
        confirm_signal: str,
        raw_signal: str,
        final_signal: str,
        reason: str,
    ) -> None:
        self._latest_decision_by_position_key[position_key] = {
            "timeframe": timeframe,
            "price": float(current_price),
            "rl_signal": rl_signal,
            "confirm_signal": confirm_signal,
            "raw_signal": raw_signal,
            "final_signal": final_signal,
            "reason": reason,
            "drawdown_lock": bool(self._drawdown_lock),
            "virtual_balance": float(self._virtual_balance),
            "virtual_drawdown_pct": float(self._virtual_drawdown_pct),
        }

    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        signals: Dict[str, Dict[str, str]] = {}
        entry_tf_normalized = _normalize_timeframe(self.entry_timeframe)

        for symbol, tf_map in data.items():
            signals[symbol] = {}

            for timeframe, df in tf_map.items():
                timeframe_normalized = _normalize_timeframe(timeframe)
                position_key = (symbol, timeframe_normalized)
                if timeframe_normalized != entry_tf_normalized:
                    signals[symbol][timeframe] = "HOLD"
                    self._record_decision(
                        position_key=position_key,
                        timeframe=timeframe,
                        current_price=0.0,
                        rl_signal="HOLD",
                        confirm_signal="HOLD",
                        raw_signal="HOLD",
                        final_signal="HOLD",
                        reason="timeframe_not_enabled",
                    )
                    continue

                if df is None or len(df) < self.min_bars:
                    signals[symbol][timeframe] = "HOLD"
                    current_price = (
                        float(df["close"].iloc[-1]) if (df is not None and len(df) > 0) else 0.0
                    )
                    self._record_decision(
                        position_key=position_key,
                        timeframe=timeframe,
                        current_price=current_price,
                        rl_signal="HOLD",
                        confirm_signal="HOLD",
                        raw_signal="HOLD",
                        final_signal="HOLD",
                        reason="insufficient_bars",
                    )
                    continue

                current_price = float(df["close"].iloc[-1])
                if current_price <= 0.0:
                    signals[symbol][timeframe] = "HOLD"
                    self._record_decision(
                        position_key=position_key,
                        timeframe=timeframe,
                        current_price=current_price,
                        rl_signal="HOLD",
                        confirm_signal="HOLD",
                        raw_signal="HOLD",
                        final_signal="HOLD",
                        reason="invalid_price",
                    )
                    continue

                self._latest_price_by_position_key[position_key] = current_price

                risk_override_signal = self._risk_guard_signal(position_key)
                if risk_override_signal is not None:
                    raw_signal = risk_override_signal
                    final_signal = self._apply_position_rules(
                        position_key, raw_signal, current_price=current_price
                    )
                    final_signal = final_signal if final_signal in VALID_SIGNALS else "HOLD"
                    signals[symbol][timeframe] = final_signal
                    self._record_decision(
                        position_key=position_key,
                        timeframe=timeframe,
                        current_price=current_price,
                        rl_signal="HOLD",
                        confirm_signal="HOLD",
                        raw_signal=raw_signal,
                        final_signal=final_signal,
                        reason="risk_guard_override",
                    )
                    continue

                rl_signal = self._predict_rl_signal(df, position_key)

                confirm_signal = "HOLD"
                if self.rl_mode == "rl_confirm":
                    confirm_signal = self._get_confirmation_signal(symbol, timeframe, data)

                raw_signal = self._merge_mode_signal(rl_signal, confirm_signal)
                if raw_signal in ("OPEN_LONG", "OPEN_SHORT") and self._effective_position_size_pct <= 0.0:
                    raw_signal = "HOLD"
                    reason = "position_size_cap_block"
                else:
                    reason = "mode_merge"

                final_signal = self._apply_position_rules(
                    position_key, raw_signal, current_price=current_price
                )
                final_signal = final_signal if final_signal in VALID_SIGNALS else "HOLD"
                signals[symbol][timeframe] = final_signal
                self._record_decision(
                    position_key=position_key,
                    timeframe=timeframe,
                    current_price=current_price,
                    rl_signal=rl_signal,
                    confirm_signal=confirm_signal,
                    raw_signal=raw_signal,
                    final_signal=final_signal,
                    reason=reason,
                )

        return signals

    def get_strategy_state(self) -> Dict[str, Any]:
        state = super().get_strategy_state()
        state.update(
            {
                "rl_mode": self.rl_mode,
                "entry_timeframe": self.entry_timeframe,
                "drawdown_lock": bool(self._drawdown_lock),
                "virtual_balance": float(self._virtual_balance),
                "virtual_equity": float(self._current_equity()),
                "virtual_peak_equity": float(self._virtual_peak_equity),
                "virtual_drawdown_pct": float(self._virtual_drawdown_pct),
                "max_drawdown_pct": float(self.max_drawdown_pct),
                "effective_position_size_pct": float(self._effective_position_size_pct),
                "latest_decisions": {
                    f"{key[0]}:{key[1]}": decision
                    for key, decision in self._latest_decision_by_position_key.items()
                },
                "model_backend": str(self._policy_metadata.get("model_backend", "linear_npz")),
                "model_format": str(self._policy_metadata.get("model_format", "npz")),
            }
        )
        return state


def create_strategy(symbols=None, timeframes=None, **params):
    if symbols is None or len(symbols) == 0:
        symbols = ["BTCUSDT"]
    if timeframes is None or len(timeframes) == 0:
        timeframes = ["1m"]

    entry_timeframe = str(params.get("entry_timeframe", "1m"))
    normalized_timeframes = {_normalize_timeframe(tf): tf for tf in timeframes}
    if _normalize_timeframe(entry_timeframe) not in normalized_timeframes:
        timeframes = list(timeframes) + [entry_timeframe]

    config = {
        "rl_mode": params.get("rl_mode", "rl_only"),
        "model_path": params.get("model_path", "rl_ai/models/phase2_policy.npz"),
        "deterministic": params.get("deterministic", True),
        "entry_timeframe": entry_timeframe,
        "confirm_strategy_name": params.get("confirm_strategy_name", "Strategy_1_Trend_Following"),
        "indicator_columns": params.get("indicator_columns", ""),
        "min_bars": params.get("min_bars", 100),
        "initial_virtual_balance": params.get("initial_virtual_balance", 10_000.0),
        "position_size_pct": params.get("position_size_pct", 0.05),
        "max_position_size_pct": params.get("max_position_size_pct", 0.10),
        "max_drawdown_pct": params.get("max_drawdown_pct", 0.15),
        "fee_pct": params.get("fee_pct", 0.0006),
    }
    return RLHybridStrategy(symbols=symbols, timeframes=timeframes, config=config)
