import logging
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from simple_strategy.backtester.backtester_engine import BacktesterEngine
from simple_strategy.shared.data_feeder import DataFeeder

from .candidate_builder import CandidateBuilder
from .gates import gate_stress, gate_train_metrics, gate_validation_consistency
from .scoring import composite_score, metrics_or_zero

logger = logging.getLogger(__name__)


class CandidateEvaluator:
    """Evaluate one candidate strategy with train/validation/stress gates."""

    def __init__(
        self,
        candidate_builder: CandidateBuilder,
        run_config: Dict[str, Any],
        data_dir: str = "data",
    ):
        self.builder = candidate_builder
        self.run_config = run_config
        self.data_feeder = DataFeeder(data_dir=data_dir)

        self.base_backtest_config = {
            "fee_pct": 0.00055,
            "spread_pct": 0.00040,
            "slippage_pct": 0.00030,
            "max_positions": 3,
            "risk_per_trade": 0.02,
            "position_size_pct": 0.05,
            "risk_exit_mode": "none",
            "risk_sl_pct": 0.02,
            "risk_atr_period": 14,
            "risk_atr_sl_multiplier": 1.3,
            "risk_atr_tp_multiplier": 1.7,
        }

    def evaluate_candidate(
        self,
        candidate: Dict[str, Any],
        generation: int,
        candidate_idx: int,
        data_override: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
        use_inner_optimization: bool = True,
    ) -> Dict[str, Any]:
        candidate_working = deepcopy(candidate)

        try:
            if use_inner_optimization and self.run_config.get("enable_inner_optimization", True):
                candidate_working = self._optimize_candidate(candidate_working, data_override)

            strategy_name = f"AE_G{generation:03d}_I{candidate_idx:04d}"
            strategy = self.builder.build_strategy(
                candidate_working,
                symbols=self.run_config["symbols"],
                timeframes=self.run_config["timeframes"],
                strategy_name=strategy_name,
            )

            train_metrics = self._run_segment(
                strategy,
                candidate=candidate_working,
                start=self.run_config["train_start"],
                end=self.run_config["train_end"],
                data_override=data_override,
            )
            train_gate = gate_train_metrics(
                train_metrics,
                min_trades=int(self.run_config.get("min_trades", 10)),
                min_return=float(self.run_config.get("min_train_return_pct", 0.0)),
                max_drawdown=float(self.run_config.get("max_train_drawdown_pct", 40.0)),
            )
            if not train_gate.passed:
                return self._failed_result(candidate_working, train_metrics, "train_gate", train_gate.reason)

            validation_metrics = self._run_segment(
                strategy,
                candidate=candidate_working,
                start=self.run_config["validation_start"],
                end=self.run_config["validation_end"],
                data_override=data_override,
            )
            validation_quality_gate = gate_train_metrics(
                validation_metrics,
                min_trades=int(self.run_config.get("validation_min_trades", 1)),
                min_return=float(self.run_config.get("min_validation_return_pct", -5.0)),
                max_drawdown=float(self.run_config.get("max_validation_drawdown_pct", 45.0)),
            )
            if not validation_quality_gate.passed:
                return self._failed_result(
                    candidate_working,
                    train_metrics,
                    "validation_gate",
                    validation_quality_gate.reason,
                )
            validation_gate = gate_validation_consistency(
                train_metrics,
                validation_metrics,
                max_return_gap=float(self.run_config.get("max_validation_gap_pct", 30.0)),
            )
            if not validation_gate.passed:
                return self._failed_result(candidate_working, train_metrics, "validation_gate", validation_gate.reason)

            stress_metrics = self._run_segment(
                strategy,
                candidate=candidate_working,
                start=self.run_config["train_start"],
                end=self.run_config["train_end"],
                data_override=data_override,
                fee_multiplier=float(self.run_config.get("stress_fee_multiplier", 1.5)),
                slippage_multiplier=float(self.run_config.get("stress_slippage_multiplier", 1.5)),
            )
            stress_gate = gate_stress(
                train_metrics,
                stress_metrics,
                max_drop=float(self.run_config.get("max_stress_drop_pct", 30.0)),
                min_return=float(self.run_config.get("min_stress_return_pct", -10.0)),
                max_drawdown=float(self.run_config.get("max_stress_drawdown_pct", 50.0)),
            )
            if not stress_gate.passed:
                return self._failed_result(candidate_working, train_metrics, "stress_gate", stress_gate.reason)

            score = composite_score(
                train_metrics,
                validation_metrics,
                stress_metrics,
                active_signal_count=len(candidate_working.get("active_signals", [])),
            )

            return {
                "passed": True,
                "score": score,
                "candidate": candidate_working,
                "metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "stress_metrics": stress_metrics,
                "failed_gate": "",
                "failure_reason": "",
            }
        except Exception as exc:
            logger.exception("Candidate evaluation failed: %s", exc)
            return self._failed_result(candidate_working, {}, "exception", str(exc))

    def _failed_result(
        self,
        candidate: Dict[str, Any],
        metrics: Dict[str, float],
        failed_gate: str,
        reason: str,
    ) -> Dict[str, Any]:
        return {
            "passed": False,
            "score": -1e9,
            "candidate": candidate,
            "metrics": metrics_or_zero(metrics),
            "validation_metrics": metrics_or_zero({}),
            "stress_metrics": metrics_or_zero({}),
            "failed_gate": failed_gate,
            "failure_reason": reason,
        }

    def _run_segment(
        self,
        strategy,
        start: str,
        end: str,
        candidate: Optional[Dict[str, Any]] = None,
        data_override: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
        fee_multiplier: float = 1.0,
        slippage_multiplier: float = 1.0,
    ) -> Dict[str, float]:
        self._ensure_utf8_stdout()
        config = deepcopy(self.base_backtest_config)
        if candidate:
            config.update(self._candidate_backtest_config(candidate))
        config["fee_pct"] *= fee_multiplier
        config["spread_pct"] *= slippage_multiplier
        config["slippage_pct"] *= slippage_multiplier

        data = self._slice_data(data_override, start, end) if data_override else None
        if data is not None and not self._has_non_empty_data(data):
            return metrics_or_zero({})

        engine = BacktesterEngine(data_feeder=self.data_feeder, strategy=strategy, config=config)
        result = engine.run_backtest(
            symbols=self.run_config["symbols"],
            timeframes=self.run_config["timeframes"],
            start_date=start,
            end_date=end,
            config=config,
            strategy=strategy,
            data=data,
            initial_balance=float(self.run_config.get("initial_balance", 10000.0)),
        )
        return metrics_or_zero(result)

    @staticmethod
    def _candidate_backtest_config(candidate: Dict[str, Any]) -> Dict[str, Any]:
        risk = deepcopy(candidate.get("risk_exit", {}))
        if not risk:
            return {}
        return {
            "risk_exit_mode": str(risk.get("risk_exit_mode", "none")).lower(),
            "risk_sl_pct": float(risk.get("risk_sl_pct", 0.02)),
            "risk_atr_period": int(risk.get("risk_atr_period", 14)),
            "risk_atr_sl_multiplier": float(risk.get("risk_atr_sl_multiplier", 1.3)),
            "risk_atr_tp_multiplier": float(risk.get("risk_atr_tp_multiplier", 1.7)),
            "position_size_pct": float(risk.get("position_size_pct", 0.05)),
        }

    @staticmethod
    def _ensure_utf8_stdout() -> None:
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            # Best effort only; keep running if stdout cannot be changed.
            pass

    @staticmethod
    def _slice_data(
        data: Dict[str, Dict[str, pd.DataFrame]],
        start: str,
        end: str,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)

        sliced: Dict[str, Dict[str, pd.DataFrame]] = {}
        for symbol, by_tf in data.items():
            sliced[symbol] = {}
            for timeframe, frame in by_tf.items():
                if frame.empty:
                    sliced[symbol][timeframe] = frame
                    continue

                idx = frame.index
                if not isinstance(idx, pd.DatetimeIndex):
                    if "datetime" in frame.columns:
                        local = frame.copy()
                        local["datetime"] = pd.to_datetime(local["datetime"])
                        local.set_index("datetime", inplace=True)
                        frame = local
                    else:
                        frame = frame.copy()
                        frame.index = pd.to_datetime(frame.index)

                segment = frame[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()
                sliced[symbol][timeframe] = segment
        return sliced

    @staticmethod
    def _has_non_empty_data(data: Dict[str, Dict[str, pd.DataFrame]]) -> bool:
        for by_tf in data.values():
            for frame in by_tf.values():
                if frame is not None and not frame.empty:
                    return True
        return False

    def _optimize_candidate(
        self,
        candidate: Dict[str, Any],
        data_override: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    ) -> Dict[str, Any]:
        trials = int(self.run_config.get("inner_opt_trials", 0))
        if trials <= 0:
            return candidate

        try:
            import optuna
        except Exception:
            return candidate

        search_space = self.builder.search_space

        def _objective(trial: "optuna.Trial") -> float:
            tuned = deepcopy(candidate)

            for indicator_id in self.builder.required_indicators(candidate):
                param_defs = search_space["indicators"][indicator_id].get("params", {})
                for name, pdef in param_defs.items():
                    tuned["indicator_params"][indicator_id][name] = self._suggest(trial, f"i__{indicator_id}__{name}", pdef)

            for signal_id in candidate.get("active_signals", []):
                param_defs = search_space["signals"][signal_id].get("params", {})
                for name, pdef in param_defs.items():
                    tuned["signal_params"][signal_id][name] = self._suggest(trial, f"s__{signal_id}__{name}", pdef)

            for filter_id in candidate.get("active_filters", []):
                param_defs = search_space.get("filters", {}).get(filter_id, {}).get("params", {})
                for name, pdef in param_defs.items():
                    tuned["filter_params"][filter_id][name] = self._suggest(trial, f"f__{filter_id}__{name}", pdef)

            for name, pdef in search_space.get("risk_exit", {}).items():
                tuned["risk_exit"][name] = self._suggest(trial, f"r__{name}", pdef)

            self.builder._normalize_candidate(tuned)

            strategy = self.builder.build_strategy(
                tuned,
                symbols=self.run_config["symbols"],
                timeframes=self.run_config["timeframes"],
                strategy_name="AE_INNER_OPT",
            )
            metrics = self._run_segment(
                strategy,
                candidate=tuned,
                start=self.run_config["train_start"],
                end=self.run_config["train_end"],
                data_override=data_override,
            )

            if metrics.get("total_trades", 0) < int(self.run_config.get("min_trades", 10)):
                return -1e6

            return (
                float(metrics.get("total_return", 0.0))
                + (0.25 * float(metrics.get("sharpe_ratio", 0.0)))
                - (0.20 * float(metrics.get("max_drawdown", 0.0)))
            )

        sampler = optuna.samplers.TPESampler(seed=int(self.run_config.get("seed", 42)))
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(_objective, n_trials=trials, show_progress_bar=False)

        if study.best_trial is None:
            return candidate
        if float(study.best_value) <= -1e5:
            return candidate

        best = deepcopy(candidate)
        for key, value in study.best_trial.params.items():
            parts = key.split("__", maxsplit=2)
            section = parts[0]
            if section == "i":
                _, block, name = parts
                best["indicator_params"][block][name] = value
            elif section == "s":
                _, block, name = parts
                best["signal_params"][block][name] = value
            elif section == "f":
                _, block, name = parts
                best["filter_params"][block][name] = value
            elif section == "r":
                _, name = parts
                best["risk_exit"][name] = value

        self.builder._normalize_candidate(best)
        return best

    def run_final_segment(
        self,
        candidate: Dict[str, Any],
        data_override: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    ) -> Dict[str, float]:
        strategy = self.builder.build_strategy(
            candidate,
            symbols=self.run_config["symbols"],
            timeframes=self.run_config["timeframes"],
            strategy_name="AE_FINAL_SEGMENT",
        )
        return self._run_segment(
            strategy,
            candidate=candidate,
            start=self.run_config["final_start"],
            end=self.run_config["final_end"],
            data_override=data_override,
        )

    @staticmethod
    def _suggest(trial, key: str, pdef: Dict[str, Any]):
        ptype = pdef["type"]
        if ptype == "int":
            return trial.suggest_int(key, int(pdef["low"]), int(pdef["high"]), step=int(pdef.get("step", 1)))
        if ptype == "float":
            return trial.suggest_float(key, float(pdef["low"]), float(pdef["high"]), step=float(pdef.get("step", 0.1)))
        if ptype == "choice":
            return trial.suggest_categorical(key, list(pdef["choices"]))
        raise ValueError(f"Unsupported parameter type: {ptype}")


def build_synthetic_data(
    symbol: str = "BTCUSDT",
    timeframe: str = "5m",
    candles: int = 500,
    start: str = "2024-01-01",
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Synthetic OHLCV for smoke testing without external files."""
    start_ts = pd.to_datetime(start)
    idx = pd.date_range(start=start_ts, periods=candles, freq="5min")

    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.0015, size=candles)
    close = 100.0 * np.exp(np.cumsum(returns))

    frame = pd.DataFrame(index=idx)
    frame["close"] = close
    frame["open"] = frame["close"].shift(1).fillna(frame["close"])

    spread = np.abs(rng.normal(0, 0.002, size=candles))
    frame["high"] = frame[["open", "close"]].max(axis=1) * (1 + spread)
    frame["low"] = frame[["open", "close"]].min(axis=1) * (1 - spread)
    frame["volume"] = rng.integers(10, 500, size=candles)

    return {symbol: {timeframe: frame}}
