import argparse
import csv
import json
import random
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from simple_strategy.auto_evolve.candidate_builder import CandidateBuilder
from simple_strategy.auto_evolve.reports import render_strategy_module
from simple_strategy.auto_evolve.search_space import load_search_space
from simple_strategy.backtester.backtester_engine import BacktesterEngine
from simple_strategy.shared.data_feeder import DataFeeder


DEFAULT_FAST_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "data_dir": "data",
    "symbols": ["BNBUSDT", "XRPUSDT", "ADAUSDT", "DOTUSDT", "NEARUSDT"],
    "timeframes": ["1", "5"],
    "start_date": "2026-01-01",
    "end_date": "2026-02-15",
    "candidate_count": 250,
    "workers": 4,
    "top_k": 10,
    "max_active_signals": 3,
    "output_dir": "simple_strategy/fast_finder/runs",
    "publish_top_n": 3,
    "quality_preset": "custom",
    "search_profile": "basic",
    "search_space_path": "",
    "initial_balance": 10000.0,
    "min_return_pct": 0.0,
    "max_drawdown_pct": 40.0,
    "min_win_rate_pct": 40.0,
    "min_trades": 30,
    "fee_pct": 0.00055,
    "spread_pct": 0.00040,
    "slippage_pct": 0.00030,
    "max_positions": 3,
    "risk_per_trade": 0.02,
    "enable_global_rules": True,
    "global_rules_profile": "balanced",
    "min_24h_notional_usdt": 50_000_000.0,
    "enable_mtf_context_filter": False,
    "enable_regime_filter": False,
    "context_fast_period": 20,
    "context_slow_period": 50,
    "regime_symbol": "",
    "replay_top_n": 30,
    "require_replay_pass": True,
    "enable_train_test_split": True,
    "train_ratio": 0.70,
    "min_test_return_pct": 0.0,
    "max_test_drawdown_pct": 25.0,
    "min_test_win_rate_pct": 40.0,
    "min_test_trades": 10,
    "enable_gate_fallback": True,
    "fallback_return_relax_factor": 0.50,
    "fallback_drawdown_relax_factor": 1.35,
    "fallback_win_rate_relax_delta": 5.0,
    "fallback_trades_relax_factor": 0.60,
}


QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "A": {
        "min_return_pct": 5.0,
        "max_drawdown_pct": 15.0,
        "min_win_rate_pct": 45.0,
        "min_trades": 40,
        "min_test_return_pct": 1.0,
        "max_test_drawdown_pct": 20.0,
        "min_test_win_rate_pct": 42.0,
        "min_test_trades": 15,
    },
    "B": {
        "min_return_pct": 10.0,
        "max_drawdown_pct": 10.0,
        "min_win_rate_pct": 50.0,
        "min_trades": 60,
        "min_test_return_pct": 3.0,
        "max_test_drawdown_pct": 15.0,
        "min_test_win_rate_pct": 47.0,
        "min_test_trades": 20,
    },
}


def _build_advanced_search_space(base_space: Dict[str, Any]) -> Dict[str, Any]:
    advanced_overlay: Dict[str, Any] = {
        "signal_combination_methods": ["majority_vote", "and_signals"],
        "indicators": {
            "macd_main": {
                "function": "macd",
                "params": {
                    "fast_period": {"type": "int", "low": 6, "high": 18, "step": 1},
                    "slow_period": {"type": "int", "low": 20, "high": 50, "step": 1},
                    "signal_period": {"type": "int", "low": 4, "high": 14, "step": 1},
                },
            },
            "bb_main": {
                "function": "bollinger_bands",
                "params": {
                    "period": {"type": "int", "low": 14, "high": 48, "step": 1},
                    "std_dev": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.1},
                },
            },
            "stoch_main": {
                "function": "stochastic",
                "params": {
                    "k_period": {"type": "int", "low": 8, "high": 24, "step": 1},
                    "d_period": {"type": "int", "low": 2, "high": 8, "step": 1},
                },
            },
            "resistance_main": {
                "function": "highest",
                "params": {
                    "period": {"type": "int", "low": 10, "high": 60, "step": 1},
                },
            },
            "support_main": {
                "function": "lowest",
                "params": {
                    "period": {"type": "int", "low": 10, "high": 60, "step": 1},
                },
            },
        },
        "signals": {
            "macd_cross": {
                "function": "macd_signals",
                "inputs": {
                    "macd_line": "macd_line",
                    "signal_line": "signal_line",
                },
                "params": {},
            },
            "bb_reversion": {
                "function": "bollinger_bands_signals",
                "inputs": {
                    "price": "price",
                    "upper_band": "upper_band",
                    "lower_band": "lower_band",
                },
                "params": {},
            },
            "stoch_reversal": {
                "function": "stochastic_signals",
                "inputs": {
                    "k_percent": "k_percent",
                    "d_percent": "d_percent",
                },
                "params": {
                    "overbought": {"type": "int", "low": 75, "high": 90, "step": 1},
                    "oversold": {"type": "int", "low": 10, "high": 25, "step": 1},
                },
            },
            "breakout_sr": {
                "function": "breakout_signals",
                "inputs": {
                    "price": "price",
                    "resistance": "resistance_main",
                    "support": "support_main",
                },
                "params": {
                    "penetration_pct": {"type": "float", "low": 0.001, "high": 0.02, "step": 0.001},
                },
            },
            "rsi_divergence": {
                "function": "divergence_signals",
                "inputs": {
                    "price": "price",
                    "indicator": "rsi_main",
                },
                "params": {
                    "lookback_period": {"type": "int", "low": 8, "high": 40, "step": 1},
                },
            },
        },
    }
    return _deep_merge(base_space, advanced_overlay)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _has_non_empty_data(data: Dict[str, Dict[str, Any]]) -> bool:
    for by_timeframe in data.values():
        for frame in by_timeframe.values():
            if frame is not None and not frame.empty:
                return True
    return False


def _split_data_for_train_test(
    data: Dict[str, Dict[str, pd.DataFrame]],
    train_ratio: float,
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, pd.DataFrame]]]:
    ratio = min(0.95, max(0.50, float(train_ratio)))
    train_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    test_data: Dict[str, Dict[str, pd.DataFrame]] = {}

    for symbol, by_timeframe in data.items():
        train_data[symbol] = {}
        test_data[symbol] = {}
        for timeframe, frame in by_timeframe.items():
            if frame is None or frame.empty:
                train_data[symbol][timeframe] = frame
                test_data[symbol][timeframe] = frame
                continue

            if len(frame) < 2:
                train_data[symbol][timeframe] = frame.copy()
                test_data[symbol][timeframe] = frame.copy()
                continue

            split_idx = int(len(frame) * ratio)
            split_idx = max(1, min(len(frame) - 1, split_idx))
            train_data[symbol][timeframe] = frame.iloc[:split_idx].copy()
            test_data[symbol][timeframe] = frame.iloc[split_idx:].copy()

    return train_data, test_data


def _date_range_for_data(data: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[str, str]:
    starts: List[pd.Timestamp] = []
    ends: List[pd.Timestamp] = []

    for by_timeframe in data.values():
        for frame in by_timeframe.values():
            if frame is None or frame.empty:
                continue
            starts.append(pd.to_datetime(frame.index.min()))
            ends.append(pd.to_datetime(frame.index.max()))

    if not starts or not ends:
        raise RuntimeError("No valid candles found while building train/test date windows.")

    start = min(starts).strftime("%Y-%m-%d %H:%M:%S")
    end = max(ends).strftime("%Y-%m-%d %H:%M:%S")
    return start, end


def _tf_to_minutes(timeframe: str) -> int:
    text = str(timeframe).strip().lower()
    try:
        if text.endswith("m"):
            return int(text[:-1])
        if text.endswith("h"):
            return int(text[:-1]) * 60
        if text.endswith("d"):
            return int(text[:-1]) * 1440
        return int(text)
    except Exception:
        return 10_000_000


def _context_trend_series(frame: pd.DataFrame, fast_period: int, slow_period: int) -> pd.Series:
    close = frame["close"].astype(float)
    fast = close.ewm(span=max(2, int(fast_period)), adjust=False, min_periods=max(2, int(fast_period))).mean()
    slow = close.ewm(span=max(3, int(slow_period)), adjust=False, min_periods=max(3, int(slow_period))).mean()
    return (fast > slow).astype("boolean")


def _apply_context_filters(
    signals_map: Dict[str, Dict[str, pd.Series]],
    data_map: Dict[str, Dict[str, pd.DataFrame]],
    run_config: Dict[str, Any],
) -> Dict[str, Dict[str, pd.Series]]:
    mtf_enabled = bool(run_config.get("enable_mtf_context_filter", False))
    regime_enabled = bool(run_config.get("enable_regime_filter", False))
    if not mtf_enabled and not regime_enabled:
        return signals_map

    configured_timeframes = list(run_config.get("timeframes", []))
    if not configured_timeframes:
        return signals_map

    sorted_timeframes = sorted(configured_timeframes, key=_tf_to_minutes)
    context_timeframe = sorted_timeframes[-1]
    fast_period = int(run_config.get("context_fast_period", 20))
    slow_period = int(run_config.get("context_slow_period", 50))
    if fast_period >= slow_period:
        slow_period = fast_period + 1

    local_trends: Dict[str, pd.Series] = {}
    for symbol, by_timeframe in data_map.items():
        frame = by_timeframe.get(context_timeframe)
        if frame is None or frame.empty or "close" not in frame.columns:
            continue
        local_trends[symbol] = _context_trend_series(frame, fast_period=fast_period, slow_period=slow_period)

    regime_trend = None
    if regime_enabled:
        regime_symbol = str(run_config.get("regime_symbol", "")).strip()
        if not regime_symbol:
            symbols = list(run_config.get("symbols", []))
            regime_symbol = symbols[0] if symbols else ""
        regime_frame = data_map.get(regime_symbol, {}).get(context_timeframe)
        if regime_frame is not None and not regime_frame.empty and "close" in regime_frame.columns:
            regime_trend = _context_trend_series(regime_frame, fast_period=fast_period, slow_period=slow_period)

    for symbol, by_timeframe in signals_map.items():
        for timeframe, signal_series in by_timeframe.items():
            if not isinstance(signal_series, pd.Series) or signal_series.empty:
                continue

            allow_long = np.ones(len(signal_series), dtype=bool)
            allow_short = np.ones(len(signal_series), dtype=bool)
            has_filter = False

            if mtf_enabled and _tf_to_minutes(timeframe) < _tf_to_minutes(context_timeframe):
                local = local_trends.get(symbol)
                if local is not None and not local.empty:
                    local_aligned = local.reindex(signal_series.index, method="ffill").fillna(False).to_numpy(dtype=bool)
                    allow_long &= local_aligned
                    allow_short &= ~local_aligned
                    has_filter = True

            if regime_enabled and regime_trend is not None and not regime_trend.empty:
                regime_aligned = regime_trend.reindex(signal_series.index, method="ffill").fillna(False).to_numpy(dtype=bool)
                allow_long &= regime_aligned
                allow_short &= ~regime_aligned
                has_filter = True

            if not has_filter:
                continue

            raw = signal_series.astype(str).to_numpy(copy=True)
            raw = np.where((raw == "OPEN_LONG") & (~allow_long), "HOLD", raw)
            raw = np.where((raw == "OPEN_SHORT") & (~allow_short), "HOLD", raw)
            by_timeframe[timeframe] = pd.Series(raw, index=signal_series.index)

    return signals_map


def _attach_context_signal_filters(strategy: Any, run_config: Dict[str, Any]) -> Any:
    if not hasattr(strategy, "generate_signals_vectorized"):
        return strategy

    mtf_enabled = bool(run_config.get("enable_mtf_context_filter", False))
    regime_enabled = bool(run_config.get("enable_regime_filter", False))
    if not mtf_enabled and not regime_enabled:
        return strategy

    original_generate = strategy.generate_signals_vectorized

    def _wrapped_generate(data_map: Dict[str, Dict[str, pd.DataFrame]]):
        signals_map = original_generate(data_map)
        try:
            return _apply_context_filters(signals_map, data_map, run_config)
        except Exception:
            return signals_map

    strategy.generate_signals_vectorized = _wrapped_generate
    return strategy


def _score(metrics: Dict[str, float]) -> float:
    return (
        float(metrics.get("total_return", 0.0))
        - (0.60 * float(metrics.get("max_drawdown", 0.0)))
        + (0.25 * float(metrics.get("sharpe_ratio", 0.0)))
        + (0.10 * float(metrics.get("win_rate", 0.0)))
        + min(float(metrics.get("total_trades", 0.0)), 300.0) / 60.0
    )


def _format_duration_hms(total_seconds: float) -> str:
    seconds = max(0, int(round(float(total_seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h {minutes}m {secs}s"


def _build_backtest_config(run_config: Dict[str, Any]) -> Dict[str, Any]:
    profile = str(run_config.get("global_rules_profile", "balanced")).strip().lower() or "balanced"
    global_rules_payload = run_config.get("global_rules")
    if isinstance(global_rules_payload, dict):
        global_rules = dict(global_rules_payload)
    else:
        global_rules = {}
    if "min_24h_notional_usdt" not in global_rules:
        global_rules["min_24h_notional_usdt"] = float(run_config.get("min_24h_notional_usdt", 50_000_000.0))

    return {
        "fee_pct": float(run_config.get("fee_pct", 0.00055)),
        "spread_pct": float(run_config.get("spread_pct", 0.00040)),
        "slippage_pct": float(run_config.get("slippage_pct", 0.00030)),
        "max_positions": int(run_config.get("max_positions", 3)),
        "risk_per_trade": float(run_config.get("risk_per_trade", 0.02)),
        "enable_global_rules": bool(run_config.get("enable_global_rules", False)),
        "global_rules_profile": profile,
        "global_rules": global_rules,
        "debug_signals": False,
        "print_trade_logs": False,
    }


def _passes_thresholds(metrics: Dict[str, float], config: Dict[str, Any]) -> bool:
    return (
        float(metrics.get("total_return", 0.0)) >= float(config.get("min_return_pct", 0.0))
        and float(metrics.get("max_drawdown", 0.0)) <= float(config.get("max_drawdown_pct", 100.0))
        and float(metrics.get("win_rate", 0.0)) >= float(config.get("min_win_rate_pct", 0.0))
        and int(metrics.get("total_trades", 0)) >= int(config.get("min_trades", 0))
    )


def _build_test_gate_config(run_config: Dict[str, Any]) -> Dict[str, float]:
    return {
        "min_return_pct": float(run_config.get("min_test_return_pct", 0.0)),
        "max_drawdown_pct": float(run_config.get("max_test_drawdown_pct", 100.0)),
        "min_win_rate_pct": float(run_config.get("min_test_win_rate_pct", 0.0)),
        "min_trades": int(run_config.get("min_test_trades", 0)),
    }


def _build_relaxed_gate_config(run_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **run_config,
        "min_return_pct": float(run_config.get("min_return_pct", 0.0))
        * float(run_config.get("fallback_return_relax_factor", 0.50)),
        "max_drawdown_pct": float(run_config.get("max_drawdown_pct", 100.0))
        * float(run_config.get("fallback_drawdown_relax_factor", 1.35)),
        "min_win_rate_pct": max(
            0.0,
            float(run_config.get("min_win_rate_pct", 0.0))
            - float(run_config.get("fallback_win_rate_relax_delta", 5.0)),
        ),
        "min_trades": max(
            0,
            int(
                float(run_config.get("min_trades", 0))
                * float(run_config.get("fallback_trades_relax_factor", 0.60))
            ),
        ),
        "min_test_return_pct": float(run_config.get("min_test_return_pct", 0.0))
        * float(run_config.get("fallback_return_relax_factor", 0.50)),
        "max_test_drawdown_pct": float(run_config.get("max_test_drawdown_pct", 100.0))
        * float(run_config.get("fallback_drawdown_relax_factor", 1.35)),
        "min_test_win_rate_pct": max(
            0.0,
            float(run_config.get("min_test_win_rate_pct", 0.0))
            - float(run_config.get("fallback_win_rate_relax_delta", 5.0)),
        ),
        "min_test_trades": max(
            0,
            int(
                float(run_config.get("min_test_trades", 0))
                * float(run_config.get("fallback_trades_relax_factor", 0.60))
            ),
        ),
    }


def _rank_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("replay_score", row.get("score", -1e9))),
            -float(row.get("score", -1e9)),
            int(row.get("index", 10**9)),
        ),
    )


def _run_replay_for_row(
    row: Dict[str, Any],
    search_space: Dict[str, Any],
    run_config: Dict[str, Any],
    full_data: Dict[str, Dict[str, pd.DataFrame]],
    feeder: DataFeeder,
) -> Dict[str, Any]:
    local_builder = CandidateBuilder(
        search_space=search_space,
        max_active_signals=int(run_config.get("max_active_signals", 3)),
    )
    strategy = local_builder.build_strategy(
        candidate=row.get("candidate", {}),
        symbols=run_config["symbols"],
        timeframes=run_config["timeframes"],
        strategy_name=f"FAST_REPLAY_{int(row.get('index', -1)):05d}",
    )
    strategy = _attach_context_signal_filters(strategy, run_config)

    backtest_config = _build_backtest_config(run_config)
    engine = BacktesterEngine(data_feeder=feeder, strategy=strategy, config=backtest_config)
    engine._save_backtest_results = lambda *args, **kwargs: None
    result = engine.run_backtest(
        symbols=run_config["symbols"],
        timeframes=run_config["timeframes"],
        start_date=str(run_config["start_date"]),
        end_date=str(run_config["end_date"]),
        config=backtest_config,
        strategy=strategy,
        data=full_data,
        initial_balance=float(run_config.get("initial_balance", 10000.0)),
    )

    replay_metrics = {
        "total_return": float(result.get("total_return", 0.0)),
        "max_drawdown": float(result.get("max_drawdown", 0.0)),
        "win_rate": float(result.get("win_rate", 0.0)),
        "total_trades": int(result.get("total_trades", 0)),
        "sharpe_ratio": float(result.get("sharpe_ratio", 0.0)),
    }
    replay_passed = _passes_thresholds(replay_metrics, run_config)
    return {
        "replay_metrics": replay_metrics,
        "replay_score": _score(replay_metrics),
        "replay_passed": bool(replay_passed),
    }


def _replay_top_candidates(
    all_results: List[Dict[str, Any]],
    search_space: Dict[str, Any],
    run_config: Dict[str, Any],
    full_data: Dict[str, Dict[str, pd.DataFrame]],
    feeder: DataFeeder,
    workers: int,
    top_k: int,
) -> None:
    if not all_results:
        return

    ranked = _rank_rows(all_results)
    replay_top_n = max(top_k, int(run_config.get("replay_top_n", top_k)))
    replay_rows = ranked[: min(len(ranked), replay_top_n)]
    if not replay_rows:
        return

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _run_replay_for_row,
                row=row,
                search_space=search_space,
                run_config=run_config,
                full_data=full_data,
                feeder=feeder,
            ): row
            for row in replay_rows
        }

        for future, row in futures.items():
            try:
                replay_result = future.result()
            except Exception:
                replay_result = {
                    "replay_metrics": {
                        "total_return": 0.0,
                        "max_drawdown": 100.0,
                        "win_rate": 0.0,
                        "total_trades": 0,
                        "sharpe_ratio": 0.0,
                    },
                    "replay_score": -1e9,
                    "replay_passed": False,
                }
            row.update(replay_result)


def _apply_quality_preset(
    run_config: Dict[str, Any],
    explicit_override: Dict[str, Any],
) -> Dict[str, Any]:
    preset_raw = str(run_config.get("quality_preset", "custom")).strip().upper()
    if preset_raw not in QUALITY_PRESETS:
        return run_config

    updated = deepcopy(run_config)
    preset_values = QUALITY_PRESETS[preset_raw]
    for key, value in preset_values.items():
        if key not in explicit_override:
            updated[key] = value
    return updated


def _evaluate_candidate(
    index: int,
    genome: List[float],
    search_space: Dict[str, Any],
    run_config: Dict[str, Any],
    train_data: Dict[str, Dict[str, Any]],
    test_data: Dict[str, Dict[str, Any]],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    feeder: DataFeeder,
) -> Dict[str, Any]:
    def _run_backtest_for_candidate(
        strategy_obj: Any,
        data_window: Dict[str, Dict[str, Any]],
        start_date: str,
        end_date: str,
    ) -> Dict[str, float]:
        backtest_config = _build_backtest_config(run_config)

        engine = BacktesterEngine(data_feeder=feeder, strategy=strategy_obj, config=backtest_config)
        engine._save_backtest_results = lambda *args, **kwargs: None
        result = engine.run_backtest(
            symbols=run_config["symbols"],
            timeframes=run_config["timeframes"],
            start_date=start_date,
            end_date=end_date,
            config=backtest_config,
            strategy=strategy_obj,
            data=data_window,
            initial_balance=float(run_config.get("initial_balance", 10000.0)),
        )

        return {
            "total_return": float(result.get("total_return", 0.0)),
            "max_drawdown": float(result.get("max_drawdown", 0.0)),
            "win_rate": float(result.get("win_rate", 0.0)),
            "total_trades": int(result.get("total_trades", 0)),
            "sharpe_ratio": float(result.get("sharpe_ratio", 0.0)),
        }

    local_builder = CandidateBuilder(
        search_space=search_space,
        max_active_signals=int(run_config.get("max_active_signals", 3)),
    )
    candidate = local_builder.decode(genome)

    train_strategy = local_builder.build_strategy(
        candidate=candidate,
        symbols=run_config["symbols"],
        timeframes=run_config["timeframes"],
        strategy_name=f"FAST_CAND_{index:05d}",
    )
    train_strategy = _attach_context_signal_filters(train_strategy, run_config)
    train_metrics = _run_backtest_for_candidate(
        strategy_obj=train_strategy,
        data_window=train_data,
        start_date=train_start,
        end_date=train_end,
    )

    if bool(run_config.get("enable_train_test_split", True)):
        test_strategy = local_builder.build_strategy(
            candidate=candidate,
            symbols=run_config["symbols"],
            timeframes=run_config["timeframes"],
            strategy_name=f"FAST_CAND_{index:05d}_TEST",
        )
        test_strategy = _attach_context_signal_filters(test_strategy, run_config)
        test_metrics = _run_backtest_for_candidate(
            strategy_obj=test_strategy,
            data_window=test_data,
            start_date=test_start,
            end_date=test_end,
        )
    else:
        test_metrics = dict(train_metrics)

    train_passed = _passes_thresholds(train_metrics, run_config)
    test_gate_config = _build_test_gate_config(run_config)
    test_passed = _passes_thresholds(test_metrics, test_gate_config)
    score = (0.40 * _score(train_metrics)) + (0.60 * _score(test_metrics))
    passed = bool(train_passed and test_passed)

    return {
        "index": index,
        "score": score,
        "passed": passed,
        "passed_relaxed": False,
        "train_passed": bool(train_passed),
        "test_passed": bool(test_passed),
        "candidate": candidate,
        "metrics": test_metrics,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


def _select_top_results(
    all_results: List[Dict[str, Any]],
    top_k: int,
    run_config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ranked = _rank_rows(all_results)
    selected: List[Dict[str, Any]] = []
    selected_row_ids: Set[int] = set()
    require_replay = bool(run_config.get("require_replay_pass", True))

    def _row_replay_allowed(row: Dict[str, Any]) -> bool:
        if not require_replay:
            return True
        return "replay_passed" in row

    def _row_replay_ok(row: Dict[str, Any]) -> bool:
        if not require_replay:
            return True
        return bool(row.get("replay_passed", False))

    fallback_info: Dict[str, Any] = {
        "triggered": False,
        "reason": "",
        "strict_pass_count": 0,
        "relaxed_pass_count": 0,
    }

    strict_rows = [row for row in ranked if bool(row.get("passed", False)) and _row_replay_ok(row)]
    fallback_info["strict_pass_count"] = len(strict_rows)
    if len(strict_rows) < top_k:
        fallback_info["triggered"] = True
        fallback_info["reason"] = "strict_gate_zero_passes" if not strict_rows else "strict_gate_partial"
    for row in strict_rows:
        row_id = id(row)
        if row_id in selected_row_ids:
            continue
        selected.append(row)
        selected_row_ids.add(row_id)
        if len(selected) >= top_k:
            return selected[:top_k], fallback_info

    if bool(run_config.get("enable_gate_fallback", True)):
        relaxed_config = _build_relaxed_gate_config(run_config)
        relaxed_test_gate = _build_test_gate_config(relaxed_config)
        relaxed_rows: List[Dict[str, Any]] = []
        for row in ranked:
            if not _row_replay_allowed(row):
                continue
            train_metrics = row.get("train_metrics", row.get("metrics", {}))
            test_metrics = row.get("test_metrics", row.get("metrics", {}))
            relaxed_train_ok = _passes_thresholds(train_metrics, relaxed_config)
            relaxed_test_ok = _passes_thresholds(test_metrics, relaxed_test_gate)
            relaxed_ok = bool(relaxed_train_ok and relaxed_test_ok and _row_replay_ok(row))
            row["passed_relaxed"] = relaxed_ok
            if relaxed_ok:
                relaxed_rows.append(row)

        fallback_info["relaxed_pass_count"] = len(relaxed_rows)
        if len(strict_rows) < top_k and relaxed_rows:
            fallback_info["triggered"] = True
            fallback_info["reason"] = "strict_gate_too_tight"
            for row in relaxed_rows:
                row_id = id(row)
                if row_id in selected_row_ids:
                    continue
                selected.append(row)
                selected_row_ids.add(row_id)
                if len(selected) >= top_k:
                    return selected[:top_k], fallback_info

    for row in ranked:
        if not _row_replay_allowed(row):
            continue
        row_id = id(row)
        if row_id in selected_row_ids:
            continue
        selected.append(row)
        selected_row_ids.add(row_id)
        if len(selected) >= top_k:
            break

    return selected[:top_k], fallback_info


def _write_reports(
    run_dir: Path,
    run_config: Dict[str, Any],
    all_results: List[Dict[str, Any]],
    top_results: List[Dict[str, Any]],
    elapsed_seconds: float,
    stopped: bool,
    fallback_info: Dict[str, Any],
) -> None:
    elapsed_hms = _format_duration_hms(elapsed_seconds)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "stopped": bool(stopped),
        "elapsed_seconds": float(elapsed_seconds),
        "elapsed_hms": elapsed_hms,
        "config": run_config,
        "evaluated": len(all_results),
        "fallback": fallback_info,
        "top_results": top_results,
    }
    (run_dir / "top_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_lines = []
    summary_lines.append("Fast Lego Strategy Finder Summary")
    summary_lines.append("=================================")
    summary_lines.append(f"Generated at: {payload['generated_at']}")
    summary_lines.append(f"Stopped: {stopped}")
    summary_lines.append(f"Elapsed: {elapsed_hms}")
    summary_lines.append(f"Evaluated: {len(all_results)}")
    summary_lines.append(
        "Fallback: triggered={triggered} reason={reason} strict={strict} relaxed={relaxed}".format(
            triggered=bool(fallback_info.get("triggered", False)),
            reason=str(fallback_info.get("reason", "")),
            strict=int(fallback_info.get("strict_pass_count", 0)),
            relaxed=int(fallback_info.get("relaxed_pass_count", 0)),
        )
    )
    summary_lines.append("")
    for rank, row in enumerate(top_results, start=1):
        metrics = row.get("metrics", {})
        train_metrics = row.get("train_metrics", {})
        replay_metrics = row.get("replay_metrics", {})
        signals = row.get("candidate", {}).get("active_signals", [])
        summary_lines.append(
            "#{rank} score={score:.4f} pass={passed} pass_relaxed={pass_relaxed} "
            "test_return={ret:.2f}% test_dd={dd:.2f}% test_win={win:.2f}% test_trades={trades} "
            "train_return={train_ret:.2f}% train_dd={train_dd:.2f}% "
            "replay_pass={replay_pass} replay_return={replay_ret:.2f}% replay_dd={replay_dd:.2f}% replay_trades={replay_trades} "
            "signals={signals}".format(
                rank=rank,
                score=float(row.get("score", 0.0)),
                passed=bool(row.get("passed", False)),
                pass_relaxed=bool(row.get("passed_relaxed", False)),
                ret=float(metrics.get("total_return", 0.0)),
                dd=float(metrics.get("max_drawdown", 0.0)),
                win=float(metrics.get("win_rate", 0.0)),
                trades=int(metrics.get("total_trades", 0)),
                train_ret=float(train_metrics.get("total_return", 0.0)),
                train_dd=float(train_metrics.get("max_drawdown", 0.0)),
                replay_pass=bool(row.get("replay_passed", False)),
                replay_ret=float(replay_metrics.get("total_return", 0.0)),
                replay_dd=float(replay_metrics.get("max_drawdown", 0.0)),
                replay_trades=int(replay_metrics.get("total_trades", 0)),
                signals=signals,
            )
        )
    (run_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    csv_path = run_dir / "top_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rank",
                "score",
                "passed",
                "passed_relaxed",
                "train_passed",
                "test_passed",
                "return_pct",
                "max_drawdown_pct",
                "win_rate_pct",
                "trades",
                "sharpe_ratio",
                "train_return_pct",
                "train_drawdown_pct",
                "train_win_rate_pct",
                "train_trades",
                "replay_passed",
                "replay_score",
                "replay_return_pct",
                "replay_drawdown_pct",
                "replay_win_rate_pct",
                "replay_trades",
                "signals",
                "combination",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(top_results, start=1):
            metrics = row.get("metrics", {})
            train_metrics = row.get("train_metrics", {})
            replay_metrics = row.get("replay_metrics", {})
            candidate = row.get("candidate", {})
            writer.writerow(
                {
                    "rank": rank,
                    "score": float(row.get("score", 0.0)),
                    "passed": bool(row.get("passed", False)),
                    "passed_relaxed": bool(row.get("passed_relaxed", False)),
                    "train_passed": bool(row.get("train_passed", False)),
                    "test_passed": bool(row.get("test_passed", False)),
                    "return_pct": float(metrics.get("total_return", 0.0)),
                    "max_drawdown_pct": float(metrics.get("max_drawdown", 0.0)),
                    "win_rate_pct": float(metrics.get("win_rate", 0.0)),
                    "trades": int(metrics.get("total_trades", 0)),
                    "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
                    "train_return_pct": float(train_metrics.get("total_return", 0.0)),
                    "train_drawdown_pct": float(train_metrics.get("max_drawdown", 0.0)),
                    "train_win_rate_pct": float(train_metrics.get("win_rate", 0.0)),
                    "train_trades": int(train_metrics.get("total_trades", 0)),
                    "replay_passed": bool(row.get("replay_passed", False)),
                    "replay_score": float(row.get("replay_score", 0.0)),
                    "replay_return_pct": float(replay_metrics.get("total_return", 0.0)),
                    "replay_drawdown_pct": float(replay_metrics.get("max_drawdown", 0.0)),
                    "replay_win_rate_pct": float(replay_metrics.get("win_rate", 0.0)),
                    "replay_trades": int(replay_metrics.get("total_trades", 0)),
                    "signals": ",".join(candidate.get("active_signals", [])),
                    "combination": candidate.get("signal_combination", "majority_vote"),
                }
            )


def _export_strategy_files(
    run_dir: Path,
    top_results: List[Dict[str, Any]],
    run_config: Dict[str, Any],
    candidate_builder: CandidateBuilder,
) -> List[str]:
    run_suffix = run_dir.name.replace("run_", "")
    strategies_dir = run_dir / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[2]
    publish_dir = project_root / "simple_strategy" / "strategies"
    publish_dir.mkdir(parents=True, exist_ok=True)
    publish_top_n = max(0, int(run_config.get("publish_top_n", 0)))

    published_files: List[str] = []
    published_count = 0
    require_replay = bool(run_config.get("require_replay_pass", True))
    for rank, row in enumerate(top_results, start=1):
        module_name = f"Strategy_FAST_{run_suffix}_{rank:02d}"
        module_text = render_strategy_module(
            strategy_name=module_name,
            candidate=row.get("candidate", {}),
            candidate_builder=candidate_builder,
            symbols=run_config["symbols"],
            timeframes=run_config["timeframes"],
        )
        run_module_path = strategies_dir / f"{module_name}.py"
        run_module_path.write_text(module_text, encoding="utf-8")

        replay_ok = bool(row.get("replay_passed", False)) if require_replay else True
        qualifies_for_publish = bool((row.get("passed", False) or row.get("passed_relaxed", False)) and replay_ok)
        if qualifies_for_publish and published_count < publish_top_n:
            target_path = publish_dir / f"{module_name}.py"
            target_path.write_text(module_text, encoding="utf-8")
            published_files.append(str(target_path))
            published_count += 1

    return published_files


def run_fast_finder(
    config_override: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, int, float, Dict[str, Any]], None]] = None,
    stop_event: Optional[Any] = None,
) -> Dict[str, Any]:
    override = config_override or {}
    run_config = _deep_merge(DEFAULT_FAST_CONFIG, override)
    run_config = _apply_quality_preset(run_config, explicit_override=override)
    run_config["symbols"] = _as_list(run_config.get("symbols", []))
    run_config["timeframes"] = _as_list(run_config.get("timeframes", []))

    if not run_config["symbols"]:
        raise ValueError("At least one symbol is required.")
    if not run_config["timeframes"]:
        raise ValueError("At least one timeframe is required.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(run_config["output_dir"]) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    search_profile = str(run_config.get("search_profile", "basic")).strip().lower()
    search_space_path = str(run_config.get("search_space_path", "")).strip()
    base_space = load_search_space(search_space_path)

    if not search_space_path and search_profile == "advanced":
        search_space = _build_advanced_search_space(base_space)
        if "max_active_signals" not in override:
            run_config["max_active_signals"] = max(4, int(run_config.get("max_active_signals", 3)))
        if "enable_mtf_context_filter" not in override:
            run_config["enable_mtf_context_filter"] = True
        if "enable_regime_filter" not in override:
            run_config["enable_regime_filter"] = True
    else:
        search_space = base_space

    candidate_builder = CandidateBuilder(
        search_space=search_space,
        max_active_signals=int(run_config.get("max_active_signals", 3)),
    )
    feeder = DataFeeder(data_dir=run_config.get("data_dir", "data"))
    data = feeder.get_data_for_symbols(
        run_config["symbols"],
        run_config["timeframes"],
        run_config["start_date"],
        run_config["end_date"],
    )
    if not _has_non_empty_data(data):
        raise RuntimeError("No data found for selected symbols/timeframes/date range.")

    train_data = data
    test_data = data
    train_start = str(run_config["start_date"])
    train_end = str(run_config["end_date"])
    test_start = train_start
    test_end = train_end

    if bool(run_config.get("enable_train_test_split", True)):
        split_ratio = float(run_config.get("train_ratio", 0.70))
        split_train_data, split_test_data = _split_data_for_train_test(data, train_ratio=split_ratio)
        if _has_non_empty_data(split_train_data) and _has_non_empty_data(split_test_data):
            train_data = split_train_data
            test_data = split_test_data
            train_start, train_end = _date_range_for_data(train_data)
            test_start, test_end = _date_range_for_data(test_data)
        else:
            run_config["enable_train_test_split"] = False

    rng = random.Random(int(run_config.get("seed", 42)))
    candidate_count = max(1, int(run_config.get("candidate_count", 1)))
    top_k = max(1, int(run_config.get("top_k", 1)))
    workers = max(1, int(run_config.get("workers", 1)))

    genomes = [candidate_builder.random_genome(rng) for _ in range(candidate_count)]

    all_results: List[Dict[str, Any]] = []
    best_score = -1e9
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: Dict[Any, int] = {}
        next_index = 0
        completed = 0

        def _submit(idx: int) -> None:
            futures[
                executor.submit(
                    _evaluate_candidate,
                    idx,
                    genomes[idx],
                    search_space,
                    run_config,
                    train_data,
                    test_data,
                    train_start,
                    train_end,
                    test_start,
                    test_end,
                    feeder,
                )
            ] = idx

        for _ in range(min(workers, candidate_count)):
            _submit(next_index)
            next_index += 1

        while futures:
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future, None)
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "index": -1,
                        "score": -1e9,
                        "passed": False,
                        "passed_relaxed": False,
                        "train_passed": False,
                        "test_passed": False,
                        "candidate": {},
                        "metrics": {
                            "total_return": 0.0,
                            "max_drawdown": 100.0,
                            "win_rate": 0.0,
                            "total_trades": 0,
                            "sharpe_ratio": 0.0,
                        },
                        "train_metrics": {
                            "total_return": 0.0,
                            "max_drawdown": 100.0,
                            "win_rate": 0.0,
                            "total_trades": 0,
                            "sharpe_ratio": 0.0,
                        },
                        "test_metrics": {
                            "total_return": 0.0,
                            "max_drawdown": 100.0,
                            "win_rate": 0.0,
                            "total_trades": 0,
                            "sharpe_ratio": 0.0,
                        },
                        "error": str(exc),
                    }

                completed += 1
                all_results.append(row)
                best_score = max(best_score, float(row.get("score", -1e9)))
                if progress_callback is not None:
                    progress_callback(completed, candidate_count, best_score, row)

                stopped = bool(stop_event is not None and getattr(stop_event, "is_set", lambda: False)())
                if stopped:
                    for pending in list(futures.keys()):
                        pending.cancel()
                    futures.clear()
                    break

                if next_index < candidate_count:
                    _submit(next_index)
                    next_index += 1

            if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
                break

    elapsed = time.perf_counter() - started
    _replay_top_candidates(
        all_results=all_results,
        search_space=search_space,
        run_config=run_config,
        full_data=data,
        feeder=feeder,
        workers=workers,
        top_k=top_k,
    )
    top_results, fallback_info = _select_top_results(
        all_results=all_results,
        top_k=top_k,
        run_config=run_config,
    )

    _write_reports(
        run_dir=run_dir,
        run_config=run_config,
        all_results=all_results,
        top_results=top_results,
        elapsed_seconds=elapsed,
        stopped=bool(stop_event is not None and getattr(stop_event, "is_set", lambda: False)()),
        fallback_info=fallback_info,
    )
    published_files = _export_strategy_files(
        run_dir=run_dir,
        top_results=top_results,
        run_config=run_config,
        candidate_builder=candidate_builder,
    )

    return {
        "run_dir": str(run_dir),
        "elapsed_seconds": float(elapsed),
        "evaluated": len(all_results),
        "requested_candidates": candidate_count,
        "top_results": top_results,
        "fallback": fallback_info,
        "published_strategy_files": published_files,
        "stopped": bool(stop_event is not None and getattr(stop_event, "is_set", lambda: False)()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast lego strategy finder")
    parser.add_argument("--quality-preset", choices=["custom", "A", "B"], default=DEFAULT_FAST_CONFIG["quality_preset"])
    parser.add_argument("--search-profile", choices=["basic", "advanced"], default=DEFAULT_FAST_CONFIG["search_profile"])
    parser.add_argument("--symbols", default="BNBUSDT,XRPUSDT,ADAUSDT,DOTUSDT,NEARUSDT")
    parser.add_argument("--timeframes", default="1,5")
    parser.add_argument("--start-date", default=DEFAULT_FAST_CONFIG["start_date"])
    parser.add_argument("--end-date", default=DEFAULT_FAST_CONFIG["end_date"])
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_FAST_CONFIG["candidate_count"])
    parser.add_argument("--workers", type=int, default=DEFAULT_FAST_CONFIG["workers"])
    parser.add_argument("--top-k", type=int, default=DEFAULT_FAST_CONFIG["top_k"])
    parser.add_argument("--publish-top-n", type=int, default=DEFAULT_FAST_CONFIG["publish_top_n"])
    parser.add_argument("--min-return-pct", type=float, default=DEFAULT_FAST_CONFIG["min_return_pct"])
    parser.add_argument("--max-drawdown-pct", type=float, default=DEFAULT_FAST_CONFIG["max_drawdown_pct"])
    parser.add_argument("--min-win-rate-pct", type=float, default=DEFAULT_FAST_CONFIG["min_win_rate_pct"])
    parser.add_argument("--min-trades", type=int, default=DEFAULT_FAST_CONFIG["min_trades"])
    parser.add_argument("--min-test-return-pct", type=float, default=DEFAULT_FAST_CONFIG["min_test_return_pct"])
    parser.add_argument("--max-test-drawdown-pct", type=float, default=DEFAULT_FAST_CONFIG["max_test_drawdown_pct"])
    parser.add_argument("--min-test-win-rate-pct", type=float, default=DEFAULT_FAST_CONFIG["min_test_win_rate_pct"])
    parser.add_argument("--min-test-trades", type=int, default=DEFAULT_FAST_CONFIG["min_test_trades"])
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_FAST_CONFIG["train_ratio"])
    parser.add_argument("--disable-train-test-split", action="store_true")
    parser.add_argument("--disable-gate-fallback", action="store_true")
    parser.add_argument("--data-dir", default=DEFAULT_FAST_CONFIG["data_dir"])
    parser.add_argument("--output-dir", default=DEFAULT_FAST_CONFIG["output_dir"])
    parser.add_argument("--search-space-path", default="")
    parser.add_argument("--seed", type=int, default=DEFAULT_FAST_CONFIG["seed"])
    parser.add_argument("--disable-global-rules", action="store_true")
    parser.add_argument("--global-rules-profile", choices=["safe", "balanced", "aggressive"], default=DEFAULT_FAST_CONFIG["global_rules_profile"])
    parser.add_argument("--min-24h-notional-usdt", type=float, default=DEFAULT_FAST_CONFIG["min_24h_notional_usdt"])
    parser.add_argument("--enable-mtf-context-filter", action="store_true", default=None)
    parser.add_argument("--enable-regime-filter", action="store_true", default=None)
    parser.add_argument("--context-fast-period", type=int, default=DEFAULT_FAST_CONFIG["context_fast_period"])
    parser.add_argument("--context-slow-period", type=int, default=DEFAULT_FAST_CONFIG["context_slow_period"])
    parser.add_argument("--regime-symbol", default="")
    parser.add_argument("--replay-top-n", type=int, default=DEFAULT_FAST_CONFIG["replay_top_n"])
    parser.add_argument("--allow-without-replay", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = {
        "quality_preset": args.quality_preset,
        "search_profile": args.search_profile,
        "symbols": args.symbols,
        "timeframes": args.timeframes,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "candidate_count": args.candidate_count,
        "workers": args.workers,
        "top_k": args.top_k,
        "publish_top_n": args.publish_top_n,
        "min_return_pct": args.min_return_pct,
        "max_drawdown_pct": args.max_drawdown_pct,
        "min_win_rate_pct": args.min_win_rate_pct,
        "min_trades": args.min_trades,
        "min_test_return_pct": args.min_test_return_pct,
        "max_test_drawdown_pct": args.max_test_drawdown_pct,
        "min_test_win_rate_pct": args.min_test_win_rate_pct,
        "min_test_trades": args.min_test_trades,
        "train_ratio": args.train_ratio,
        "enable_train_test_split": not bool(args.disable_train_test_split),
        "enable_gate_fallback": not bool(args.disable_gate_fallback),
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "search_space_path": args.search_space_path,
        "seed": args.seed,
        "enable_global_rules": not bool(args.disable_global_rules),
        "global_rules_profile": args.global_rules_profile,
        "min_24h_notional_usdt": float(args.min_24h_notional_usdt),
        "context_fast_period": args.context_fast_period,
        "context_slow_period": args.context_slow_period,
        "regime_symbol": args.regime_symbol,
        "replay_top_n": args.replay_top_n,
        "require_replay_pass": not bool(args.allow_without_replay),
    }
    if args.enable_mtf_context_filter is not None:
        config["enable_mtf_context_filter"] = args.enable_mtf_context_filter
    if args.enable_regime_filter is not None:
        config["enable_regime_filter"] = args.enable_regime_filter

    result = run_fast_finder(config_override=config)
    print(f"Run directory: {result['run_dir']}")
    print(f"Evaluated: {result['evaluated']}/{result['requested_candidates']}")
    print(f"Elapsed: {_format_duration_hms(float(result['elapsed_seconds']))}")
    if result["top_results"]:
        best = result["top_results"][0]
        metrics = best.get("metrics", {})
        print(
            "Best score={score:.4f} return={ret:.2f}% dd={dd:.2f}% win={win:.2f}% trades={trades}".format(
                score=float(best.get("score", 0.0)),
                ret=float(metrics.get("total_return", 0.0)),
                dd=float(metrics.get("max_drawdown", 0.0)),
                win=float(metrics.get("win_rate", 0.0)),
                trades=int(metrics.get("total_trades", 0)),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
