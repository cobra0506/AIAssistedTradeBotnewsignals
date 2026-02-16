import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


DEFAULT_SEARCH_SPACE: Dict[str, Any] = {
    "indicators": {
        "rsi_main": {
            "function": "rsi",
            "params": {
                "period": {"type": "int", "low": 7, "high": 28, "step": 1},
            },
        },
        "ema_fast": {
            "function": "ema",
            "params": {
                "period": {"type": "int", "low": 5, "high": 40, "step": 1},
            },
        },
        "ema_slow": {
            "function": "ema",
            "params": {
                "period": {"type": "int", "low": 20, "high": 140, "step": 1},
            },
        },
        "sma_fast": {
            "function": "sma",
            "params": {
                "period": {"type": "int", "low": 5, "high": 50, "step": 1},
            },
        },
        "sma_slow": {
            "function": "sma",
            "params": {
                "period": {"type": "int", "low": 20, "high": 200, "step": 1},
            },
        },
    },
    "signals": {
        "rsi_osob": {
            "function": "overbought_oversold",
            "inputs": {
                "indicator": "rsi_main",
            },
            "params": {
                "overbought": {"type": "int", "low": 65, "high": 85, "step": 1},
                "oversold": {"type": "int", "low": 15, "high": 40, "step": 1},
            },
        },
        "ema_cross": {
            "function": "ma_crossover",
            "inputs": {
                "fast_ma": "ema_fast",
                "slow_ma": "ema_slow",
            },
            "params": {},
        },
        "sma_cross": {
            "function": "ma_crossover",
            "inputs": {
                "fast_ma": "sma_fast",
                "slow_ma": "sma_slow",
            },
            "params": {},
        },
        "rsi_trend_combo": {
            "function": "rsi_mean_reversion_with_trend",
            "inputs": {
                "rsi": "rsi_main",
                "ema_fast": "ema_fast",
                "ema_slow": "ema_slow",
            },
            "params": {
                "overbought": {"type": "int", "low": 65, "high": 85, "step": 1},
                "oversold": {"type": "int", "low": 15, "high": 40, "step": 1},
            },
        },
    },
    "signal_combination_methods": ["majority_vote", "and_signals", "unanimous"],
}


DEFAULT_RUN_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "symbols": ["BTCUSDT"],
    "timeframes": ["5m"],
    "train_start": "2024-01-01",
    "train_end": "2024-06-30",
    "validation_start": "2024-07-01",
    "validation_end": "2024-09-30",
    "final_start": "2024-10-01",
    "final_end": "2024-12-31",
    "population_size": 64,
    "generations": 40,
    "elite_count": 6,
    "tournament_size": 3,
    "crossover_probability": 0.7,
    "mutation_probability": 0.3,
    "inner_opt_trials": 20,
    "top_k": 10,
    "min_trades": 10,
    "validation_min_trades": 5,
    "min_train_return_pct": 0.0,
    "max_train_drawdown_pct": 40.0,
    "min_validation_return_pct": -5.0,
    "max_validation_drawdown_pct": 45.0,
    "max_validation_gap_pct": 30.0,
    "max_stress_drop_pct": 30.0,
    "min_stress_return_pct": -10.0,
    "max_stress_drawdown_pct": 50.0,
    "max_active_signals": 3,
    "workers": 4,
    "checkpoint_every_generations": 2,
    "stress_fee_multiplier": 1.5,
    "stress_slippage_multiplier": 1.5,
    "output_dir": "simple_strategy/auto_evolve/runs",
    "save_strategy_files": True,
    "enable_inner_optimization": True,
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_run_config(config_path: str = "") -> Dict[str, Any]:
    config = deepcopy(DEFAULT_RUN_CONFIG)
    if not config_path:
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")

    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return deep_merge(config, payload)


def load_search_space(search_space_path: str = "") -> Dict[str, Any]:
    space = deepcopy(DEFAULT_SEARCH_SPACE)
    if not search_space_path:
        return space

    path = Path(search_space_path)
    if not path.exists():
        raise FileNotFoundError(f"Search space config not found: {search_space_path}")

    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return deep_merge(space, payload)
