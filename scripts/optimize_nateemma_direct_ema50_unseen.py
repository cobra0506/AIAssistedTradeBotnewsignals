"""Optimize EMA50 on a training window, then retest on unseen data."""

from __future__ import annotations

import concurrent.futures
import contextlib
import importlib
import io
import itertools
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simple_strategy.backtester.backtester_engine import BacktesterEngine
from simple_strategy.shared.data_feeder import DataFeeder

MODULE_NAME = "Strategy_Import_Nateemma_Direct_EMA50"
TRADE_SYMBOLS = ["BNBUSDT", "ADAUSDT", "XRPUSDT", "ALGOUSDT", "ARBUSDT", "ATOMUSDT", "DOTUSDT", "FILUSDT", "NEARUSDT", "OPUSDT"]
TIMEFRAMES = ["5m", "15m"]
TRAIN_DAYS = 90
UNSEEN_DAYS = 31
MAX_WORKERS = 4

PARAM_GRID = {
    "ema50_period": [34, 50, 72],
    "trend_ma_period": [34, 50, 72],
    "trend_deadband_pct": [0.05, 0.10, 0.15],
}

DEFAULT_PARAMS = {"ema50_period": 50, "trend_ma_period": 50, "trend_deadband_pct": 0.15}


def _data_range(symbol: str, timeframe: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    df = pd.read_csv(Path("data") / f"{symbol}_{timeframe.rstrip('m')}.csv", parse_dates=["datetime"])
    return pd.Timestamp(df["datetime"].iloc[0]), pd.Timestamp(df["datetime"].iloc[-1])


def _score_row(row: dict) -> tuple:
    return (float(row["total_return_pct"]), float(row["sharpe_ratio"]), float(row["win_rate_pct"]), -float(row["max_drawdown_pct"]), float(row["total_trades"]))


def _run_trial(params: dict, start_date: pd.Timestamp, end_date: pd.Timestamp, label: str) -> dict:
    module = importlib.import_module(f"simple_strategy.strategies.{MODULE_NAME}")
    feeder = DataFeeder(data_dir="data", memory_limit_percent=85)
    feeder.load_data(TRADE_SYMBOLS, TIMEFRAMES, start_date=start_date, end_date=end_date)
    strategy = module.create_strategy(symbols=TRADE_SYMBOLS, timeframes=TIMEFRAMES, **params)
    engine = BacktesterEngine(data_feeder=feeder, strategy=strategy, config={"processing_mode": "sequential", "batch_size": 2000, "memory_limit_percent": 85})
    engine._save_backtest_results = lambda *args, **kwargs: None
    with contextlib.redirect_stdout(io.StringIO()):
        result = engine.run_backtest(TRADE_SYMBOLS, TIMEFRAMES, start_date, end_date)
    return {
        "strategy_module": MODULE_NAME,
        "label": label,
        "params": params,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "total_return_pct": round(float(result.get("total_return", 0.0)), 4),
        "win_rate_pct": round(float(result.get("win_rate", 0.0)), 4),
        "sharpe_ratio": round(float(result.get("sharpe_ratio", 0.0)), 4),
        "max_drawdown_pct": round(float(result.get("max_drawdown", 0.0)), 4),
        "total_trades": int(result.get("total_trades", 0)),
    }


def main():
    logging.getLogger().setLevel(logging.WARNING)
    wall_start = time.time()
    wall_start_dt = datetime.fromtimestamp(wall_start)
    overall_start, overall_end = _data_range("BNBUSDT", "15m")
    unseen_end = overall_end
    unseen_start = unseen_end - timedelta(days=UNSEEN_DAYS)
    train_end = unseen_start
    train_start = max(overall_start, train_end - timedelta(days=TRAIN_DAYS))

    param_names = list(PARAM_GRID.keys())
    combos = [dict(zip(param_names, values)) for values in itertools.product(*(PARAM_GRID[name] for name in param_names))]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_run_trial, params, train_start, train_end, "train") for params in combos]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    rows.sort(key=_score_row, reverse=True)
    best_train = rows[0]
    default_unseen = _run_trial(DEFAULT_PARAMS, unseen_start, unseen_end, "unseen_default")
    best_unseen = _run_trial(best_train["params"], unseen_start, unseen_end, "unseen_optimized")

    payload = {
        "strategy_module": MODULE_NAME,
        "trade_symbols": TRADE_SYMBOLS,
        "timeframes": TIMEFRAMES,
        "train_days": TRAIN_DAYS,
        "unseen_days": UNSEEN_DAYS,
        "train_start_date": str(train_start),
        "train_end_date": str(train_end),
        "unseen_start_date": str(unseen_start),
        "unseen_end_date": str(unseen_end),
        "wall_clock_start": wall_start_dt.isoformat(sep=" ", timespec="seconds"),
        "wall_clock_end": datetime.fromtimestamp(time.time()).isoformat(sep=" ", timespec="seconds"),
        "wall_clock_elapsed_seconds": round(time.time() - wall_start, 2),
        "trial_count": len(rows),
        "default_params": DEFAULT_PARAMS,
        "best_train": best_train,
        "default_unseen": default_unseen,
        "best_unseen": best_unseen,
        "all_train_trials": rows,
    }
    out_dir = Path("docs/strategy_source_catalog/intake/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "github_nateemma_direct_ema50_unseen_optimization.json"
    md_path = out_dir / "github_nateemma_direct_ema50_unseen_optimization.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# EMA50 Unseen-Data Optimization Results",
                "",
                f"- Training range: `{train_start}` -> `{train_end}`",
                f"- Unseen range: `{unseen_start}` -> `{unseen_end}`",
                f"- Best train params: `{best_train['params']}`",
                f"- Default unseen return: `{default_unseen['total_return_pct']}`",
                f"- Optimized unseen return: `{best_unseen['total_return_pct']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
