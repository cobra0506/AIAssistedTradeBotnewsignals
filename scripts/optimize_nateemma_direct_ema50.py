"""Optimize EMA50 on the standard 10-symbol / 31-day screen."""

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
SCREEN_DAYS = 31
MAX_WORKERS = 4

PARAM_GRID = {
    "ema50_period": [34, 50, 72],
    "trend_ma_period": [34, 50, 72],
    "trend_deadband_pct": [0.05, 0.10, 0.15],
}


def _data_range(symbol: str, timeframe: str):
    df = pd.read_csv(Path("data") / f"{symbol}_{timeframe.rstrip('m')}.csv")
    return pd.to_datetime(df["datetime"].iloc[0]), pd.to_datetime(df["datetime"].iloc[-1])


def _score_row(row: dict) -> tuple:
    return (
        float(row["total_return_pct"]),
        float(row["sharpe_ratio"]),
        float(row["win_rate_pct"]),
        -float(row["max_drawdown_pct"]),
        float(row["total_trades"]),
    )


def _run_trial(params: dict, start_date: pd.Timestamp, end_date: pd.Timestamp):
    module = importlib.import_module(f"simple_strategy.strategies.{MODULE_NAME}")
    feeder = DataFeeder(data_dir="data", memory_limit_percent=85)
    feeder.load_data(TRADE_SYMBOLS, TIMEFRAMES, start_date=start_date, end_date=end_date)
    strategy = module.create_strategy(symbols=TRADE_SYMBOLS, timeframes=TIMEFRAMES, **params)
    engine = BacktesterEngine(
        data_feeder=feeder,
        strategy=strategy,
        config={"processing_mode": "sequential", "batch_size": 2000, "memory_limit_percent": 85},
    )
    engine._save_backtest_results = lambda *args, **kwargs: None
    with contextlib.redirect_stdout(io.StringIO()):
        result = engine.run_backtest(TRADE_SYMBOLS, TIMEFRAMES, start_date, end_date)
    return {
        "strategy_module": MODULE_NAME,
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

    start_date, end_date = _data_range("BNBUSDT", "15m")
    start_date = max(start_date, end_date - timedelta(days=SCREEN_DAYS))

    param_names = list(PARAM_GRID.keys())
    combos = [dict(zip(param_names, values)) for values in itertools.product(*(PARAM_GRID[name] for name in param_names))]

    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_run_trial, params, start_date, end_date) for params in combos]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    rows.sort(key=_score_row, reverse=True)
    best = rows[0]
    retest = _run_trial(best["params"], start_date, end_date)

    wall_end = time.time()
    wall_end_dt = datetime.fromtimestamp(wall_end)
    wall_elapsed = round(wall_end - wall_start, 2)

    output_dir = Path("docs/strategy_source_catalog/intake/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy_module": MODULE_NAME,
        "trade_symbols": TRADE_SYMBOLS,
        "timeframes": TIMEFRAMES,
        "screen_days": SCREEN_DAYS,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "wall_clock_start": wall_start_dt.isoformat(sep=" ", timespec="seconds"),
        "wall_clock_end": wall_end_dt.isoformat(sep=" ", timespec="seconds"),
        "wall_clock_elapsed_seconds": wall_elapsed,
        "trial_count": len(rows),
        "best_trial": best,
        "retest": retest,
        "all_trials": rows,
    }
    json_path = output_dir / "github_nateemma_direct_ema50_optimization.json"
    md_path = output_dir / "github_nateemma_direct_ema50_optimization.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
