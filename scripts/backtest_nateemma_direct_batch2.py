"""Backtest the second direct-rule nateemma batch with the standard heavy screen."""

from __future__ import annotations

import concurrent.futures
import contextlib
import importlib
import io
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

STRATEGY_MODULES = [
    "Strategy_Import_Nateemma_Direct_BollingerBounce",
    "Strategy_Import_Nateemma_Direct_BuyDips",
    "Strategy_Import_Nateemma_Direct_DCBBBounce",
    "Strategy_Import_Nateemma_Direct_DonchianBounce",
    "Strategy_Import_Nateemma_Direct_DonchianChannel",
    "Strategy_Import_Nateemma_Direct_EMA003",
    "Strategy_Import_Nateemma_Direct_EMA50",
    "Strategy_Import_Nateemma_Direct_EMABounce",
    "Strategy_Import_Nateemma_Direct_EMABreakout",
    "Strategy_Import_Nateemma_Direct_EMACross",
]

TRADE_SYMBOLS = ["BNBUSDT", "ADAUSDT", "XRPUSDT", "ALGOUSDT", "ARBUSDT", "ATOMUSDT", "DOTUSDT", "FILUSDT", "NEARUSDT", "OPUSDT"]
TIMEFRAMES = ["5m", "15m"]
SCREEN_DAYS = 31
MAX_WORKERS = 4


def _data_range(symbol: str, timeframe: str):
    df = pd.read_csv(Path("data") / f"{symbol}_{timeframe.rstrip('m')}.csv")
    return pd.to_datetime(df["datetime"].iloc[0]), pd.to_datetime(df["datetime"].iloc[-1])


def _run_single_strategy(module_name: str, start_date: pd.Timestamp, end_date: pd.Timestamp):
    module = importlib.import_module(f"simple_strategy.strategies.{module_name}")
    feeder = DataFeeder(data_dir="data", memory_limit_percent=85)
    feeder.load_data(TRADE_SYMBOLS, TIMEFRAMES, start_date=start_date, end_date=end_date)
    strategy = module.create_strategy(symbols=TRADE_SYMBOLS, timeframes=TIMEFRAMES)
    engine = BacktesterEngine(
        data_feeder=feeder,
        strategy=strategy,
        config={"processing_mode": "sequential", "batch_size": 2000, "memory_limit_percent": 85},
    )
    engine._save_backtest_results = lambda *args, **kwargs: None
    data = feeder.get_data_for_symbols(TRADE_SYMBOLS, TIMEFRAMES, start_date, end_date)
    with contextlib.redirect_stdout(io.StringIO()):
        result = engine.run_backtest(TRADE_SYMBOLS, TIMEFRAMES, start_date, end_date, data=data)
    return {
        "strategy_module": module_name,
        "symbols": ",".join(TRADE_SYMBOLS),
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
    for logger_name in (
        "simple_strategy.shared.data_feeder",
        "simple_strategy.backtester.backtester_engine",
        "simple_strategy.backtester.risk_manager",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    wall_start_ts = time.time()
    wall_start_dt = datetime.fromtimestamp(wall_start_ts)

    start_date, end_date = _data_range("BNBUSDT", "15m")
    start_date = max(start_date, end_date - timedelta(days=SCREEN_DAYS))

    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_run_single_strategy, module_name, start_date, end_date) for module_name in STRATEGY_MODULES]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda item: item["strategy_module"])

    wall_end_ts = time.time()
    wall_end_dt = datetime.fromtimestamp(wall_end_ts)
    wall_elapsed_seconds = round(wall_end_ts - wall_start_ts, 2)

    output_dir = Path("docs/strategy_source_catalog/intake/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "github_nateemma_direct_batch_002_results.json"
    md_path = output_dir / "github_nateemma_direct_batch_002_results.md"

    payload = {
        "batch_id": "github_nateemma_direct_batch_002",
        "trade_symbols": TRADE_SYMBOLS,
        "timeframes": TIMEFRAMES,
        "screen_days": SCREEN_DAYS,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "wall_clock_start": wall_start_dt.isoformat(sep=" ", timespec="seconds"),
        "wall_clock_end": wall_end_dt.isoformat(sep=" ", timespec="seconds"),
        "wall_clock_elapsed_seconds": wall_elapsed_seconds,
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Direct Batch 002 Results",
        "",
        "- Source place: `nateemma/strategies`",
        "- Batch: `github_nateemma_direct_batch_002`",
        f"- Trade symbols: `{', '.join(TRADE_SYMBOLS)}`",
        f"- Screen window: last `{SCREEN_DAYS}` days",
        f"- Date range: `{start_date}` -> `{end_date}`",
        f"- Wall-clock start: `{payload['wall_clock_start']}`",
        f"- Wall-clock end: `{payload['wall_clock_end']}`",
        f"- Wall-clock elapsed seconds: `{wall_elapsed_seconds}`",
        "",
        "| Strategy | Return % | Trades | Win Rate % | Sharpe | Max DD % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['strategy_module']} | {row['total_return_pct']:.4f} | {row['total_trades']} | "
            f"{row['win_rate_pct']:.4f} | {row['sharpe_ratio']:.4f} | {row['max_drawdown_pct']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
