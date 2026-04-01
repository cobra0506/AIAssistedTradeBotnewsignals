"""Backtest the second 10 nateemma imported strategies with a heavier screen."""

from __future__ import annotations

import contextlib
import concurrent.futures
import importlib
import io
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simple_strategy.backtester.backtester_engine import BacktesterEngine
from simple_strategy.shared.data_feeder import DataFeeder

STRATEGY_MODULES = [
    "Strategy_Import_Nateemma_Anomaly_MFI",
    "Strategy_Import_Nateemma_Anomaly_MINMAX",
    "Strategy_Import_Nateemma_Anomaly_NSEQ",
    "Strategy_Import_Nateemma_Anomaly_OVER",
    "Strategy_Import_Nateemma_Anomaly_PROFIT",
    "Strategy_Import_Nateemma_Anomaly_PV",
    "Strategy_Import_Nateemma_Anomaly_SLOPE",
    "Strategy_Import_Nateemma_Anomaly_SMOOTH",
    "Strategy_Import_Nateemma_Anomaly_STOCHASTIC",
    "Strategy_Import_Nateemma_Anomaly_SWING",
]

SYMBOLS = ["BNBUSDT", "ADAUSDT", "XRPUSDT", "ALGOUSDT", "ARBUSDT", "ATOMUSDT", "DOTUSDT", "FILUSDT", "NEARUSDT", "OPUSDT"]
TIMEFRAMES = ["5m", "15m"]
SCREEN_DAYS = 31
MAX_WORKERS = 4


def _data_range(symbol: str, timeframe: str):
    df = pd.read_csv(Path("data") / f"{symbol}_{timeframe.rstrip('m')}.csv")
    return pd.to_datetime(df["datetime"].iloc[0]), pd.to_datetime(df["datetime"].iloc[-1])


def _run_single_strategy(module_name: str, start_date: pd.Timestamp, end_date: pd.Timestamp):
    module = importlib.import_module(f"simple_strategy.strategies.{module_name}")
    feeder = DataFeeder(data_dir="data", memory_limit_percent=85)
    feeder.load_data(SYMBOLS, TIMEFRAMES, start_date=start_date, end_date=end_date)
    strategy = module.create_strategy(symbols=SYMBOLS, timeframes=TIMEFRAMES)
    engine = BacktesterEngine(
        data_feeder=feeder,
        strategy=strategy,
        config={"processing_mode": "sequential", "batch_size": 2000, "memory_limit_percent": 85},
    )
    engine._save_backtest_results = lambda *args, **kwargs: None
    with contextlib.redirect_stdout(io.StringIO()):
        result = engine.run_backtest(SYMBOLS, TIMEFRAMES, start_date, end_date)
    return {
        "strategy_module": module_name,
        "symbols": ",".join(SYMBOLS),
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

    start_date, end_date = _data_range("BNBUSDT", "15m")
    start_date = max(start_date, end_date - timedelta(days=SCREEN_DAYS))

    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_run_single_strategy, module_name, start_date, end_date) for module_name in STRATEGY_MODULES]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda item: item["strategy_module"])

    output_dir = Path("docs/strategy_source_catalog/intake/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "github_nateemma_strategies_batch_002_results.json"
    md_path = output_dir / "github_nateemma_strategies_batch_002_results.md"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# Batch 002 Results",
        "",
        "- Source place: `nateemma/strategies`",
        "- Batch: `github_nateemma_strategies_batch_002`",
        f"- Symbols: `{', '.join(SYMBOLS)}`",
        f"- Screen window: last `{SCREEN_DAYS}` days",
        f"- Date range: `{start_date}` -> `{end_date}`",
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
