"""Backtest the first 10 nateemma imported strategies and write a summary."""

from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path
from datetime import timedelta

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simple_strategy.backtester.backtester_engine import BacktesterEngine
from simple_strategy.shared.data_feeder import DataFeeder

STRATEGY_MODULES = [
    "Strategy_Import_Nateemma_Anomaly_ADX",
    "Strategy_Import_Nateemma_Anomaly_ALL",
    "Strategy_Import_Nateemma_Anomaly_AROON",
    "Strategy_Import_Nateemma_Anomaly_BBW",
    "Strategy_Import_Nateemma_Anomaly_DWT",
    "Strategy_Import_Nateemma_Anomaly_FBB",
    "Strategy_Import_Nateemma_Anomaly_FWR",
    "Strategy_Import_Nateemma_Anomaly_HIGHLOW",
    "Strategy_Import_Nateemma_Anomaly_JUMP",
    "Strategy_Import_Nateemma_Anomaly_MACD",
]

SYMBOLS = ["BNBUSDT"]
TIMEFRAMES = ["5m", "15m"]
SCREEN_DAYS = 14


def _data_range(symbol: str, timeframe: str):
    df = pd.read_csv(Path("data") / f"{symbol}_{timeframe.rstrip('m')}.csv")
    return pd.to_datetime(df["datetime"].iloc[0]), pd.to_datetime(df["datetime"].iloc[-1])


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
    feeder = DataFeeder(data_dir="data", memory_limit_percent=85)
    feeder.load_data(SYMBOLS, TIMEFRAMES, start_date=start_date, end_date=end_date)
    rows = []
    for module_name in STRATEGY_MODULES:
        module = importlib.import_module(f"simple_strategy.strategies.{module_name}")
        strategy = module.create_strategy(symbols=SYMBOLS, timeframes=TIMEFRAMES)
        engine = BacktesterEngine(
            data_feeder=feeder,
            strategy=strategy,
            config={"processing_mode": "sequential", "batch_size": 2000, "memory_limit_percent": 85},
        )
        engine._save_backtest_results = lambda *args, **kwargs: None
        result = engine.run_backtest(SYMBOLS, TIMEFRAMES, start_date, end_date)
        rows.append(
            {
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
        )

    output_dir = Path("docs/strategy_source_catalog/intake/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "github_nateemma_strategies_batch_001_results.json"
    md_path = output_dir / "github_nateemma_strategies_batch_001_results.md"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# Batch 001 Results",
        "",
        "- Source place: `nateemma/strategies`",
        "- Batch: `github_nateemma_strategies_batch_001`",
        "- Symbols: `BNBUSDT`",
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
