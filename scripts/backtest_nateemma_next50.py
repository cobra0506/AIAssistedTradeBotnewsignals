"""Run the next 50 nateemma strategy imports using the standard heavy screen."""

from __future__ import annotations

import concurrent.futures
import contextlib
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

SYMBOLS = ["BNBUSDT", "ADAUSDT", "XRPUSDT", "ALGOUSDT", "ARBUSDT", "ATOMUSDT", "DOTUSDT", "FILUSDT", "NEARUSDT", "OPUSDT"]
TIMEFRAMES = ["5m", "15m"]
SCREEN_DAYS = 31
MAX_WORKERS = 12

BATCHES = {
    "004": [
        "Strategy_Import_Nateemma_NNTC_FBB_ENSEMBLE",
        "Strategy_Import_Nateemma_NNTC_FBB_GRU",
        "Strategy_Import_Nateemma_NNTC_FBB_LSTM",
        "Strategy_Import_Nateemma_NNTC_FBB_MULTIHEAD",
        "Strategy_Import_Nateemma_NNTC_FBB_TRANSFORMER",
        "Strategy_Import_Nateemma_NNTC_FBB_WAVENET",
        "Strategy_Import_Nateemma_NNTC_FWR_LSTM",
        "Strategy_Import_Nateemma_NNTC_HIGHLOW_ENSEMBLE",
        "Strategy_Import_Nateemma_NNTC_HIGHLOW_LSTM",
        "Strategy_Import_Nateemma_NNTC_JUMP_ENSEMBLE",
    ],
    "005": [
        "Strategy_Import_Nateemma_NNTC_JUMP_LSTM",
        "Strategy_Import_Nateemma_NNTC_JUMP_TRANSFORMER",
        "Strategy_Import_Nateemma_NNTC_MACD2_ATTENTION",
        "Strategy_Import_Nateemma_NNTC_MACD3_LSTM",
        "Strategy_Import_Nateemma_NNTC_MACD_ATTENTION",
        "Strategy_Import_Nateemma_NNTC_MACD_ENSEMBLE",
        "Strategy_Import_Nateemma_NNTC_MACD_GRU",
        "Strategy_Import_Nateemma_NNTC_MACD_LSTM",
        "Strategy_Import_Nateemma_NNTC_MACD_MULTIHEAD",
        "Strategy_Import_Nateemma_NNTC_MACD_TRANSFORMER",
    ],
    "006": [
        "Strategy_Import_Nateemma_NNTC_MFI_LSTM",
        "Strategy_Import_Nateemma_NNTC_MINMAX_LSTM",
        "Strategy_Import_Nateemma_NNTC_NSEQ_ATTENTION",
        "Strategy_Import_Nateemma_NNTC_NSEQ_ENSEMBLE",
        "Strategy_Import_Nateemma_NNTC_NSEQ_GRU",
        "Strategy_Import_Nateemma_NNTC_NSEQ_LSTM",
        "Strategy_Import_Nateemma_NNTC_NSEQ_TRANSFORMER",
        "Strategy_Import_Nateemma_NNTC_NSEQ_WAVENET",
        "Strategy_Import_Nateemma_NNTC_OVER_LSTM",
        "Strategy_Import_Nateemma_NNTC_PROFIT_ADDITIVEATTENTION",
    ],
    "007": [
        "Strategy_Import_Nateemma_NNTC_PROFIT_ATTENTION",
        "Strategy_Import_Nateemma_NNTC_PROFIT_CNN",
        "Strategy_Import_Nateemma_NNTC_PROFIT_ENSEMBLE",
        "Strategy_Import_Nateemma_NNTC_PROFIT_GRU",
        "Strategy_Import_Nateemma_NNTC_PROFIT_LSTM",
        "Strategy_Import_Nateemma_NNTC_PROFIT_LSTM2",
        "Strategy_Import_Nateemma_NNTC_PROFIT_LSTM3",
        "Strategy_Import_Nateemma_NNTC_PROFIT_MLP",
        "Strategy_Import_Nateemma_NNTC_PROFIT_TRANSFORMER",
        "Strategy_Import_Nateemma_NNTC_PROFIT_WAVENET",
    ],
    "008": [
        "Strategy_Import_Nateemma_NNTC_PROFIT_WAVENET2",
        "Strategy_Import_Nateemma_NNTC_PROFIT_WAVENET3",
        "Strategy_Import_Nateemma_NNTC_PV_ENSEMBLE",
        "Strategy_Import_Nateemma_NNTC_PV_LSTM",
        "Strategy_Import_Nateemma_NNTC_PV_MLP",
        "Strategy_Import_Nateemma_NNTC_PV_WAVENET",
        "Strategy_Import_Nateemma_NNTC_SLOPE_LSTM",
        "Strategy_Import_Nateemma_NNTC_SMOOTH_LSTM",
        "Strategy_Import_Nateemma_NNTC_STOCHASTIC_LSTM",
        "Strategy_Import_Nateemma_NNTC_SWING_LSTM",
    ],
}


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


def _write_markdown(batch_id: str, rows, start_date, end_date):
    output_dir = Path("docs/strategy_source_catalog/intake/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"github_nateemma_strategies_batch_{batch_id}_results.md"
    lines = [
        f"# Batch {batch_id} Results",
        "",
        "- Source place: `nateemma/strategies`",
        f"- Batch: `github_nateemma_strategies_batch_{batch_id}`",
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

    output_dir = Path("docs/strategy_source_catalog/intake/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = {}

    for batch_id, modules in BATCHES.items():
        rows = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(_run_single_strategy, module_name, start_date, end_date) for module_name in modules]
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
        rows.sort(key=lambda item: item["strategy_module"])
        overall[batch_id] = rows
        (output_dir / f"github_nateemma_strategies_batch_{batch_id}_results.json").write_text(
            json.dumps(rows, indent=2),
            encoding="utf-8",
        )
        _write_markdown(batch_id, rows, start_date, end_date)
        print(f"Finished batch {batch_id}")

    summary_path = output_dir / "github_nateemma_strategies_next50_summary.json"
    summary_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
