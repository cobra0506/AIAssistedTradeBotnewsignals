"""Compare BigCandleHalfTarget thresholds on train and unseen windows."""

from __future__ import annotations

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

MODULE_NAME = "Strategy_Custom_BigCandle_HalfTarget"
TRADE_SYMBOLS = ["BNBUSDT", "ADAUSDT", "XRPUSDT", "ALGOUSDT", "ARBUSDT", "ATOMUSDT", "DOTUSDT", "FILUSDT", "NEARUSDT", "OPUSDT"]
TIMEFRAMES = ["5m", "15m"]
TRAIN_DAYS = 31
UNSEEN_DAYS = 31


def _data_range(symbol: str, timeframe: str):
    df = pd.read_csv(Path("data") / f"{symbol}_{timeframe.rstrip('m')}.csv")
    return pd.to_datetime(df["datetime"].iloc[0]), pd.to_datetime(df["datetime"].iloc[-1])


def _run_window(params: dict, start_date: pd.Timestamp, end_date: pd.Timestamp, label: str) -> dict:
    module = importlib.import_module(f"simple_strategy.strategies.{MODULE_NAME}")
    feeder = DataFeeder(data_dir="data", memory_limit_percent=85)
    feeder.load_data(TRADE_SYMBOLS, TIMEFRAMES, start_date=start_date, end_date=end_date)
    strategy = module.create_strategy(symbols=TRADE_SYMBOLS, timeframes=TIMEFRAMES, **params)
    engine = BacktesterEngine(data_feeder=feeder, strategy=strategy, config={"processing_mode": "sequential", "batch_size": 2000, "memory_limit_percent": 85})
    engine._save_backtest_results = lambda *args, **kwargs: None
    with contextlib.redirect_stdout(io.StringIO()):
        result = engine.run_backtest(TRADE_SYMBOLS, TIMEFRAMES, start_date, end_date)
    return {
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

    threshold_2 = {"min_candle_pct": 2.0}
    threshold_1 = {"min_candle_pct": 1.0}

    payload = {
        "strategy_module": MODULE_NAME,
        "trade_symbols": TRADE_SYMBOLS,
        "timeframes": TIMEFRAMES,
        "train_start_date": str(train_start),
        "train_end_date": str(train_end),
        "unseen_start_date": str(unseen_start),
        "unseen_end_date": str(unseen_end),
        "wall_clock_start": wall_start_dt.isoformat(sep=" ", timespec="seconds"),
        "train_threshold_2pct": _run_window(threshold_2, train_start, train_end, "train_threshold_2pct"),
        "train_threshold_1pct": _run_window(threshold_1, train_start, train_end, "train_threshold_1pct"),
        "unseen_threshold_2pct": _run_window(threshold_2, unseen_start, unseen_end, "unseen_threshold_2pct"),
        "unseen_threshold_1pct": _run_window(threshold_1, unseen_start, unseen_end, "unseen_threshold_1pct"),
        "wall_clock_end": datetime.fromtimestamp(time.time()).isoformat(sep=" ", timespec="seconds"),
        "wall_clock_elapsed_seconds": round(time.time() - wall_start, 2),
    }

    out_dir = Path("docs/strategy_source_catalog/intake/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "custom_big_candle_half_target_unseen_validation.json"
    md_path = out_dir / "custom_big_candle_half_target_unseen_validation.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# BigCandleHalfTarget Unseen Validation",
                "",
                f"- Train range: `{train_start}` -> `{train_end}`",
                f"- Unseen range: `{unseen_start}` -> `{unseen_end}`",
                f"- 2% unseen return: `{payload['unseen_threshold_2pct']['total_return_pct']}`",
                f"- 1% unseen return: `{payload['unseen_threshold_1pct']['total_return_pct']}`",
                f"- 2% unseen trades: `{payload['unseen_threshold_2pct']['total_trades']}`",
                f"- 1% unseen trades: `{payload['unseen_threshold_1pct']['total_trades']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
