"""Backtest the custom big-candle half-target strategy on the standard heavy screen."""

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
SCREEN_DAYS = 31
STRATEGY_PARAMS = {"min_candle_pct": 2.0, "target_fraction": 0.5}


def _data_range(symbol: str, timeframe: str):
    df = pd.read_csv(Path("data") / f"{symbol}_{timeframe.rstrip('m')}.csv")
    return pd.to_datetime(df["datetime"].iloc[0]), pd.to_datetime(df["datetime"].iloc[-1])


def main():
    logging.getLogger().setLevel(logging.WARNING)
    wall_start = time.time()
    wall_start_dt = datetime.fromtimestamp(wall_start)
    start_date, end_date = _data_range("BNBUSDT", "15m")
    start_date = max(start_date, end_date - timedelta(days=SCREEN_DAYS))

    module = importlib.import_module(f"simple_strategy.strategies.{MODULE_NAME}")
    feeder = DataFeeder(data_dir="data", memory_limit_percent=85)
    feeder.load_data(TRADE_SYMBOLS, TIMEFRAMES, start_date=start_date, end_date=end_date)
    strategy = module.create_strategy(symbols=TRADE_SYMBOLS, timeframes=TIMEFRAMES, **STRATEGY_PARAMS)
    engine = BacktesterEngine(data_feeder=feeder, strategy=strategy, config={"processing_mode": "sequential", "batch_size": 2000, "memory_limit_percent": 85})
    engine._save_backtest_results = lambda *args, **kwargs: None
    with contextlib.redirect_stdout(io.StringIO()):
        result = engine.run_backtest(TRADE_SYMBOLS, TIMEFRAMES, start_date, end_date)

    payload = {
        "strategy_module": MODULE_NAME,
        "trade_symbols": TRADE_SYMBOLS,
        "timeframes": TIMEFRAMES,
        "screen_days": SCREEN_DAYS,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "wall_clock_start": wall_start_dt.isoformat(sep=" ", timespec="seconds"),
        "wall_clock_end": datetime.fromtimestamp(time.time()).isoformat(sep=" ", timespec="seconds"),
        "wall_clock_elapsed_seconds": round(time.time() - wall_start, 2),
        "total_return_pct": round(float(result.get("total_return", 0.0)), 4),
        "win_rate_pct": round(float(result.get("win_rate", 0.0)), 4),
        "sharpe_ratio": round(float(result.get("sharpe_ratio", 0.0)), 4),
        "max_drawdown_pct": round(float(result.get("max_drawdown", 0.0)), 4),
        "total_trades": int(result.get("total_trades", 0)),
    }
    out_dir = Path("docs/strategy_source_catalog/intake/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "custom_big_candle_half_target_results.json"
    md_path = out_dir / "custom_big_candle_half_target_results.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Custom Big Candle Half Target Results",
                "",
                "- Logic used:",
                f"- if candle range is at least {STRATEGY_PARAMS['min_candle_pct']}%",
                "- green candle -> open short",
                "- red candle -> open long",
                "- no trend filter is applied",
                f"- exit when price retraces by {STRATEGY_PARAMS['target_fraction']:.2f} of the trigger candle's range",
                "",
                f"- Return %: `{payload['total_return_pct']}`",
                f"- Trades: `{payload['total_trades']}`",
                f"- Win rate %: `{payload['win_rate_pct']}`",
                f"- Sharpe: `{payload['sharpe_ratio']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
