"""Sweep BigCandle variants on train data and validate finalists on unseen data."""

from __future__ import annotations

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

MODULE_NAME = "Strategy_Custom_BigCandle_HalfTarget"
TRADE_SYMBOLS = ["BNBUSDT", "ADAUSDT", "XRPUSDT", "ALGOUSDT", "ARBUSDT", "ATOMUSDT", "DOTUSDT", "FILUSDT", "NEARUSDT", "OPUSDT"]
TIMEFRAMES = ["5m", "15m"]
COARSE_DAYS = 10
TRAIN_DAYS = 31
UNSEEN_DAYS = 31


def _data_range(symbol: str, timeframe: str):
    df = pd.read_csv(Path("data") / f"{symbol}_{timeframe.rstrip('m')}.csv")
    return pd.to_datetime(df["datetime"].iloc[0]), pd.to_datetime(df["datetime"].iloc[-1])


def _run_window(params: dict, start_date: pd.Timestamp, end_date: pd.Timestamp) -> dict:
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
        "params": dict(params),
        "total_return_pct": round(float(result.get("total_return", 0.0)), 4),
        "win_rate_pct": round(float(result.get("win_rate", 0.0)), 4),
        "sharpe_ratio": round(float(result.get("sharpe_ratio", 0.0)), 4),
        "max_drawdown_pct": round(float(result.get("max_drawdown", 0.0)), 4),
        "total_trades": int(result.get("total_trades", 0)),
    }


def _score_candidate(payload: dict) -> tuple:
    return (
        float(payload.get("total_return_pct", 0.0)),
        float(payload.get("sharpe_ratio", 0.0)),
        -float(payload.get("max_drawdown_pct", 0.0)),
        int(payload.get("total_trades", 0)),
    )


def _candidate_grid():
    thresholds = [2.0, 3.0]
    target_fractions = [0.5, 1.0]
    max_hold_bars = [0, 4]
    min_body_ratios = [0.0, 0.5]
    volume_spikes = [0.0, 1.3]
    cooldown_bars = [0, 2, 4]
    for threshold, target_fraction, max_hold, body_ratio, volume_spike, cooldown in itertools.product(
        thresholds,
        target_fractions,
        max_hold_bars,
        min_body_ratios,
        volume_spikes,
        cooldown_bars,
    ):
        yield {
            "min_candle_pct": threshold,
            "target_fraction": target_fraction,
            "max_hold_bars": max_hold,
            "min_body_ratio": body_ratio,
            "volume_spike_multiplier": volume_spike,
            "volume_lookback": 20,
            "cooldown_bars": cooldown,
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
    coarse_start = max(overall_start, train_end - timedelta(days=COARSE_DAYS))

    coarse_results = []
    for params in _candidate_grid():
        coarse_results.append(_run_window(params, coarse_start, train_end))

    coarse_ranked = sorted(coarse_results, key=_score_candidate, reverse=True)
    train_results = []
    for candidate in coarse_ranked[:8]:
        train_results.append(_run_window(candidate["params"], train_start, train_end))

    train_ranked = sorted(train_results, key=_score_candidate, reverse=True)
    unseen_results = []
    for candidate in train_ranked[:4]:
        unseen_results.append(_run_window(candidate["params"], unseen_start, unseen_end))

    unseen_ranked = sorted(unseen_results, key=_score_candidate, reverse=True)
    payload = {
        "strategy_module": MODULE_NAME,
        "trade_symbols": TRADE_SYMBOLS,
        "timeframes": TIMEFRAMES,
        "coarse_start_date": str(coarse_start),
        "train_start_date": str(train_start),
        "train_end_date": str(train_end),
        "unseen_start_date": str(unseen_start),
        "unseen_end_date": str(unseen_end),
        "wall_clock_start": wall_start_dt.isoformat(sep=" ", timespec="seconds"),
        "wall_clock_end": datetime.fromtimestamp(time.time()).isoformat(sep=" ", timespec="seconds"),
        "wall_clock_elapsed_seconds": round(time.time() - wall_start, 2),
        "coarse_top_12": coarse_ranked[:12],
        "train_top_5": train_ranked[:5],
        "unseen_top_5": unseen_ranked[:5],
        "best_train": train_ranked[0] if train_ranked else None,
        "best_unseen": unseen_ranked[0] if unseen_ranked else None,
        "tested_candidate_count": len(coarse_results),
    }

    out_dir = Path("docs/strategy_source_catalog/intake/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "custom_big_candle_builder_optimization.json"
    md_path = out_dir / "custom_big_candle_builder_optimization.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Big Candle Builder Optimization",
        "",
        f"- Coarse window: `{coarse_start}` -> `{train_end}`",
        f"- Train window: `{train_start}` -> `{train_end}`",
        f"- Unseen window: `{unseen_start}` -> `{unseen_end}`",
        f"- Candidates tested: `{payload['tested_candidate_count']}`",
    ]
    if payload["best_train"]:
        lines.extend(
            [
                "",
                "## Best Train",
                f"- Params: `{payload['best_train']['params']}`",
                f"- Return %: `{payload['best_train']['total_return_pct']}`",
                f"- Trades: `{payload['best_train']['total_trades']}`",
                f"- Sharpe: `{payload['best_train']['sharpe_ratio']}`",
            ]
        )
    if payload["best_unseen"]:
        lines.extend(
            [
                "",
                "## Best Unseen",
                f"- Params: `{payload['best_unseen']['params']}`",
                f"- Return %: `{payload['best_unseen']['total_return_pct']}`",
                f"- Trades: `{payload['best_unseen']['total_trades']}`",
                f"- Sharpe: `{payload['best_unseen']['sharpe_ratio']}`",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
