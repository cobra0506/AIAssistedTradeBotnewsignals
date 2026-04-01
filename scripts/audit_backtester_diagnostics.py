"""Audit the backtester with deterministic sanity checks and import diagnostics."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simple_strategy.backtester.backtester_engine import BacktesterEngine
from simple_strategy.shared.data_feeder import DataFeeder
from simple_strategy.shared.strategy_base import StrategyBase
from simple_strategy.strategies.imported_nateemma_batch1_helper import (
    ImportedNateemmaBatch1Strategy,
    _regime,
)


ONE_SYMBOL = ["BNBUSDT"]
TEN_SYMBOLS = [
    "BNBUSDT",
    "ADAUSDT",
    "XRPUSDT",
    "ALGOUSDT",
    "ARBUSDT",
    "ATOMUSDT",
    "DOTUSDT",
    "FILUSDT",
    "NEARUSDT",
    "OPUSDT",
]
TIMEFRAMES = ["5m", "15m"]
SCREEN_DAYS = 31


class AuditBacktesterEngine(BacktesterEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_stats = {
            "open_attempts": 0,
            "blocked_max_positions": 0,
            "blocked_symbol_already_open": 0,
        }

    def _open_backtest_position(self, *args, **kwargs) -> bool:
        self.audit_stats["open_attempts"] += 1
        symbol = kwargs.get("symbol")
        max_positions = self.config.get("max_positions", 3)
        if len(self.positions) >= max_positions:
            self.audit_stats["blocked_max_positions"] += 1
        elif symbol in self.positions:
            self.audit_stats["blocked_symbol_already_open"] += 1
        return super()._open_backtest_position(*args, **kwargs)


class BuyHoldLongStrategy(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__("Audit_Buy_Hold_Long", symbols, timeframes, config)
        self._opened = set()

    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        result: Dict[str, Dict[str, str]] = {}
        timeframe = self.timeframes[0]
        for symbol, tf_map in data.items():
            result[symbol] = {tf: "HOLD" for tf in tf_map}
            if timeframe not in tf_map or tf_map[timeframe].empty:
                continue
            if symbol not in self._opened:
                self._opened.add(symbol)
                result[symbol][timeframe] = "OPEN_LONG"
        return result


class AlwaysShortStrategy(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__("Audit_Always_Short", symbols, timeframes, config)
        self._opened = set()

    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        result: Dict[str, Dict[str, str]] = {}
        timeframe = self.timeframes[0]
        for symbol, tf_map in data.items():
            result[symbol] = {tf: "HOLD" for tf in tf_map}
            if timeframe not in tf_map or tf_map[timeframe].empty:
                continue
            if symbol not in self._opened:
                self._opened.add(symbol)
                result[symbol][timeframe] = "OPEN_SHORT"
        return result


class ScriptedTradeStrategy(StrategyBase):
    def __init__(
        self,
        symbols: List[str],
        timeframes: List[str],
        config: Dict[str, Any],
        schedule: Dict[str, Dict[pd.Timestamp, str]],
    ):
        super().__init__("Audit_Scripted_Trade", symbols, timeframes, config)
        self.schedule = schedule

    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        result: Dict[str, Dict[str, str]] = {}
        timeframe = self.timeframes[0]
        for symbol, tf_map in data.items():
            result[symbol] = {tf: "HOLD" for tf in tf_map}
            if timeframe not in tf_map or tf_map[timeframe].empty:
                continue
            timestamp = pd.Timestamp(tf_map[timeframe].index[-1])
            result[symbol][timeframe] = self.schedule.get(symbol, {}).get(timestamp, "HOLD")
        return result


class EmaCrossNoTrendStrategy(StrategyBase):
    def __init__(self, symbols: List[str], timeframes: List[str], config: Dict[str, Any]):
        super().__init__("Audit_EMA_Cross_No_Trend", symbols, timeframes, config)

    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        result: Dict[str, Dict[str, str]] = {}
        timeframe = self.timeframes[0]
        for symbol, tf_map in data.items():
            result[symbol] = {tf: "HOLD" for tf in tf_map}
            df = tf_map.get(timeframe)
            if df is None or len(df) < 25:
                continue
            close = df["close"]
            fast = close.ewm(span=8, adjust=False).mean()
            slow = close.ewm(span=21, adjust=False).mean()
            prev_fast = float(fast.iloc[-2])
            prev_slow = float(slow.iloc[-2])
            last_fast = float(fast.iloc[-1])
            last_slow = float(slow.iloc[-1])
            signal = "HOLD"
            if prev_fast <= prev_slow and last_fast > last_slow:
                signal = "OPEN_LONG"
            elif prev_fast >= prev_slow and last_fast < last_slow:
                signal = "OPEN_SHORT"
            result[symbol][timeframe] = signal
        return result


class UngatedImportedStrategy(ImportedNateemmaBatch1Strategy):
    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, str]]:
        result: Dict[str, Dict[str, str]] = {}
        cfg = self.variant_config
        for symbol, tf_map in data.items():
            result[symbol] = {timeframe: "HOLD" for timeframe in tf_map}
            entry_df = tf_map.get(cfg.entry_timeframe)
            if entry_df is None or entry_df.empty:
                continue
            raw_signal = self._compute_variant_signal(entry_df.copy())
            position_key = (symbol, cfg.entry_timeframe)
            final_signal = self._apply_position_rules(position_key, raw_signal)
            result[symbol][cfg.entry_timeframe] = final_signal
        return result


@dataclass
class TestResult:
    name: str
    status: str
    details: Dict[str, Any]


def _load_data(symbols: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp):
    feeder = DataFeeder(data_dir="data", memory_limit_percent=85)
    feeder.load_data(symbols, TIMEFRAMES, start_date=start_date, end_date=end_date)
    return feeder


def _run_backtest(
    strategy: StrategyBase,
    *,
    symbols: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    feeder = _load_data(symbols, start_date, end_date)
    engine = AuditBacktesterEngine(data_feeder=feeder, strategy=strategy, config=config)
    engine._save_backtest_results = lambda *args, **kwargs: None
    with contextlib.redirect_stdout(io.StringIO()):
        result = engine.run_backtest(symbols, TIMEFRAMES, start_date, end_date)
    return {
        "result": result,
        "final_balance": float(engine.balance),
        "trades": [asdict(trade) for trade in engine.performance_tracker.trades],
        "audit_stats": dict(engine.audit_stats),
        "global_rules": result.get("global_rules", {}),
    }


def _date_range(symbol: str = "BNBUSDT", timeframe: str = "15m") -> tuple[pd.Timestamp, pd.Timestamp]:
    df = pd.read_csv(Path("data") / f"{symbol}_{timeframe.rstrip('m')}.csv")
    start = pd.to_datetime(df["datetime"].iloc[0])
    end = pd.to_datetime(df["datetime"].iloc[-1])
    return max(start, end - timedelta(days=SCREEN_DAYS)), end


def _roundtrip_cost_pct(fee_pct: float, spread_pct: float, slippage_pct: float) -> float:
    return (2.0 * fee_pct) + (2.0 * spread_pct) + (2.0 * slippage_pct)


def _run_sanity_checks(start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[TestResult]:
    results: List[TestResult] = []
    feeder = _load_data(ONE_SYMBOL, start_date, end_date)
    price_df = feeder.data_cache["BNBUSDT"]["5m"]
    first_close = float(price_df["close"].iloc[0])
    last_close = float(price_df["close"].iloc[-1])
    market_return_pct = ((last_close / first_close) - 1.0) * 100.0

    base_config = {
        "processing_mode": "sequential",
        "batch_size": 2000,
        "memory_limit_percent": 85,
        "position_size_pct": 1.0,
        "max_positions": 1,
        "fee_pct": 0.0,
        "spread_pct": 0.0,
        "slippage_pct": 0.0,
    }

    long_strategy = BuyHoldLongStrategy(ONE_SYMBOL, ["5m"], {})
    long_run = _run_backtest(long_strategy, symbols=ONE_SYMBOL, start_date=start_date, end_date=end_date, config=base_config)
    long_return = float(long_run["result"]["total_return"])
    results.append(
        TestResult(
            name="buy_hold_long_zero_friction",
            status="PASS" if abs(long_return - market_return_pct) < 0.05 else "FAIL",
            details={
                "expected_market_return_pct": round(market_return_pct, 4),
                "backtester_return_pct": round(long_return, 4),
                "final_balance": round(long_run["final_balance"], 4),
                "trade_count": int(long_run["result"]["total_trades"]),
            },
        )
    )

    short_strategy = AlwaysShortStrategy(ONE_SYMBOL, ["5m"], {})
    short_run = _run_backtest(short_strategy, symbols=ONE_SYMBOL, start_date=start_date, end_date=end_date, config=base_config)
    short_return = float(short_run["result"]["total_return"])
    expected_short_return = -market_return_pct
    results.append(
        TestResult(
            name="always_short_zero_friction",
            status="PASS" if abs(short_return - expected_short_return) < 0.05 else "FAIL",
            details={
                "expected_return_pct": round(expected_short_return, 4),
                "backtester_return_pct": round(short_return, 4),
                "final_balance": round(short_run["final_balance"], 4),
                "trade_count": int(short_run["result"]["total_trades"]),
            },
        )
    )
    return results


def _run_scripted_parity(start_date: pd.Timestamp, end_date: pd.Timestamp) -> TestResult:
    feeder = _load_data(ONE_SYMBOL, start_date, end_date)
    df = feeder.data_cache["BNBUSDT"]["5m"].copy()
    idx_open_long = 60
    idx_close_long = 90
    idx_open_short = 120
    idx_close_short = 150
    schedule = {
        "BNBUSDT": {
            pd.Timestamp(df.index[idx_open_long]): "OPEN_LONG",
            pd.Timestamp(df.index[idx_close_long]): "CLOSE_LONG",
            pd.Timestamp(df.index[idx_open_short]): "OPEN_SHORT",
            pd.Timestamp(df.index[idx_close_short]): "CLOSE_SHORT",
        }
    }
    config = {
        "processing_mode": "sequential",
        "batch_size": 2000,
        "memory_limit_percent": 85,
        "position_size_pct": 0.1,
        "max_positions": 1,
        "fee_pct": 0.00055,
        "spread_pct": 0.0004,
        "slippage_pct": 0.0003,
    }
    strategy = ScriptedTradeStrategy(ONE_SYMBOL, ["5m"], {}, schedule=schedule)
    run = _run_backtest(strategy, symbols=ONE_SYMBOL, start_date=start_date, end_date=end_date, config=config)

    balance = 10000.0
    entry_long_raw = float(df["close"].iloc[idx_open_long])
    exit_long_raw = float(df["close"].iloc[idx_close_long])
    entry_short_raw = float(df["close"].iloc[idx_open_short])
    exit_short_raw = float(df["close"].iloc[idx_close_short])
    fee_pct = float(config["fee_pct"])
    spread_pct = float(config["spread_pct"])
    slippage_pct = float(config["slippage_pct"])

    entry_long = entry_long_raw * (1 + spread_pct + slippage_pct)
    qty_long = (balance * config["position_size_pct"]) / entry_long
    entry_long_fee = entry_long * qty_long * fee_pct
    balance -= entry_long_fee
    exit_long = exit_long_raw * (1 - spread_pct - slippage_pct)
    exit_long_fee = exit_long * qty_long * fee_pct
    long_pnl = ((exit_long - entry_long) * qty_long) - exit_long_fee
    balance += long_pnl

    entry_short = entry_short_raw * (1 - spread_pct - slippage_pct)
    qty_short = (balance * config["position_size_pct"]) / entry_short
    entry_short_fee = entry_short * qty_short * fee_pct
    balance -= entry_short_fee
    exit_short = exit_short_raw * (1 + spread_pct + slippage_pct)
    exit_short_fee = exit_short * qty_short * fee_pct
    short_pnl = ((entry_short - exit_short) * qty_short) - exit_short_fee
    balance += short_pnl

    expected_final_balance = balance
    reported_return = float(run["result"]["total_return"])
    actual_balance_return = ((run["final_balance"] - 10000.0) / 10000.0) * 100.0
    status = "PASS" if abs(run["final_balance"] - expected_final_balance) < 0.05 else "FAIL"
    return TestResult(
        name="scripted_trade_hand_calc_parity",
        status=status,
        details={
            "expected_final_balance": round(expected_final_balance, 4),
            "actual_final_balance": round(run["final_balance"], 4),
            "reported_total_return_pct": round(reported_return, 4),
            "actual_balance_return_pct": round(actual_balance_return, 4),
            "trade_count": int(run["result"]["total_trades"]),
            "note": "If reported return differs from actual balance return, fee reporting is wrong.",
        },
    )


def _run_ema_cross_no_opt(start_date: pd.Timestamp, end_date: pd.Timestamp) -> TestResult:
    config = {
        "processing_mode": "sequential",
        "batch_size": 2000,
        "memory_limit_percent": 85,
        "position_size_pct": 0.1,
        "max_positions": 3,
    }
    strategy = EmaCrossNoTrendStrategy(TEN_SYMBOLS, ["5m"], {})
    run = _run_backtest(strategy, symbols=TEN_SYMBOLS, start_date=start_date, end_date=end_date, config=config)
    status = "PASS" if int(run["result"]["total_trades"]) > 0 else "FAIL"
    return TestResult(
        name="ema_cross_no_optimization",
        status=status,
        details={
            "total_return_pct": round(float(run["result"]["total_return"]), 4),
            "final_balance": round(run["final_balance"], 4),
            "trade_count": int(run["result"]["total_trades"]),
            "max_drawdown_pct": round(float(run["result"]["max_drawdown"]), 4),
            "blocked_max_positions": int(run["audit_stats"]["blocked_max_positions"]),
        },
    )


def _run_import_diagnostics(start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[TestResult]:
    base_config = {
        "processing_mode": "sequential",
        "batch_size": 2000,
        "memory_limit_percent": 85,
    }

    tests: List[TestResult] = []
    cases = [
        (
            "import_mfi_current_settings",
            ImportedNateemmaBatch1Strategy("Anomaly_mfi", TEN_SYMBOLS, TIMEFRAMES, {}),
            dict(base_config),
        ),
        (
            "import_mfi_max_positions_10",
            ImportedNateemmaBatch1Strategy("Anomaly_mfi", TEN_SYMBOLS, TIMEFRAMES, {}),
            {**base_config, "max_positions": 10},
        ),
        (
            "import_mfi_max_positions_10_zero_friction",
            ImportedNateemmaBatch1Strategy("Anomaly_mfi", TEN_SYMBOLS, TIMEFRAMES, {}),
            {
                **base_config,
                "max_positions": 10,
                "fee_pct": 0.0,
                "spread_pct": 0.0,
                "slippage_pct": 0.0,
            },
        ),
        (
            "import_mfi_no_trend_gate_zero_friction",
            UngatedImportedStrategy("Anomaly_mfi", TEN_SYMBOLS, TIMEFRAMES, {}),
            {
                **base_config,
                "max_positions": 10,
                "fee_pct": 0.0,
                "spread_pct": 0.0,
                "slippage_pct": 0.0,
            },
        ),
    ]

    for name, strategy, config in cases:
        run = _run_backtest(strategy, symbols=TEN_SYMBOLS, start_date=start_date, end_date=end_date, config=config)
        tests.append(
            TestResult(
                name=name,
                status="INFO",
                details={
                    "total_return_pct": round(float(run["result"]["total_return"]), 4),
                    "final_balance": round(run["final_balance"], 4),
                    "trade_count": int(run["result"]["total_trades"]),
                    "win_rate_pct": round(float(run["result"]["win_rate"]), 4),
                    "max_drawdown_pct": round(float(run["result"]["max_drawdown"]), 4),
                    "blocked_max_positions": int(run["audit_stats"]["blocked_max_positions"]),
                    "blocked_symbol_already_open": int(run["audit_stats"]["blocked_symbol_already_open"]),
                },
            )
        )
    return tests


def _summarize_findings(results: List[TestResult]) -> Dict[str, Any]:
    mapped = {item.name: item for item in results}
    scripted = mapped["scripted_trade_hand_calc_parity"].details
    current_import = mapped["import_mfi_current_settings"].details
    maxpos_import = mapped["import_mfi_max_positions_10"].details
    zero_cost_import = mapped["import_mfi_max_positions_10_zero_friction"].details
    ungated_import = mapped["import_mfi_no_trend_gate_zero_friction"].details

    findings = []

    if mapped["buy_hold_long_zero_friction"].status == "PASS" and mapped["always_short_zero_friction"].status == "PASS":
        findings.append("Core long/short execution looks mechanically correct on zero-friction sanity tests.")
    else:
        findings.append("Core long/short execution failed the zero-friction sanity tests.")

    if mapped["scripted_trade_hand_calc_parity"].status == "PASS":
        findings.append("Trade execution matches a hand-calculated scripted example.")
    else:
        findings.append("Trade execution does not match the hand-calculated scripted example.")

    if abs(scripted["reported_total_return_pct"] - scripted["actual_balance_return_pct"]) > 0.05:
        findings.append("Reported total return is inconsistent with actual final balance when entry fees are present.")

    if current_import["blocked_max_positions"] > 0 and maxpos_import["total_return_pct"] > current_import["total_return_pct"]:
        findings.append("The default max_positions setting is materially blocking entries in multi-symbol tests.")

    if zero_cost_import["total_return_pct"] > current_import["total_return_pct"]:
        findings.append("Fees/spread/slippage materially reduce imported-strategy results, but do not fully explain the losses.")

    if ungated_import["total_return_pct"] > zero_cost_import["total_return_pct"]:
        findings.append("The trend gate reduces some valid entries for this imported strategy, but removing it still does not create a winner.")

    return {"findings": findings}


def _write_report(results: List[TestResult], summary: Dict[str, Any]) -> None:
    output_dir = Path("docs/backtester_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": pd.Timestamp.now(tz="Africa/Johannesburg").isoformat(),
        "results": [asdict(item) for item in results],
        "summary": summary,
    }
    json_path = output_dir / "backtester_diagnostics_20260329.json"
    md_path = output_dir / "backtester_diagnostics_20260329.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Backtester Diagnostics 2026-03-29",
        "",
        "## Findings",
    ]
    for finding in summary["findings"]:
        lines.append(f"- {finding}")
    lines.extend(["", "## Test Results", ""])
    for item in results:
        lines.append(f"### {item.name}")
        lines.append(f"- Status: `{item.status}`")
        for key, value in item.details.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    start_date, end_date = _date_range()
    results: List[TestResult] = []
    results.extend(_run_sanity_checks(start_date, end_date))
    results.append(_run_scripted_parity(start_date, end_date))
    results.append(_run_ema_cross_no_opt(start_date, end_date))
    results.extend(_run_import_diagnostics(start_date, end_date))
    summary = _summarize_findings(results)
    _write_report(results, summary)
    print(json.dumps({"results": [asdict(item) for item in results], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
