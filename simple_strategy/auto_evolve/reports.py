import json
import csv
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .candidate_builder import CandidateBuilder
from .checkpoint import save_json


def _component_indicator_map(search_space: Dict[str, Any]) -> Dict[str, List[str]]:
    component_map: Dict[str, List[str]] = {}
    for indicator_id, indicator_def in search_space.get("indicators", {}).items():
        func_name = indicator_def.get("function")
        components: List[str] = []
        if func_name == "macd":
            components = ["macd_line", "signal_line", "histogram"]
        elif func_name == "bollinger_bands":
            components = ["upper_band", "middle_band", "lower_band"]
        elif func_name == "stochastic":
            components = ["k_percent", "d_percent"]
        for component in components:
            component_map.setdefault(component, []).append(indicator_id)
    return component_map


def _required_indicators(candidate: Dict[str, Any], search_space: Dict[str, Any]) -> List[str]:
    indicators = search_space.get("indicators", {})
    signals = search_space.get("signals", {})
    component_map = _component_indicator_map(search_space)

    required = set()
    for signal_id in candidate.get("active_signals", []):
        signal_def = signals.get(signal_id, {})
        for input_ref in signal_def.get("inputs", {}).values():
            if input_ref == "price":
                continue
            if input_ref in indicators:
                required.add(input_ref)
            elif input_ref in component_map:
                for mapped_indicator in component_map[input_ref]:
                    required.add(mapped_indicator)

    return sorted(required)


def _canonical_candidate(candidate: Dict[str, Any], search_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    active_signals = sorted(list(dict.fromkeys(candidate.get("active_signals", []))))
    combination = candidate.get("signal_combination", "majority_vote")

    canonical: Dict[str, Any] = {
        "active_signals": active_signals,
        "signal_combination": combination,
        "indicator_params": {},
        "signal_params": {},
    }

    if search_space:
        keep_indicators = _required_indicators(candidate, search_space)
        for indicator_id in keep_indicators:
            canonical["indicator_params"][indicator_id] = candidate.get("indicator_params", {}).get(indicator_id, {})
    else:
        canonical["indicator_params"] = candidate.get("indicator_params", {})

    for signal_id in active_signals:
        canonical["signal_params"][signal_id] = candidate.get("signal_params", {}).get(signal_id, {})

    return canonical


def candidate_fingerprint(candidate: Dict[str, Any], search_space: Optional[Dict[str, Any]] = None) -> str:
    canonical = _canonical_candidate(candidate, search_space=search_space)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def top_unique_results(
    results: List[Dict[str, Any]],
    top_k: int,
    search_space: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []

    ranked = sorted(results, key=lambda r: float(r.get("score", -1e9)), reverse=True)
    for row in ranked:
        key = candidate_fingerprint(row["candidate"], search_space=search_space)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
        if len(unique) >= top_k:
            break

    return unique


def write_reports(
    run_dir: Path,
    top_results: List[Dict[str, Any]],
    run_config: Dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": run_config,
        "top_results": top_results,
    }
    save_json(run_dir / "top_results.json", payload)

    lines = []
    lines.append("Auto Evolution Summary")
    lines.append("======================")
    lines.append(f"Generated at: {payload['generated_at']}")
    lines.append(f"Top results: {len(top_results)}")
    lines.append("")

    for rank, row in enumerate(top_results, start=1):
        metrics = row.get("metrics", {})
        lines.append(f"#{rank} | score={row.get('score', 0.0):.4f}")
        lines.append(f"  signals: {row.get('candidate', {}).get('active_signals', [])}")
        lines.append(
            "  train: return={:.2f}% sharpe={:.3f} drawdown={:.2f}% trades={}".format(
                float(metrics.get("total_return", 0.0)),
                float(metrics.get("sharpe_ratio", 0.0)),
                float(metrics.get("max_drawdown", 0.0)),
                int(metrics.get("total_trades", 0)),
            )
        )
        lines.append("")

    (run_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    metric_columns = [
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "win_rate",
        "total_trades",
    ]

    txt_lines: List[str] = []
    txt_lines.append("Top 10 Metrics Summary")
    txt_lines.append("======================")
    txt_lines.append("Columns: profit%, max_drawdown%, sharpe, win_rate%, trades")
    txt_lines.append("")

    csv_headers = [
        "rank",
        "passed",
        "score",
        "final_score",
        "signals",
        "failed_gate",
        "failure_reason",
        "train_profit_pct",
        "train_max_drawdown_pct",
        "train_sharpe",
        "train_win_rate_pct",
        "train_trades",
        "validation_profit_pct",
        "validation_max_drawdown_pct",
        "validation_sharpe",
        "validation_win_rate_pct",
        "validation_trades",
        "final_profit_pct",
        "final_max_drawdown_pct",
        "final_sharpe",
        "final_win_rate_pct",
        "final_trades",
    ]
    csv_rows: List[Dict[str, Any]] = []

    def _as_metric_block(metrics: Dict[str, Any]) -> Dict[str, float]:
        return {
            key: float(metrics.get(key, 0.0))
            for key in metric_columns
        }

    for rank, row in enumerate(top_results, start=1):
        train = _as_metric_block(row.get("metrics", {}))
        validation = _as_metric_block(row.get("validation_metrics", {}))
        final = _as_metric_block(row.get("final_metrics", {}))
        signals = row.get("candidate", {}).get("active_signals", [])

        txt_lines.append(f"#{rank} | passed={bool(row.get('passed', False))} | score={float(row.get('score', 0.0)):.4f}")
        txt_lines.append(f"  signals: {signals}")
        txt_lines.append(
            "  train: profit={:.2f}% drawdown={:.2f}% sharpe={:.3f} win_rate={:.2f}% trades={}".format(
                train["total_return"],
                train["max_drawdown"],
                train["sharpe_ratio"],
                train["win_rate"],
                int(train["total_trades"]),
            )
        )
        txt_lines.append(
            "  validation: profit={:.2f}% drawdown={:.2f}% sharpe={:.3f} win_rate={:.2f}% trades={}".format(
                validation["total_return"],
                validation["max_drawdown"],
                validation["sharpe_ratio"],
                validation["win_rate"],
                int(validation["total_trades"]),
            )
        )
        txt_lines.append(
            "  final: profit={:.2f}% drawdown={:.2f}% sharpe={:.3f} win_rate={:.2f}% trades={}".format(
                final["total_return"],
                final["max_drawdown"],
                final["sharpe_ratio"],
                final["win_rate"],
                int(final["total_trades"]),
            )
        )
        txt_lines.append("")

        csv_rows.append(
            {
                "rank": rank,
                "passed": bool(row.get("passed", False)),
                "score": float(row.get("score", 0.0)),
                "final_score": float(row.get("final_score", row.get("score", 0.0))),
                "signals": ",".join(str(s) for s in signals),
                "failed_gate": row.get("failed_gate", ""),
                "failure_reason": row.get("failure_reason", ""),
                "train_profit_pct": train["total_return"],
                "train_max_drawdown_pct": train["max_drawdown"],
                "train_sharpe": train["sharpe_ratio"],
                "train_win_rate_pct": train["win_rate"],
                "train_trades": int(train["total_trades"]),
                "validation_profit_pct": validation["total_return"],
                "validation_max_drawdown_pct": validation["max_drawdown"],
                "validation_sharpe": validation["sharpe_ratio"],
                "validation_win_rate_pct": validation["win_rate"],
                "validation_trades": int(validation["total_trades"]),
                "final_profit_pct": final["total_return"],
                "final_max_drawdown_pct": final["max_drawdown"],
                "final_sharpe": final["sharpe_ratio"],
                "final_win_rate_pct": final["win_rate"],
                "final_trades": int(final["total_trades"]),
            }
        )

    (run_dir / "top10_summary.txt").write_text("\n".join(txt_lines), encoding="utf-8")
    with (run_dir / "top10_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(csv_rows)


def export_strategy_files(
    run_dir: Path,
    candidate_builder: CandidateBuilder,
    top_results: List[Dict[str, Any]],
    symbols: List[str],
    timeframes: List[str],
) -> None:
    out_dir = run_dir / "strategies"
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, row in enumerate(top_results, start=1):
        candidate = row["candidate"]
        module_text = render_strategy_module(
            strategy_name=f"Strategy_AE_Top_{rank:02d}",
            candidate=candidate,
            candidate_builder=candidate_builder,
            symbols=symbols,
            timeframes=timeframes,
        )
        (out_dir / f"Strategy_AE_Top_{rank:02d}.py").write_text(module_text, encoding="utf-8")


def render_strategy_module(
    strategy_name: str,
    candidate: Dict[str, Any],
    candidate_builder: CandidateBuilder,
    symbols: List[str],
    timeframes: List[str],
) -> str:
    search_space = candidate_builder.search_space
    required = candidate_builder.required_indicators(candidate)

    indicator_funcs = sorted({search_space["indicators"][iid]["function"] for iid in required})
    signal_funcs = sorted({search_space["signals"][sid]["function"] for sid in candidate.get("active_signals", [])})

    indicator_import = ", ".join(indicator_funcs) if indicator_funcs else ""
    signal_import = ", ".join(signal_funcs) if signal_funcs else ""

    lines: List[str] = []
    lines.append(f'"""Generated strategy: {strategy_name}"""')
    lines.append("from simple_strategy.strategies.strategy_builder import StrategyBuilder")
    if indicator_import:
        lines.append(f"from simple_strategy.strategies.indicators_library import {indicator_import}")
    if signal_import:
        lines.append(f"from simple_strategy.strategies.signals_library import {signal_import}")
    lines.append("")
    lines.append(f"STRATEGY_PARAMETERS = {{}}")
    lines.append("")
    lines.append("def create_strategy(symbols=None, timeframes=None, **params):")
    lines.append(f"    symbols = symbols or {symbols!r}")
    lines.append(f"    timeframes = timeframes or {timeframes!r}")
    lines.append("    builder = StrategyBuilder(symbols=symbols, timeframes=timeframes)")

    for indicator_id in required:
        func_name = search_space["indicators"][indicator_id]["function"]
        param_map = candidate["indicator_params"].get(indicator_id, {})
        args = ", ".join([f"{k}={v!r}" for k, v in param_map.items()])
        lines.append(f"    builder.add_indicator('{indicator_id}', {func_name}, {args})")

    for signal_id in candidate.get("active_signals", []):
        sdef = search_space["signals"][signal_id]
        func_name = sdef["function"]
        kwargs = {}
        kwargs.update(sdef.get("inputs", {}))
        kwargs.update(candidate["signal_params"].get(signal_id, {}))
        args = ", ".join([f"{k}={v!r}" for k, v in kwargs.items()])
        lines.append(f"    builder.add_signal_rule('{signal_id}', {func_name}, {args})")

    combination = candidate.get("signal_combination", "majority_vote")
    lines.append(f"    builder.set_signal_combination('{combination}')")
    lines.append(f"    builder.set_strategy_info('{strategy_name}', '1.0.0')")
    lines.append("    return builder.build()")
    lines.append("")

    return "\n".join(lines)
