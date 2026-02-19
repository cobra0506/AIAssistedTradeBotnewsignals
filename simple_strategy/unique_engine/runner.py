import argparse
import csv
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from simple_strategy.auto_evolve.candidate_builder import CandidateBuilder
from simple_strategy.auto_evolve.reports import candidate_fingerprint, render_strategy_module
from simple_strategy.auto_evolve.search_space import load_search_space
from simple_strategy.fast_finder.runner import (
    DEFAULT_FAST_CONFIG,
    QUALITY_PRESETS,
    _build_advanced_search_space,
    run_fast_finder,
)


DEFAULT_UNIQUE_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "data_dir": "data",
    "symbols": ["BNBUSDT", "XRPUSDT", "ADAUSDT", "DOTUSDT", "NEARUSDT"],
    "timeframes": ["1", "5"],
    "start_date": "2026-01-01",
    "end_date": "2026-02-15",
    "candidate_count_per_blueprint": 80,
    "workers": 4,
    "top_k": 10,
    "publish_top_n": 3,
    "quality_preset": "A",
    "blueprints": "all",
    "search_space_path": "",
    "output_dir": "simple_strategy/unique_engine/runs",
    "enable_mtf_context_filter": True,
    "enable_regime_filter": True,
    "context_fast_period": 20,
    "context_slow_period": 50,
    "regime_symbol": "",
    "enable_train_test_split": True,
    "train_ratio": 0.70,
    "enable_gate_fallback": True,
    "min_return_pct": float(DEFAULT_FAST_CONFIG["min_return_pct"]),
    "max_drawdown_pct": float(DEFAULT_FAST_CONFIG["max_drawdown_pct"]),
    "min_win_rate_pct": float(DEFAULT_FAST_CONFIG["min_win_rate_pct"]),
    "min_trades": int(DEFAULT_FAST_CONFIG["min_trades"]),
    "min_test_return_pct": float(DEFAULT_FAST_CONFIG["min_test_return_pct"]),
    "max_test_drawdown_pct": float(DEFAULT_FAST_CONFIG["max_test_drawdown_pct"]),
    "min_test_win_rate_pct": float(DEFAULT_FAST_CONFIG["min_test_win_rate_pct"]),
    "min_test_trades": int(DEFAULT_FAST_CONFIG["min_test_trades"]),
    "fee_pct": float(DEFAULT_FAST_CONFIG["fee_pct"]),
    "spread_pct": float(DEFAULT_FAST_CONFIG["spread_pct"]),
    "slippage_pct": float(DEFAULT_FAST_CONFIG["slippage_pct"]),
    "max_positions": int(DEFAULT_FAST_CONFIG["max_positions"]),
    "risk_per_trade": float(DEFAULT_FAST_CONFIG["risk_per_trade"]),
}


UNIQUE_BLUEPRINTS: List[Dict[str, Any]] = [
    {
        "id": "mtf_trend_reversion",
        "name": "MTF Trend Reversion",
        "signals": ["rsi_trend_combo", "ema_cross", "macd_cross"],
        "required_signals": ["rsi_trend_combo"],
        "signal_combination": "and_signals",
        "enable_mtf_context_filter": True,
        "enable_regime_filter": False,
        "max_active_signals": 3,
    },
    {
        "id": "regime_breakout",
        "name": "Regime Breakout",
        "signals": ["breakout_sr", "ema_cross", "macd_cross"],
        "required_signals": ["breakout_sr"],
        "signal_combination": "and_signals",
        "enable_mtf_context_filter": True,
        "enable_regime_filter": True,
        "max_active_signals": 3,
    },
    {
        "id": "mean_reversion_cluster",
        "name": "Mean Reversion Cluster",
        "signals": ["bb_reversion", "stoch_reversal", "rsi_osob"],
        "required_signals": ["bb_reversion", "stoch_reversal"],
        "signal_combination": "majority_vote",
        "enable_mtf_context_filter": False,
        "enable_regime_filter": True,
        "max_active_signals": 3,
    },
    {
        "id": "cross_symbol_momentum",
        "name": "Cross Symbol Momentum",
        "signals": ["macd_cross", "ema_cross", "sma_cross"],
        "required_signals": ["macd_cross", "ema_cross"],
        "signal_combination": "and_signals",
        "enable_mtf_context_filter": True,
        "enable_regime_filter": True,
        "max_active_signals": 3,
    },
]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _normalize_blueprint_selection(raw_value: Any) -> List[str]:
    selected = _as_list(raw_value)
    if not selected:
        return [bp["id"] for bp in UNIQUE_BLUEPRINTS]

    lowered = [item.lower() for item in selected]
    if "all" in lowered:
        return [bp["id"] for bp in UNIQUE_BLUEPRINTS]
    return lowered


def _apply_quality_preset(run_config: Dict[str, Any], explicit_override: Dict[str, Any]) -> Dict[str, Any]:
    preset_raw = str(run_config.get("quality_preset", "custom")).strip().upper()
    if preset_raw not in QUALITY_PRESETS:
        return run_config

    updated = deepcopy(run_config)
    preset_values = QUALITY_PRESETS[preset_raw]
    for key, value in preset_values.items():
        if key not in explicit_override:
            updated[key] = value
    return updated


def _build_blueprint_lookup() -> Dict[str, Dict[str, Any]]:
    return {entry["id"].lower(): deepcopy(entry) for entry in UNIQUE_BLUEPRINTS}


def _build_blueprint_search_space(base_search_space: Dict[str, Any], blueprint: Dict[str, Any]) -> Dict[str, Any]:
    search_space = deepcopy(base_search_space)
    allowed_signals = [sig for sig in blueprint.get("signals", []) if sig in search_space.get("signals", {})]
    if not allowed_signals:
        raise ValueError(f"Blueprint '{blueprint['id']}' has no signals available in search space.")

    search_space["signals"] = {sid: search_space["signals"][sid] for sid in allowed_signals}

    preferred_combo = str(blueprint.get("signal_combination", "majority_vote")).strip()
    combos = list(search_space.get("signal_combination_methods", []))
    if preferred_combo:
        combos = [preferred_combo] + [entry for entry in combos if entry != preferred_combo]
    search_space["signal_combination_methods"] = combos or ["majority_vote"]
    return search_space


def _row_sort_key(row: Dict[str, Any]) -> Tuple[float, str, int]:
    return (
        -float(row.get("score", -1e9)),
        str(row.get("blueprint_id", "")),
        int(row.get("index", 10**9)),
    )


def _select_unique_top_rows(
    all_rows: List[Dict[str, Any]],
    top_k: int,
    search_space: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()

    fallback: Dict[str, Any] = {
        "triggered": False,
        "reason": "",
        "strict_blueprint_match_count": 0,
    }

    ranked = sorted(all_rows, key=_row_sort_key)
    strict_rows = [row for row in ranked if bool(row.get("blueprint_match", False))]
    fallback["strict_blueprint_match_count"] = len(strict_rows)

    if len(strict_rows) < top_k:
        fallback["triggered"] = True
        fallback["reason"] = "insufficient_strict_blueprint_matches"

    def _append_rows(rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            fp = candidate_fingerprint(row.get("candidate", {}), search_space=search_space)
            if fp in seen:
                continue
            seen.add(fp)
            selected.append(row)
            if len(selected) >= top_k:
                break

    _append_rows(strict_rows)
    if len(selected) < top_k:
        _append_rows(ranked)

    return selected[:top_k], fallback


def _write_reports(
    run_dir: Path,
    run_config: Dict[str, Any],
    all_rows: List[Dict[str, Any]],
    top_rows: List[Dict[str, Any]],
    blueprint_runs: List[Dict[str, Any]],
    fallback: Dict[str, Any],
    elapsed_seconds: float,
    stopped: bool,
) -> None:
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "stopped": bool(stopped),
        "elapsed_seconds": float(elapsed_seconds),
        "config": run_config,
        "evaluated": len(all_rows),
        "fallback": fallback,
        "blueprint_runs": blueprint_runs,
        "top_results": top_rows,
    }
    (run_dir / "top_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_lines: List[str] = []
    summary_lines.append("Unique Advanced Strategy Engine Summary")
    summary_lines.append("======================================")
    summary_lines.append(f"Generated at: {payload['generated_at']}")
    summary_lines.append(f"Stopped: {stopped}")
    summary_lines.append(f"Elapsed seconds: {elapsed_seconds:.2f}")
    summary_lines.append(f"Evaluated: {len(all_rows)}")
    summary_lines.append(
        "Fallback: triggered={triggered} reason={reason} strict_blueprint_match_count={strict}".format(
            triggered=bool(fallback.get("triggered", False)),
            reason=str(fallback.get("reason", "")),
            strict=int(fallback.get("strict_blueprint_match_count", 0)),
        )
    )
    summary_lines.append("")

    for rank, row in enumerate(top_rows, start=1):
        test_metrics = row.get("metrics", {})
        train_metrics = row.get("train_metrics", {})
        summary_lines.append(
            "#{rank} blueprint={blueprint} score={score:.4f} pass={passed} "
            "test_return={test_ret:.2f}% test_dd={test_dd:.2f}% test_win={test_win:.2f}% test_trades={test_trades} "
            "train_return={train_ret:.2f}% train_dd={train_dd:.2f}% signals={signals}".format(
                rank=rank,
                blueprint=str(row.get("blueprint_id", "")),
                score=float(row.get("score", 0.0)),
                passed=bool(row.get("passed", False)),
                test_ret=float(test_metrics.get("total_return", 0.0)),
                test_dd=float(test_metrics.get("max_drawdown", 0.0)),
                test_win=float(test_metrics.get("win_rate", 0.0)),
                test_trades=int(test_metrics.get("total_trades", 0)),
                train_ret=float(train_metrics.get("total_return", 0.0)),
                train_dd=float(train_metrics.get("max_drawdown", 0.0)),
                signals=row.get("candidate", {}).get("active_signals", []),
            )
        )
    (run_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    with (run_dir / "top_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rank",
                "blueprint_id",
                "blueprint_name",
                "blueprint_match",
                "score",
                "passed",
                "passed_relaxed",
                "train_passed",
                "test_passed",
                "test_return_pct",
                "test_max_drawdown_pct",
                "test_win_rate_pct",
                "test_trades",
                "test_sharpe_ratio",
                "train_return_pct",
                "train_max_drawdown_pct",
                "train_win_rate_pct",
                "train_trades",
                "signals",
                "combination",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(top_rows, start=1):
            test_metrics = row.get("metrics", {})
            train_metrics = row.get("train_metrics", {})
            candidate = row.get("candidate", {})
            writer.writerow(
                {
                    "rank": rank,
                    "blueprint_id": str(row.get("blueprint_id", "")),
                    "blueprint_name": str(row.get("blueprint_name", "")),
                    "blueprint_match": bool(row.get("blueprint_match", False)),
                    "score": float(row.get("score", 0.0)),
                    "passed": bool(row.get("passed", False)),
                    "passed_relaxed": bool(row.get("passed_relaxed", False)),
                    "train_passed": bool(row.get("train_passed", False)),
                    "test_passed": bool(row.get("test_passed", False)),
                    "test_return_pct": float(test_metrics.get("total_return", 0.0)),
                    "test_max_drawdown_pct": float(test_metrics.get("max_drawdown", 0.0)),
                    "test_win_rate_pct": float(test_metrics.get("win_rate", 0.0)),
                    "test_trades": int(test_metrics.get("total_trades", 0)),
                    "test_sharpe_ratio": float(test_metrics.get("sharpe_ratio", 0.0)),
                    "train_return_pct": float(train_metrics.get("total_return", 0.0)),
                    "train_max_drawdown_pct": float(train_metrics.get("max_drawdown", 0.0)),
                    "train_win_rate_pct": float(train_metrics.get("win_rate", 0.0)),
                    "train_trades": int(train_metrics.get("total_trades", 0)),
                    "signals": ",".join(candidate.get("active_signals", [])),
                    "combination": candidate.get("signal_combination", "majority_vote"),
                }
            )


def _export_strategy_files(
    run_dir: Path,
    top_rows: List[Dict[str, Any]],
    run_config: Dict[str, Any],
    base_search_space: Dict[str, Any],
) -> List[str]:
    run_suffix = run_dir.name.replace("run_", "")
    strategies_dir = run_dir / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[2]
    publish_dir = project_root / "simple_strategy" / "strategies"
    publish_dir.mkdir(parents=True, exist_ok=True)
    publish_top_n = max(0, int(run_config.get("publish_top_n", 0)))

    candidate_builder = CandidateBuilder(
        search_space=base_search_space,
        max_active_signals=int(run_config.get("max_active_signals", 3)),
    )

    published_files: List[str] = []
    published_count = 0
    for rank, row in enumerate(top_rows, start=1):
        module_name = f"Strategy_UNIQUE_{run_suffix}_{rank:02d}"
        module_text = render_strategy_module(
            strategy_name=module_name,
            candidate=row.get("candidate", {}),
            candidate_builder=candidate_builder,
            symbols=run_config["symbols"],
            timeframes=run_config["timeframes"],
        )
        run_module_path = strategies_dir / f"{module_name}.py"
        run_module_path.write_text(module_text, encoding="utf-8")

        qualifies = bool(row.get("blueprint_match", False) and row.get("passed", False))
        if qualifies and published_count < publish_top_n:
            target_path = publish_dir / f"{module_name}.py"
            target_path.write_text(module_text, encoding="utf-8")
            published_files.append(str(target_path))
            published_count += 1

    return published_files


def run_unique_engine(
    config_override: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, int, float, Dict[str, Any]], None]] = None,
    stop_event: Optional[Any] = None,
) -> Dict[str, Any]:
    override = config_override or {}
    run_config = _deep_merge(DEFAULT_UNIQUE_CONFIG, override)
    run_config = _apply_quality_preset(run_config, explicit_override=override)

    run_config["symbols"] = _as_list(run_config.get("symbols", []))
    run_config["timeframes"] = _as_list(run_config.get("timeframes", []))
    if not run_config["symbols"]:
        raise ValueError("At least one symbol is required.")
    if not run_config["timeframes"]:
        raise ValueError("At least one timeframe is required.")

    blueprint_lookup = _build_blueprint_lookup()
    selected_ids = _normalize_blueprint_selection(run_config.get("blueprints", "all"))
    selected_blueprints: List[Dict[str, Any]] = []
    for bp_id in selected_ids:
        entry = blueprint_lookup.get(bp_id.lower())
        if entry is not None:
            selected_blueprints.append(entry)
    if not selected_blueprints:
        raise ValueError("No valid blueprints selected.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(run_config["output_dir"]) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "search_spaces").mkdir(parents=True, exist_ok=True)

    base_space = load_search_space(str(run_config.get("search_space_path", "")).strip())
    advanced_space = _build_advanced_search_space(base_space)

    candidate_count = max(1, int(run_config.get("candidate_count_per_blueprint", 1)))
    requested_total = candidate_count * len(selected_blueprints)
    completed_total = 0
    global_best = -1e9
    all_rows: List[Dict[str, Any]] = []
    blueprint_runs: List[Dict[str, Any]] = []

    started = time.perf_counter()
    for blueprint in selected_blueprints:
        if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
            break

        bp_id = str(blueprint["id"])
        bp_name = str(blueprint.get("name", bp_id))
        bp_required = list(blueprint.get("required_signals", []))

        bp_space = _build_blueprint_search_space(advanced_space, blueprint)
        bp_space_path = run_dir / "search_spaces" / f"{bp_id}.json"
        bp_space_path.write_text(json.dumps(bp_space, indent=2), encoding="utf-8")

        sub_config = {
            "seed": int(run_config.get("seed", 42)),
            "data_dir": run_config["data_dir"],
            "symbols": run_config["symbols"],
            "timeframes": run_config["timeframes"],
            "start_date": run_config["start_date"],
            "end_date": run_config["end_date"],
            "candidate_count": candidate_count,
            "workers": int(run_config.get("workers", 1)),
            "top_k": candidate_count,
            "publish_top_n": 0,
            "quality_preset": run_config.get("quality_preset", "custom"),
            "search_profile": "advanced",
            "search_space_path": str(bp_space_path),
            "output_dir": str(run_dir / "subruns" / bp_id),
            "max_active_signals": int(blueprint.get("max_active_signals", run_config.get("max_active_signals", 3))),
            "enable_mtf_context_filter": bool(
                blueprint.get("enable_mtf_context_filter", run_config.get("enable_mtf_context_filter", True))
            ),
            "enable_regime_filter": bool(
                blueprint.get("enable_regime_filter", run_config.get("enable_regime_filter", True))
            ),
            "context_fast_period": int(run_config.get("context_fast_period", 20)),
            "context_slow_period": int(run_config.get("context_slow_period", 50)),
            "regime_symbol": str(run_config.get("regime_symbol", "")),
            "min_return_pct": float(run_config.get("min_return_pct", 0.0)),
            "max_drawdown_pct": float(run_config.get("max_drawdown_pct", 100.0)),
            "min_win_rate_pct": float(run_config.get("min_win_rate_pct", 0.0)),
            "min_trades": int(run_config.get("min_trades", 0)),
            "min_test_return_pct": float(run_config.get("min_test_return_pct", 0.0)),
            "max_test_drawdown_pct": float(run_config.get("max_test_drawdown_pct", 100.0)),
            "min_test_win_rate_pct": float(run_config.get("min_test_win_rate_pct", 0.0)),
            "min_test_trades": int(run_config.get("min_test_trades", 0)),
            "enable_train_test_split": bool(run_config.get("enable_train_test_split", True)),
            "train_ratio": float(run_config.get("train_ratio", 0.70)),
            "enable_gate_fallback": bool(run_config.get("enable_gate_fallback", True)),
            "fee_pct": float(run_config.get("fee_pct", DEFAULT_FAST_CONFIG["fee_pct"])),
            "spread_pct": float(run_config.get("spread_pct", DEFAULT_FAST_CONFIG["spread_pct"])),
            "slippage_pct": float(run_config.get("slippage_pct", DEFAULT_FAST_CONFIG["slippage_pct"])),
            "max_positions": int(run_config.get("max_positions", DEFAULT_FAST_CONFIG["max_positions"])),
            "risk_per_trade": float(run_config.get("risk_per_trade", DEFAULT_FAST_CONFIG["risk_per_trade"])),
        }

        def _progress(done: int, total: int, best_score: float, row: Dict[str, Any]) -> None:
            nonlocal global_best
            row_score = float(row.get("score", best_score))
            global_best = max(global_best, row_score)
            wrapped_row = dict(row)
            wrapped_row["blueprint_id"] = bp_id
            wrapped_row["blueprint_name"] = bp_name
            if progress_callback is not None:
                progress_callback(
                    min(requested_total, completed_total + done),
                    requested_total,
                    global_best,
                    wrapped_row,
                )

        try:
            sub_result = run_fast_finder(
                config_override=sub_config,
                progress_callback=_progress,
                stop_event=stop_event,
            )
        except Exception as exc:
            blueprint_runs.append(
                {
                    "blueprint_id": bp_id,
                    "blueprint_name": bp_name,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            continue

        evaluated = int(sub_result.get("evaluated", 0))
        completed_total += evaluated

        for row in sub_result.get("top_results", []):
            candidate = row.get("candidate", {})
            active = set(candidate.get("active_signals", []))
            blueprint_match = all(required in active for required in bp_required)
            wrapped = deepcopy(row)
            wrapped["blueprint_id"] = bp_id
            wrapped["blueprint_name"] = bp_name
            wrapped["blueprint_required_signals"] = bp_required
            wrapped["blueprint_match"] = bool(blueprint_match)
            all_rows.append(wrapped)

        blueprint_runs.append(
            {
                "blueprint_id": bp_id,
                "blueprint_name": bp_name,
                "status": "stopped" if bool(sub_result.get("stopped", False)) else "complete",
                "run_dir": sub_result.get("run_dir", ""),
                "evaluated": evaluated,
            }
        )

        if bool(sub_result.get("stopped", False)):
            break

    elapsed = time.perf_counter() - started
    stopped = bool(stop_event is not None and getattr(stop_event, "is_set", lambda: False)())

    if not all_rows:
        raise RuntimeError("Unique engine did not produce any candidate results.")

    top_k = max(1, int(run_config.get("top_k", 1)))
    top_rows, fallback = _select_unique_top_rows(
        all_rows=all_rows,
        top_k=top_k,
        search_space=advanced_space,
    )

    _write_reports(
        run_dir=run_dir,
        run_config=run_config,
        all_rows=all_rows,
        top_rows=top_rows,
        blueprint_runs=blueprint_runs,
        fallback=fallback,
        elapsed_seconds=elapsed,
        stopped=stopped,
    )
    published_files = _export_strategy_files(
        run_dir=run_dir,
        top_rows=top_rows,
        run_config=run_config,
        base_search_space=advanced_space,
    )

    return {
        "run_dir": str(run_dir),
        "elapsed_seconds": float(elapsed),
        "evaluated": int(completed_total),
        "requested_candidates": int(requested_total),
        "top_results": top_rows,
        "fallback": fallback,
        "blueprint_runs": blueprint_runs,
        "published_strategy_files": published_files,
        "stopped": stopped,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unique advanced strategy engine")
    parser.add_argument("--quality-preset", choices=["custom", "A", "B"], default=DEFAULT_UNIQUE_CONFIG["quality_preset"])
    parser.add_argument("--blueprints", default=DEFAULT_UNIQUE_CONFIG["blueprints"])
    parser.add_argument("--symbols", default="BNBUSDT,XRPUSDT,ADAUSDT,DOTUSDT,NEARUSDT")
    parser.add_argument("--timeframes", default="1,5")
    parser.add_argument("--start-date", default=DEFAULT_UNIQUE_CONFIG["start_date"])
    parser.add_argument("--end-date", default=DEFAULT_UNIQUE_CONFIG["end_date"])
    parser.add_argument(
        "--candidate-count-per-blueprint",
        type=int,
        default=DEFAULT_UNIQUE_CONFIG["candidate_count_per_blueprint"],
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_UNIQUE_CONFIG["workers"])
    parser.add_argument("--top-k", type=int, default=DEFAULT_UNIQUE_CONFIG["top_k"])
    parser.add_argument("--publish-top-n", type=int, default=DEFAULT_UNIQUE_CONFIG["publish_top_n"])
    parser.add_argument("--min-return-pct", type=float, default=DEFAULT_UNIQUE_CONFIG["min_return_pct"])
    parser.add_argument("--max-drawdown-pct", type=float, default=DEFAULT_UNIQUE_CONFIG["max_drawdown_pct"])
    parser.add_argument("--min-win-rate-pct", type=float, default=DEFAULT_UNIQUE_CONFIG["min_win_rate_pct"])
    parser.add_argument("--min-trades", type=int, default=DEFAULT_UNIQUE_CONFIG["min_trades"])
    parser.add_argument("--min-test-return-pct", type=float, default=DEFAULT_UNIQUE_CONFIG["min_test_return_pct"])
    parser.add_argument("--max-test-drawdown-pct", type=float, default=DEFAULT_UNIQUE_CONFIG["max_test_drawdown_pct"])
    parser.add_argument("--min-test-win-rate-pct", type=float, default=DEFAULT_UNIQUE_CONFIG["min_test_win_rate_pct"])
    parser.add_argument("--min-test-trades", type=int, default=DEFAULT_UNIQUE_CONFIG["min_test_trades"])
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_UNIQUE_CONFIG["train_ratio"])
    parser.add_argument("--disable-train-test-split", action="store_true")
    parser.add_argument("--disable-gate-fallback", action="store_true")
    parser.add_argument("--data-dir", default=DEFAULT_UNIQUE_CONFIG["data_dir"])
    parser.add_argument("--output-dir", default=DEFAULT_UNIQUE_CONFIG["output_dir"])
    parser.add_argument("--search-space-path", default="")
    parser.add_argument("--seed", type=int, default=DEFAULT_UNIQUE_CONFIG["seed"])
    parser.add_argument("--enable-mtf-context-filter", action="store_true", default=None)
    parser.add_argument("--enable-regime-filter", action="store_true", default=None)
    parser.add_argument("--context-fast-period", type=int, default=DEFAULT_UNIQUE_CONFIG["context_fast_period"])
    parser.add_argument("--context-slow-period", type=int, default=DEFAULT_UNIQUE_CONFIG["context_slow_period"])
    parser.add_argument("--regime-symbol", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = {
        "quality_preset": args.quality_preset,
        "blueprints": args.blueprints,
        "symbols": args.symbols,
        "timeframes": args.timeframes,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "candidate_count_per_blueprint": args.candidate_count_per_blueprint,
        "workers": args.workers,
        "top_k": args.top_k,
        "publish_top_n": args.publish_top_n,
        "min_return_pct": args.min_return_pct,
        "max_drawdown_pct": args.max_drawdown_pct,
        "min_win_rate_pct": args.min_win_rate_pct,
        "min_trades": args.min_trades,
        "min_test_return_pct": args.min_test_return_pct,
        "max_test_drawdown_pct": args.max_test_drawdown_pct,
        "min_test_win_rate_pct": args.min_test_win_rate_pct,
        "min_test_trades": args.min_test_trades,
        "train_ratio": args.train_ratio,
        "enable_train_test_split": not bool(args.disable_train_test_split),
        "enable_gate_fallback": not bool(args.disable_gate_fallback),
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "search_space_path": args.search_space_path,
        "seed": args.seed,
        "context_fast_period": args.context_fast_period,
        "context_slow_period": args.context_slow_period,
        "regime_symbol": args.regime_symbol,
    }
    if args.enable_mtf_context_filter is not None:
        config["enable_mtf_context_filter"] = args.enable_mtf_context_filter
    if args.enable_regime_filter is not None:
        config["enable_regime_filter"] = args.enable_regime_filter

    result = run_unique_engine(config_override=config)
    print(f"Run directory: {result['run_dir']}")
    print(f"Evaluated: {result['evaluated']}/{result['requested_candidates']}")
    print(f"Elapsed seconds: {result['elapsed_seconds']:.2f}")
    if result["top_results"]:
        best = result["top_results"][0]
        metrics = best.get("metrics", {})
        print(
            "Best blueprint={bp} score={score:.4f} return={ret:.2f}% dd={dd:.2f}% win={win:.2f}% trades={trades}".format(
                bp=str(best.get("blueprint_id", "")),
                score=float(best.get("score", 0.0)),
                ret=float(metrics.get("total_return", 0.0)),
                dd=float(metrics.get("max_drawdown", 0.0)),
                win=float(metrics.get("win_rate", 0.0)),
                trades=int(metrics.get("total_trades", 0)),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
