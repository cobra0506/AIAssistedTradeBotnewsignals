import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_registry(registry_path: str) -> Dict[str, Any]:
    path = Path(registry_path)
    if not path.exists():
        return {"champion": None, "history": []}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {"champion": None, "history": []}
    payload.setdefault("champion", None)
    payload.setdefault("history", [])
    return payload


def save_registry(registry_path: str, payload: Dict[str, Any]) -> Path:
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def should_promote(
    candidate_metrics: Dict[str, Any],
    champion_metrics: Dict[str, Any] | None,
    min_reward_improvement: float = 0.0,
    max_drawdown_pct: float = 0.30,
) -> tuple[bool, str]:
    candidate_reward = _safe_float(candidate_metrics.get("total_reward"), default=-1e9)
    candidate_drawdown = _safe_float(candidate_metrics.get("max_drawdown_pct"), default=1.0)

    if candidate_drawdown > max_drawdown_pct:
        return False, "candidate_drawdown_above_limit"

    if champion_metrics is None:
        return True, "no_existing_champion"

    champion_reward = _safe_float(champion_metrics.get("total_reward"), default=-1e9)
    champion_drawdown = _safe_float(champion_metrics.get("max_drawdown_pct"), default=1.0)

    reward_ok = candidate_reward >= (champion_reward + float(min_reward_improvement))
    drawdown_ok = candidate_drawdown <= max(champion_drawdown, max_drawdown_pct)

    if reward_ok and drawdown_ok:
        return True, "better_reward_and_safe_drawdown"
    if not reward_ok:
        return False, "insufficient_reward_improvement"
    return False, "drawdown_not_better_than_champion"


def promote_model_if_better(
    model_path: str,
    candidate_metrics: Dict[str, Any],
    registry_path: str = "rl_ai/models/model_registry.json",
    min_reward_improvement: float = 0.0,
    max_drawdown_pct: float = 0.30,
    extra_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    registry = load_registry(registry_path)
    champion = registry.get("champion")
    champion_metrics = champion.get("metrics") if isinstance(champion, dict) else None

    promote, reason = should_promote(
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
        min_reward_improvement=min_reward_improvement,
        max_drawdown_pct=max_drawdown_pct,
    )

    decision = {
        "promoted": bool(promote),
        "reason": reason,
        "model_path": str(model_path),
        "metrics": candidate_metrics,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra_metadata:
        decision["extra_metadata"] = dict(extra_metadata)

    history = registry.get("history", [])
    if not isinstance(history, list):
        history = []
    history.append(decision)
    registry["history"] = history[-200:]

    if promote:
        registry["champion"] = decision

    save_registry(registry_path, registry)
    return decision
