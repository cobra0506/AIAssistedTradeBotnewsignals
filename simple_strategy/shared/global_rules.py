"""Global risk and execution rules shared across backtest/paper/live engines."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


GLOBAL_RULES_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "profile": "balanced",
    "owner_lock_enabled": True,
    "one_position_per_symbol": True,
    "emergency_close_enabled": True,
    "position_stop_loss_pct": 0.02,
    "max_hold_minutes": 480,
    "max_notional_exposure_pct": 0.75,
    "max_total_open_risk_pct": 0.10,
    "min_24h_notional_usdt": 50_000_000.0,
    "max_spread_bps": 12.0,
    "min_expected_move_to_cost_ratio": 2.5,
}

GLOBAL_RULES_PROFILES: Dict[str, Dict[str, Any]] = {
    "safe": {
        "position_stop_loss_pct": 0.015,
        "max_hold_minutes": 360,
        "max_notional_exposure_pct": 0.50,
        "max_total_open_risk_pct": 0.06,
        "max_spread_bps": 8.0,
        "min_expected_move_to_cost_ratio": 3.0,
    },
    "balanced": {
        "position_stop_loss_pct": 0.02,
        "max_hold_minutes": 480,
        "max_notional_exposure_pct": 0.75,
        "max_total_open_risk_pct": 0.10,
        "max_spread_bps": 12.0,
        "min_expected_move_to_cost_ratio": 2.5,
    },
    "aggressive": {
        "position_stop_loss_pct": 0.03,
        "max_hold_minutes": 720,
        "max_notional_exposure_pct": 0.90,
        "max_total_open_risk_pct": 0.15,
        "max_spread_bps": 18.0,
        "min_expected_move_to_cost_ratio": 2.0,
    },
}


@dataclass(frozen=True)
class GlobalRules:
    enabled: bool
    profile: str
    owner_lock_enabled: bool
    one_position_per_symbol: bool
    emergency_close_enabled: bool
    position_stop_loss_pct: float
    max_hold_minutes: int
    max_notional_exposure_pct: float
    max_total_open_risk_pct: float
    min_24h_notional_usdt: float
    max_spread_bps: float
    min_expected_move_to_cost_ratio: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def round_trip_cost_pct(self, fee_pct: float, spread_pct: float, slippage_pct: float) -> float:
        fee = max(0.0, float(fee_pct))
        spread = max(0.0, float(spread_pct))
        slippage = max(0.0, float(slippage_pct))
        return 100.0 * ((2.0 * fee) + (2.0 * spread) + (2.0 * slippage))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "GlobalRules":
        payload = deepcopy(GLOBAL_RULES_DEFAULTS)
        payload.update(raw or {})
        return cls(
            enabled=bool(payload.get("enabled", False)),
            profile=str(payload.get("profile", "balanced")),
            owner_lock_enabled=bool(payload.get("owner_lock_enabled", True)),
            one_position_per_symbol=bool(payload.get("one_position_per_symbol", True)),
            emergency_close_enabled=bool(payload.get("emergency_close_enabled", True)),
            position_stop_loss_pct=max(0.0001, float(payload.get("position_stop_loss_pct", 0.02))),
            max_hold_minutes=max(1, int(payload.get("max_hold_minutes", 480))),
            max_notional_exposure_pct=max(0.0, float(payload.get("max_notional_exposure_pct", 0.75))),
            max_total_open_risk_pct=max(0.0, float(payload.get("max_total_open_risk_pct", 0.10))),
            min_24h_notional_usdt=max(0.0, float(payload.get("min_24h_notional_usdt", 50_000_000.0))),
            max_spread_bps=max(0.0, float(payload.get("max_spread_bps", 12.0))),
            min_expected_move_to_cost_ratio=max(
                0.0, float(payload.get("min_expected_move_to_cost_ratio", 2.5))
            ),
        )


def resolve_global_rules(config: Dict[str, Any] | None = None) -> GlobalRules:
    cfg = config or {}
    merged = deepcopy(GLOBAL_RULES_DEFAULTS)
    profile = str(cfg.get("global_rules_profile", merged["profile"])).strip().lower()

    file_profile_data: Dict[str, Any] = {}
    configured_path = str(cfg.get("global_rules_path", "")).strip()
    if configured_path:
        file_profile_data = _load_profile_from_file(Path(configured_path), profile=profile)
    else:
        default_path = Path(__file__).resolve().parents[1] / "configs" / "global_rules.json"
        file_profile_data = _load_profile_from_file(default_path, profile=profile)

    if file_profile_data:
        merged.update(file_profile_data)
    else:
        merged.update(GLOBAL_RULES_PROFILES.get(profile, {}))

    merged["profile"] = profile if profile in GLOBAL_RULES_PROFILES else "balanced"

    if "enable_global_rules" in cfg:
        merged["enabled"] = bool(cfg.get("enable_global_rules"))

    rule_override = cfg.get("global_rules")
    if isinstance(rule_override, dict):
        merged.update(rule_override)

    return GlobalRules.from_dict(merged)


def _load_profile_from_file(path: Path, profile: str) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        profiles = payload.get("profiles", {}) if isinstance(payload, dict) else {}
        entry = profiles.get(profile, {}) if isinstance(profiles, dict) else {}
        return entry if isinstance(entry, dict) else {}
    except Exception:
        return {}
