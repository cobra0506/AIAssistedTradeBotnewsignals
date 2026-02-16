from dataclasses import dataclass
from typing import Dict, Optional


ALLOWED_SIGNALS = {"OPEN_LONG", "CLOSE_LONG", "OPEN_SHORT", "CLOSE_SHORT", "HOLD"}


@dataclass
class GateResult:
    passed: bool
    reason: str = ""


def gate_train_metrics(
    metrics: Dict[str, float],
    min_trades: int,
    min_return: float = -100.0,
    max_drawdown: Optional[float] = None,
    min_sharpe: Optional[float] = None,
    min_win_rate: Optional[float] = None,
) -> GateResult:
    trades = int(metrics.get("total_trades", 0))
    total_return = float(metrics.get("total_return", 0.0))
    drawdown = float(metrics.get("max_drawdown", 0.0))
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    win_rate = float(metrics.get("win_rate", 0.0))

    if trades < min_trades:
        return GateResult(False, f"min_trades_failed:{trades}<{min_trades}")

    if total_return < min_return:
        return GateResult(False, f"min_return_failed:{total_return:.2f}<{min_return:.2f}")

    if max_drawdown is not None and drawdown > max_drawdown:
        return GateResult(False, f"max_drawdown_failed:{drawdown:.2f}>{max_drawdown:.2f}")

    if min_sharpe is not None and sharpe < min_sharpe:
        return GateResult(False, f"min_sharpe_failed:{sharpe:.3f}<{min_sharpe:.3f}")

    if min_win_rate is not None and win_rate < min_win_rate:
        return GateResult(False, f"min_win_rate_failed:{win_rate:.2f}<{min_win_rate:.2f}")

    return GateResult(True, "")


def gate_validation_consistency(
    train_metrics: Dict[str, float],
    validation_metrics: Dict[str, float],
    max_return_gap: float = 40.0,
) -> GateResult:
    train_return = float(train_metrics.get("total_return", 0.0))
    validation_return = float(validation_metrics.get("total_return", 0.0))

    gap = abs(train_return - validation_return)
    if gap > max_return_gap:
        return GateResult(False, f"stability_gap_failed:{gap:.2f}>{max_return_gap:.2f}")

    return GateResult(True, "")


def gate_stress(
    metrics: Dict[str, float],
    stress_metrics: Dict[str, float],
    max_drop: float = 35.0,
    min_return: float = -100.0,
    max_drawdown: Optional[float] = None,
) -> GateResult:
    normal_return = float(metrics.get("total_return", 0.0))
    stress_return = float(stress_metrics.get("total_return", 0.0))
    stress_drawdown = float(stress_metrics.get("max_drawdown", 0.0))

    drop = normal_return - stress_return
    if drop > max_drop:
        return GateResult(False, f"stress_drop_failed:{drop:.2f}>{max_drop:.2f}")

    if stress_return < min_return:
        return GateResult(False, f"stress_return_failed:{stress_return:.2f}<{min_return:.2f}")

    if max_drawdown is not None and stress_drawdown > max_drawdown:
        return GateResult(False, f"stress_drawdown_failed:{stress_drawdown:.2f}>{max_drawdown:.2f}")

    return GateResult(True, "")
