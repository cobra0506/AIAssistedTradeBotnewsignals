from typing import Dict


def composite_score(
    metrics: Dict[str, float],
    validation_metrics: Dict[str, float],
    stress_metrics: Dict[str, float],
    active_signal_count: int,
) -> float:
    """Compute a robust score favoring return quality and stability."""
    train_return = float(metrics.get("total_return", 0.0))
    train_sharpe = float(metrics.get("sharpe_ratio", 0.0))
    train_win_rate = float(metrics.get("win_rate", 0.0))
    train_drawdown = float(metrics.get("max_drawdown", 0.0))
    train_trades = int(metrics.get("total_trades", 0))

    val_return = float(validation_metrics.get("total_return", 0.0))
    val_sharpe = float(validation_metrics.get("sharpe_ratio", 0.0))
    val_trades = int(validation_metrics.get("total_trades", 0))

    stress_return = float(stress_metrics.get("total_return", 0.0))

    stability_penalty = abs(train_return - val_return)
    stress_penalty = max(0.0, train_return - stress_return)
    complexity_penalty = max(0, active_signal_count - 1) * 2.0
    low_trade_penalty = max(0, 2 - train_trades) * 6.0
    validation_trade_penalty = max(0, 1 - val_trades) * 24.0
    negative_validation_penalty = max(0.0, -val_return) * 3.0

    score = (
        (0.40 * train_return)
        + (0.20 * train_sharpe)
        + (0.10 * train_win_rate)
        + (0.15 * val_return)
        + (0.10 * val_sharpe)
        - (0.25 * train_drawdown)
        - (0.10 * stability_penalty)
        - (0.10 * stress_penalty)
        - low_trade_penalty
        - validation_trade_penalty
        - negative_validation_penalty
        - complexity_penalty
    )
    return float(score)


def metrics_or_zero(metrics: Dict[str, float]) -> Dict[str, float]:
    defaults = {
        "win_rate": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "total_return": 0.0,
        "total_trades": 0,
    }
    payload = dict(defaults)
    payload.update(metrics or {})
    return payload
