from dataclasses import dataclass
from typing import Dict


@dataclass
class RewardConfig:
    profit_weight: float = 1.0
    drawdown_weight: float = 0.75
    fee_weight: float = 1.0
    trade_penalty: float = 0.0001
    clip_low: float = -1.0
    clip_high: float = 1.0


@dataclass
class RewardBreakdown:
    total: float
    profit_component: float
    drawdown_penalty: float
    fee_penalty: float
    trade_penalty: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "total": float(self.total),
            "profit_component": float(self.profit_component),
            "drawdown_penalty": float(self.drawdown_penalty),
            "fee_penalty": float(self.fee_penalty),
            "trade_penalty": float(self.trade_penalty),
        }


def calculate_reward(
    previous_equity: float,
    current_equity: float,
    previous_drawdown_pct: float,
    current_drawdown_pct: float,
    fee_paid: float = 0.0,
    trade_executed: bool = False,
    config: RewardConfig = None,
) -> RewardBreakdown:
    """
    Reward = profit gain - drawdown increase - fees - optional trade churn penalty.
    """
    cfg = config or RewardConfig()

    safe_previous_equity = max(float(previous_equity), 1e-9)
    current_equity = float(current_equity)
    fee_paid = max(float(fee_paid), 0.0)

    step_return = (current_equity / safe_previous_equity) - 1.0
    profit_component = step_return * cfg.profit_weight

    drawdown_increase = max(0.0, float(current_drawdown_pct) - float(previous_drawdown_pct))
    drawdown_penalty = drawdown_increase * cfg.drawdown_weight

    fee_penalty = (fee_paid / safe_previous_equity) * cfg.fee_weight
    churn_penalty = cfg.trade_penalty if trade_executed else 0.0

    raw_total = profit_component - drawdown_penalty - fee_penalty - churn_penalty
    total = max(cfg.clip_low, min(cfg.clip_high, raw_total))

    return RewardBreakdown(
        total=total,
        profit_component=profit_component,
        drawdown_penalty=drawdown_penalty,
        fee_penalty=fee_penalty,
        trade_penalty=churn_penalty,
    )
