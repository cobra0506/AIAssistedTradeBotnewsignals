from dataclasses import dataclass, field
from typing import Tuple

from rl_ai.shared.reward import RewardConfig


@dataclass
class RLTradingConfig:
    """
    Basic config for the first RL environment implementation.
    """

    initial_balance: float = 10_000.0
    position_size_fraction: float = 1.0
    fee_pct: float = 0.0006
    max_drawdown_stop_pct: float = 0.60
    min_bars_warmup: int = 1
    indicator_columns: Tuple[str, ...] = ()
    reward_config: RewardConfig = field(default_factory=RewardConfig)
