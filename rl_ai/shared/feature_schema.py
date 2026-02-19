from dataclasses import dataclass, field
from typing import List


def _default_base_features() -> List[str]:
    return [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_1",
        "hl_spread",
        "oc_change",
        "volume_change_1",
    ]


def _default_account_features() -> List[str]:
    return [
        "has_position",
        "is_short",
        "unrealized_pnl_pct",
        "drawdown_pct",
        "balance_ratio",
    ]


@dataclass(frozen=True)
class FeatureSchema:
    """
    Fixed feature schema used by RL models.

    Keeping feature order stable is critical for model training and inference.
    """

    indicator_features: List[str] = field(default_factory=list)
    base_features: List[str] = field(default_factory=_default_base_features)
    account_features: List[str] = field(default_factory=_default_account_features)

    @property
    def feature_names(self) -> List[str]:
        return self.base_features + self.indicator_features + self.account_features

    @property
    def size(self) -> int:
        return len(self.feature_names)
