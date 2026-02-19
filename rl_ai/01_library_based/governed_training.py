import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from .config import RLTradingConfig
from .dataset_split import SplitConfig
from .model_registry import promote_model_if_better
from .train_pipeline import TrainingConfig, train_with_split
from .walk_forward import WalkForwardConfig, run_walk_forward_validation, walk_forward_passes


@dataclass(frozen=True)
class GovernanceConfig:
    enable_walk_forward: bool = True
    min_walk_forward_avg_reward: float = 0.0
    max_walk_forward_avg_drawdown_pct: float = 0.30
    enable_promotion: bool = True
    registry_path: str = "rl_ai/models/model_registry.json"
    min_reward_improvement: float = 0.0
    max_drawdown_pct: float = 0.30


def train_with_governance(
    data: pd.DataFrame,
    training_config: Optional[TrainingConfig] = None,
    env_config: Optional[RLTradingConfig] = None,
    split_config: Optional[SplitConfig] = None,
    walk_forward_config: Optional[WalkForwardConfig] = None,
    governance_config: Optional[GovernanceConfig] = None,
) -> Dict[str, Any]:
    train_cfg = training_config or TrainingConfig()
    gov_cfg = governance_config or GovernanceConfig()

    summary = train_with_split(
        data=data,
        split_config=split_config,
        training_config=train_cfg,
        env_config=env_config,
    )

    result: Dict[str, Any] = dict(summary)
    walk_forward_summary = None
    walk_forward_ok = True
    if gov_cfg.enable_walk_forward:
        walk_forward_summary = run_walk_forward_validation(
            data=data,
            training_config=train_cfg,
            env_config=env_config,
            config=walk_forward_config,
        )
        walk_forward_ok = walk_forward_passes(
            walk_forward_summary,
            min_average_reward=gov_cfg.min_walk_forward_avg_reward,
            max_average_drawdown_pct=gov_cfg.max_walk_forward_avg_drawdown_pct,
        )
        result["walk_forward"] = {
            "passed": bool(walk_forward_ok),
            "summary": walk_forward_summary,
        }

    promotion_decision = None
    if gov_cfg.enable_promotion:
        if walk_forward_summary:
            candidate_metrics = {
                "total_reward": float(walk_forward_summary.get("average_total_reward", 0.0)),
                "max_drawdown_pct": float(
                    walk_forward_summary.get("average_max_drawdown_pct", 1.0)
                ),
            }
        else:
            candidate_metrics = {
                "total_reward": float(result.get("test_metrics", {}).get("total_reward", 0.0)),
                "max_drawdown_pct": float(
                    result.get("test_metrics", {}).get("max_drawdown_pct", 1.0)
                ),
            }

        if walk_forward_ok:
            promotion_decision = promote_model_if_better(
                model_path=str(result.get("model_path")),
                candidate_metrics=candidate_metrics,
                registry_path=gov_cfg.registry_path,
                min_reward_improvement=gov_cfg.min_reward_improvement,
                max_drawdown_pct=gov_cfg.max_drawdown_pct,
                extra_metadata={"source": "train_with_governance"},
            )
        else:
            promotion_decision = {
                "promoted": False,
                "reason": "walk_forward_failed",
                "model_path": str(result.get("model_path")),
                "metrics": candidate_metrics,
            }

        result["promotion"] = promotion_decision

    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Governed RL training with walk-forward + promotion.")
    parser.add_argument("--data-csv", required=True, help="Path to candle CSV file")
    parser.add_argument("--model-out", default="rl_ai/models/governed_policy.npz")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--report-dir", default="rl_ai/reports/governed")
    parser.add_argument("--registry-path", default="rl_ai/models/model_registry.json")
    parser.add_argument("--skip-walk-forward", action="store_true")
    parser.add_argument("--skip-promotion", action="store_true")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    data = pd.read_csv(args.data_csv)
    result = train_with_governance(
        data=data,
        training_config=TrainingConfig(
            episodes=max(1, int(args.episodes)),
            model_output_path=str(args.model_out),
            save_report=True,
            report_output_dir=str(args.report_dir),
        ),
        governance_config=GovernanceConfig(
            enable_walk_forward=not bool(args.skip_walk_forward),
            enable_promotion=not bool(args.skip_promotion),
            registry_path=str(args.registry_path),
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
