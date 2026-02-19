import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .config import RLTradingConfig
from .env_trading import RLTradingEnv
from .model_io import load_policy_model
from .policy_model import LinearSoftmaxPolicy
from .reporting import save_evaluation_report


@dataclass(frozen=True)
class EvaluationConfig:
    episodes: int = 1
    deterministic: bool = True
    seed: int = 42


def evaluate_policy(
    model: LinearSoftmaxPolicy,
    data: pd.DataFrame,
    env_config: Optional[RLTradingConfig] = None,
    config: Optional[EvaluationConfig] = None,
) -> Dict[str, Any]:
    cfg = config or EvaluationConfig()
    if cfg.episodes < 1:
        raise ValueError("episodes must be >= 1")

    runtime_env_config = env_config or RLTradingConfig()
    rng = np.random.default_rng(cfg.seed)

    total_reward = 0.0
    total_trades = 0
    total_steps = 0
    final_equities = []
    max_drawdowns = []
    action_counts = {}
    signal_counts = {
        "OPEN_LONG": 0,
        "CLOSE_LONG": 0,
        "OPEN_SHORT": 0,
        "CLOSE_SHORT": 0,
        "HOLD": 0,
    }

    for _ in range(cfg.episodes):
        env = RLTradingEnv(data=data, config=runtime_env_config)
        observation, _ = env.reset()
        done = False
        episode_reward = 0.0
        last_info = {}

        while not done:
            if cfg.deterministic:
                action = model.greedy_action(observation)
            else:
                action = model.sample_action(observation, rng=rng).action
            action_key = str(int(action))
            action_counts[action_key] = int(action_counts.get(action_key, 0)) + 1

            observation, reward, done, info = env.step(action)
            episode_reward += float(reward)
            total_steps += 1
            if info.get("trade_executed"):
                total_trades += 1
            signal_name = str(info.get("signal", "HOLD"))
            if signal_name not in signal_counts:
                signal_name = "HOLD"
            signal_counts[signal_name] = int(signal_counts.get(signal_name, 0)) + 1
            last_info = info

        total_reward += episode_reward
        final_equities.append(float(last_info.get("equity", env.equity)))
        max_drawdowns.append(float(last_info.get("max_drawdown_pct", env.max_drawdown_pct)))

    episode_count = float(cfg.episodes)
    return {
        "episodes": int(cfg.episodes),
        "total_reward": float(total_reward / episode_count),
        "final_equity": float(np.mean(final_equities)) if final_equities else 0.0,
        "max_drawdown_pct": float(np.mean(max_drawdowns)) if max_drawdowns else 0.0,
        "trades": int(total_trades),
        "steps": int(total_steps),
        "action_counts": action_counts,
        "signal_counts": signal_counts,
    }


def evaluate_saved_model(
    model_path: str,
    data: pd.DataFrame,
    env_config: Optional[RLTradingConfig] = None,
    config: Optional[EvaluationConfig] = None,
    report_dir: Optional[str] = None,
) -> Dict[str, Any]:
    model, metadata = load_policy_model(model_path)
    metrics = evaluate_policy(model=model, data=data, env_config=env_config, config=config)
    metrics["model_path"] = model_path
    metrics["metadata"] = metadata
    if report_dir:
        metrics["report_files"] = save_evaluation_report(metrics, report_dir)
    return metrics


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved RL policy on CSV candles.")
    parser.add_argument("--data-csv", required=True, help="Path to candle CSV file")
    parser.add_argument("--model-path", required=True, help="Path to .npz model file")
    parser.add_argument("--episodes", type=int, default=1, help="Evaluation episodes")
    parser.add_argument(
        "--report-dir",
        default="",
        help="Optional output directory for evaluation HTML/JSON report",
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    data = pd.read_csv(args.data_csv)
    metrics = evaluate_saved_model(
        model_path=args.model_path,
        data=data,
        config=EvaluationConfig(episodes=max(1, int(args.episodes))),
        report_dir=str(args.report_dir).strip() or None,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
