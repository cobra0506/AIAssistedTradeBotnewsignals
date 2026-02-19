import argparse
import json
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import RLTradingConfig
from .dataset_split import SplitConfig, split_time_series
from .env_trading import RLTradingEnv
from .evaluate_pipeline import EvaluationConfig, evaluate_policy
from .model_io import load_policy_model, save_policy_model
from .policy_model import LinearSoftmaxPolicy
from .reporting import save_training_report


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 15
    learning_rate: float = 0.01
    gamma: float = 0.99
    seed: int = 42
    eval_every: int = 1
    model_output_path: str = "rl_ai/models/phase2_policy.npz"
    save_report: bool = True
    report_output_dir: str = "rl_ai/reports/latest"

    def validate(self):
        if self.episodes < 1:
            raise ValueError("episodes must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not (0 < self.gamma <= 1):
            raise ValueError("gamma must be in (0, 1]")
        if self.eval_every < 1:
            raise ValueError("eval_every must be >= 1")
        if not str(self.model_output_path).strip():
            raise ValueError("model_output_path must not be empty")
        if self.save_report and not str(self.report_output_dir).strip():
            raise ValueError("report_output_dir must not be empty when save_report is enabled")


def _discounted_returns(rewards, gamma: float) -> np.ndarray:
    discounted = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + gamma * running
        discounted[idx] = running

    if discounted.size > 1:
        std = float(discounted.std())
        if std > 1e-8:
            discounted = (discounted - float(discounted.mean())) / (std + 1e-8)
        else:
            discounted = discounted - float(discounted.mean())
    return discounted


def _run_training_episode(
    env: RLTradingEnv,
    model: LinearSoftmaxPolicy,
    config: TrainingConfig,
    rng: np.random.Generator,
) -> Dict[str, float]:
    observation, _ = env.reset()
    done = False

    observations = []
    actions = []
    rewards = []
    total_reward = 0.0
    trades = 0
    last_info = {}

    while not done:
        obs_vector = np.asarray(observation, dtype=np.float32)
        action_result = model.sample_action(obs_vector, rng=rng)
        next_observation, reward, done, info = env.step(action_result.action)

        observations.append(obs_vector)
        actions.append(int(action_result.action))
        rewards.append(float(reward))

        total_reward += float(reward)
        if info.get("trade_executed"):
            trades += 1

        observation = next_observation
        last_info = info

    if not rewards:
        return {
            "total_reward": 0.0,
            "final_equity": float(env.equity),
            "max_drawdown_pct": float(env.max_drawdown_pct),
            "trades": float(trades),
        }

    returns = _discounted_returns(rewards, gamma=config.gamma)
    gradient_w = np.zeros_like(model.weights, dtype=np.float64)
    gradient_b = np.zeros_like(model.bias, dtype=np.float64)

    for obs_vector, action, advantage in zip(observations, actions, returns):
        probabilities = model.action_probabilities(obs_vector).astype(np.float64)
        policy_gradient = -probabilities
        policy_gradient[action] += 1.0

        gradient_w += np.outer(obs_vector, policy_gradient) * float(advantage)
        gradient_b += policy_gradient * float(advantage)

    step_scale = float(config.learning_rate) / float(max(1, len(actions)))
    model.weights += (step_scale * gradient_w).astype(np.float32)
    model.bias += (step_scale * gradient_b).astype(np.float32)

    np.clip(model.weights, -10.0, 10.0, out=model.weights)
    np.clip(model.bias, -10.0, 10.0, out=model.bias)

    return {
        "total_reward": float(total_reward),
        "final_equity": float(last_info.get("equity", env.equity)),
        "max_drawdown_pct": float(last_info.get("max_drawdown_pct", env.max_drawdown_pct)),
        "trades": float(trades),
    }


def train_policy(
    train_data: pd.DataFrame,
    validation_data: Optional[pd.DataFrame] = None,
    training_config: Optional[TrainingConfig] = None,
    env_config: Optional[RLTradingConfig] = None,
) -> Dict[str, object]:
    if train_data is None or train_data.empty:
        raise ValueError("train_data is empty")

    cfg = training_config or TrainingConfig()
    cfg.validate()
    runtime_env_config = env_config or RLTradingConfig()

    train_env = RLTradingEnv(data=train_data, config=runtime_env_config)
    model = LinearSoftmaxPolicy(
        observation_size=train_env.observation_size,
        action_size=train_env.action_space_size,
        seed=cfg.seed,
    )
    rng = np.random.default_rng(cfg.seed)

    best_score = float("-inf")
    best_weights = model.weights.copy()
    best_bias = model.bias.copy()
    history = []

    for episode in range(1, cfg.episodes + 1):
        episode_metrics = _run_training_episode(train_env, model, cfg, rng)
        validation_reward = None
        ranking_score = float(episode_metrics["total_reward"])

        if (
            validation_data is not None
            and not validation_data.empty
            and episode % cfg.eval_every == 0
        ):
            validation_metrics = evaluate_policy(
                model=model,
                data=validation_data,
                env_config=runtime_env_config,
                config=EvaluationConfig(episodes=1, deterministic=True, seed=cfg.seed),
            )
            validation_reward = float(validation_metrics["total_reward"])
            ranking_score = validation_reward

        if ranking_score > best_score:
            best_score = ranking_score
            best_weights = model.weights.copy()
            best_bias = model.bias.copy()

        history.append(
            {
                "episode": episode,
                "train_total_reward": float(episode_metrics["total_reward"]),
                "train_final_equity": float(episode_metrics["final_equity"]),
                "validation_total_reward": validation_reward,
            }
        )

    model.weights = best_weights
    model.bias = best_bias

    metadata = {
        "training_config": asdict(cfg),
        "best_score": float(best_score),
        "episodes": int(cfg.episodes),
    }
    model_path = save_policy_model(model, cfg.model_output_path, metadata=metadata)

    return {
        "model_path": str(model_path),
        "episodes": int(cfg.episodes),
        "best_score": float(best_score),
        "history": history,
    }


def train_with_split(
    data: pd.DataFrame,
    split_config: Optional[SplitConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    env_config: Optional[RLTradingConfig] = None,
) -> Dict[str, object]:
    cfg = training_config or TrainingConfig()
    cfg.validate()

    train_data, validation_data, test_data = split_time_series(data, config=split_config)

    train_summary = train_policy(
        train_data=train_data,
        validation_data=validation_data,
        training_config=cfg,
        env_config=env_config,
    )

    model, _ = load_policy_model(train_summary["model_path"])
    test_metrics = evaluate_policy(
        model=model,
        data=test_data,
        env_config=env_config,
        config=EvaluationConfig(episodes=1, deterministic=True),
    )

    result = {
        "train_rows": len(train_data),
        "validation_rows": len(validation_data),
        "test_rows": len(test_data),
        "model_path": train_summary["model_path"],
        "best_score": train_summary["best_score"],
        "train_history": train_summary.get("history", []),
        "training_config": asdict(cfg),
        "test_metrics": test_metrics,
    }

    if cfg.save_report:
        report_files = save_training_report(result, cfg.report_output_dir)
        result["report_files"] = report_files

    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RL policy on candle CSV data.")
    parser.add_argument("--data-csv", required=True, help="Path to candle CSV file")
    parser.add_argument(
        "--model-out",
        default="rl_ai/models/phase2_policy.npz",
        help="Path to save trained model (.npz)",
    )
    parser.add_argument("--episodes", type=int, default=15, help="Training episodes")
    parser.add_argument(
        "--report-dir",
        default="rl_ai/reports/latest",
        help="Directory for HTML/JSON/CSV training report outputs",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Disable report file generation",
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    data = pd.read_csv(args.data_csv)
    summary = train_with_split(
        data=data,
        training_config=TrainingConfig(
            episodes=max(1, int(args.episodes)),
            model_output_path=str(args.model_out),
            save_report=not bool(args.no_report),
            report_output_dir=str(args.report_dir),
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
