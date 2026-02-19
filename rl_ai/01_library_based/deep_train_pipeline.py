import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .config import RLTradingConfig
from .dataset_split import SplitConfig, split_time_series
from .env_trading import RLTradingEnv
from .evaluate_pipeline import EvaluationConfig, evaluate_saved_model
from .reporting import save_training_report


def _require_deep_dependencies():
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import A2C, DQN, PPO
    except Exception as exc:
        raise ImportError(
            "Deep RL training requires optional packages. "
            "Install with: pip install torch stable-baselines3 gymnasium"
        ) from exc

    return gym, spaces, {"ppo": PPO, "a2c": A2C, "dqn": DQN}


def _build_gym_env(data: pd.DataFrame, env_config: RLTradingConfig, gym, spaces):
    class _GymTradingEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self._base_env = RLTradingEnv(data=data, config=env_config)
            self.action_space = spaces.Discrete(self._base_env.action_space_size)
            self.observation_space = spaces.Box(
                low=-1e9,
                high=1e9,
                shape=(self._base_env.observation_size,),
                dtype=np.float32,
            )

        def reset(self, *, seed=None, options=None):
            del options
            if seed is not None:
                try:
                    super().reset(seed=seed)
                except Exception:
                    pass
            observation, info = self._base_env.reset()
            return np.asarray(observation, dtype=np.float32), dict(info)

        def step(self, action):
            observation, reward, done, info = self._base_env.step(int(action))
            terminated = bool(done)
            truncated = False
            return (
                np.asarray(observation, dtype=np.float32),
                float(reward),
                terminated,
                truncated,
                dict(info),
            )

    return _GymTradingEnv()


@dataclass(frozen=True)
class DeepTrainingConfig:
    algorithm: str = "ppo"
    total_timesteps: int = 200_000
    seed: int = 42
    learning_rate: float = 3e-4
    model_output_path: str = "rl_ai/models/deep_policy.zip"
    save_report: bool = True
    report_output_dir: str = "rl_ai/reports/deep_latest"

    def validate(self):
        algo = str(self.algorithm).strip().lower()
        if algo not in {"ppo", "a2c", "dqn"}:
            raise ValueError("algorithm must be one of: ppo, a2c, dqn")
        if int(self.total_timesteps) < 1_000:
            raise ValueError("total_timesteps must be >= 1000")
        if float(self.learning_rate) <= 0:
            raise ValueError("learning_rate must be > 0")
        if not str(self.model_output_path).strip():
            raise ValueError("model_output_path must not be empty")
        if self.save_report and not str(self.report_output_dir).strip():
            raise ValueError("report_output_dir must not be empty when save_report is enabled")

    @property
    def normalized_algorithm(self) -> str:
        return str(self.algorithm).strip().lower()


def _metadata_sidecar_path(model_path: Path) -> Path:
    return Path(f"{model_path}.metadata.json")


def train_deep_with_split(
    data: pd.DataFrame,
    split_config: Optional[SplitConfig] = None,
    training_config: Optional[DeepTrainingConfig] = None,
    env_config: Optional[RLTradingConfig] = None,
) -> Dict[str, Any]:
    if data is None or data.empty:
        raise ValueError("data is empty")

    cfg = training_config or DeepTrainingConfig()
    cfg.validate()

    gym, spaces, algorithm_map = _require_deep_dependencies()

    train_data, validation_data, test_data = split_time_series(data, config=split_config)
    runtime_env_config = env_config or RLTradingConfig()

    train_env = _build_gym_env(train_data, runtime_env_config, gym, spaces)
    algorithm_name = cfg.normalized_algorithm
    algorithm_class = algorithm_map[algorithm_name]

    model_kwargs: Dict[str, Any] = {
        "policy": "MlpPolicy",
        "env": train_env,
        "verbose": 0,
        "seed": int(cfg.seed),
        "learning_rate": float(cfg.learning_rate),
    }

    # Use wider networks by default for deeper representation.
    if algorithm_name in {"ppo", "a2c"}:
        model_kwargs["policy_kwargs"] = {"net_arch": [256, 256]}
    if algorithm_name == "dqn":
        model_kwargs["buffer_size"] = max(10_000, int(len(train_data) * 20))
        model_kwargs["learning_starts"] = min(5_000, max(200, int(len(train_data) * 2)))
        model_kwargs["batch_size"] = 64
        model_kwargs["policy_kwargs"] = {"net_arch": [256, 256]}

    model = algorithm_class(**model_kwargs)
    model.learn(total_timesteps=int(cfg.total_timesteps))

    model_path = Path(str(cfg.model_output_path))
    if model_path.suffix.lower() != ".zip":
        model_path = model_path.with_suffix(".zip")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    metadata = {
        "model_backend": "stable_baselines3",
        "model_format": "zip",
        "algorithm": algorithm_name,
        "training_config": asdict(cfg),
    }
    _metadata_sidecar_path(model_path).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    test_metrics = evaluate_saved_model(
        model_path=str(model_path),
        data=test_data,
        env_config=runtime_env_config,
        config=EvaluationConfig(episodes=1, deterministic=True),
        report_dir=str(cfg.report_output_dir) if cfg.save_report else None,
    )

    result: Dict[str, Any] = {
        "model_backend": "stable_baselines3",
        "algorithm": algorithm_name,
        "model_path": str(model_path),
        "train_rows": int(len(train_data)),
        "validation_rows": int(len(validation_data)),
        "test_rows": int(len(test_data)),
        "total_timesteps": int(cfg.total_timesteps),
        "training_config": asdict(cfg),
        "test_metrics": test_metrics,
    }

    if cfg.save_report:
        result["report_files"] = save_training_report(result, cfg.report_output_dir)

    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train deep RL model (Stable-Baselines3).")
    parser.add_argument("--data-csv", required=True, help="Path to candle CSV file")
    parser.add_argument("--algorithm", default="ppo", choices=["ppo", "a2c", "dqn"])
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--model-out",
        default="rl_ai/models/deep_policy.zip",
        help="Path to save deep model (.zip)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--report-dir",
        default="rl_ai/reports/deep_latest",
        help="Directory for report outputs",
    )
    parser.add_argument("--no-report", action="store_true")
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    data = pd.read_csv(args.data_csv)
    summary = train_deep_with_split(
        data=data,
        training_config=DeepTrainingConfig(
            algorithm=str(args.algorithm),
            total_timesteps=max(1_000, int(args.timesteps)),
            seed=int(args.seed),
            learning_rate=float(args.learning_rate),
            model_output_path=str(args.model_out),
            save_report=not bool(args.no_report),
            report_output_dir=str(args.report_dir),
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
