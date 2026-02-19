from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .config import RLTradingConfig
from .env_trading import RLTradingEnv
from .evaluate_pipeline import EvaluationConfig, evaluate_policy
from .model_io import load_policy_model, save_policy_model
from .multi_context_data import ContextDataset, prepare_multi_context_datasets
from .policy_model import LinearSoftmaxPolicy
from .reporting import save_training_report
from .train_pipeline import TrainingConfig, _run_training_episode


@dataclass(frozen=True)
class MultiContextConfig:
    data_dir: str = "data"
    symbols: tuple[str, ...] = ()
    timeframes: tuple[str, ...] = ()
    indicator_columns: tuple[str, ...] = ()
    min_rows_per_context: int = 220
    validation_ratio: float = 0.20
    test_ratio: float = 0.20
    add_context_features: bool = True

    def validate(self):
        if not str(self.data_dir).strip():
            raise ValueError("data_dir must not be empty")
        if self.min_rows_per_context < 90:
            raise ValueError("min_rows_per_context must be >= 90")
        if not (0 < self.validation_ratio < 0.49):
            raise ValueError("validation_ratio must be in (0, 0.49)")
        if not (0 < self.test_ratio < 0.49):
            raise ValueError("test_ratio must be in (0, 0.49)")
        if self.validation_ratio + self.test_ratio >= 0.8:
            raise ValueError("validation_ratio + test_ratio too large")


def _unique(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _split_context_dataset(
    context: ContextDataset,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[ContextDataset, ContextDataset, ContextDataset]:
    total = len(context.data)
    test_rows = max(20, int(total * test_ratio))
    val_rows = max(20, int(total * validation_ratio))
    train_rows = total - val_rows - test_rows
    if train_rows < 50:
        raise ValueError(
            f"Context {context.symbol}-{context.timeframe} has too few rows after split: {total}"
        )

    train_df = context.data.iloc[:train_rows].copy().reset_index(drop=True)
    val_df = context.data.iloc[train_rows:train_rows + val_rows].copy().reset_index(drop=True)
    test_df = context.data.iloc[train_rows + val_rows:].copy().reset_index(drop=True)

    return (
        ContextDataset(context.symbol, context.timeframe, train_df, context.source_path),
        ContextDataset(context.symbol, context.timeframe, val_df, context.source_path),
        ContextDataset(context.symbol, context.timeframe, test_df, context.source_path),
    )


def train_multi_context(
    config: Optional[MultiContextConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    env_config: Optional[RLTradingConfig] = None,
) -> Dict[str, Any]:
    cfg = config or MultiContextConfig()
    cfg.validate()
    train_cfg = training_config or TrainingConfig()
    train_cfg.validate()
    base_env_cfg = env_config or RLTradingConfig()

    contexts, meta = prepare_multi_context_datasets(
        data_dir=cfg.data_dir,
        symbols=list(cfg.symbols) or None,
        timeframes=list(cfg.timeframes) or None,
        indicator_columns=cfg.indicator_columns,
        min_rows=cfg.min_rows_per_context,
        add_context_features=cfg.add_context_features,
    )
    if not contexts:
        raise ValueError("No contexts loaded from data_dir for multi-context training")

    split_train: List[ContextDataset] = []
    split_val: List[ContextDataset] = []
    split_test: List[ContextDataset] = []
    for context in contexts:
        train_ctx, val_ctx, test_ctx = _split_context_dataset(
            context,
            validation_ratio=cfg.validation_ratio,
            test_ratio=cfg.test_ratio,
        )
        split_train.append(train_ctx)
        split_val.append(val_ctx)
        split_test.append(test_ctx)

    shared_indicators = _unique(
        list(base_env_cfg.indicator_columns)
        + list(cfg.indicator_columns)
        + (["ctx_symbol_id", "ctx_timeframe_id"] if cfg.add_context_features else [])
    )
    runtime_env_cfg = replace(base_env_cfg, indicator_columns=tuple(shared_indicators))

    first_env = RLTradingEnv(data=split_train[0].data, config=runtime_env_cfg)
    model = LinearSoftmaxPolicy(
        observation_size=first_env.observation_size,
        action_size=first_env.action_space_size,
        seed=train_cfg.seed,
    )
    rng = np.random.default_rng(train_cfg.seed)

    best_score = float("-inf")
    best_weights = model.weights.copy()
    best_bias = model.bias.copy()
    history = []

    for episode in range(1, train_cfg.episodes + 1):
        context = split_train[(episode - 1) % len(split_train)]
        env = RLTradingEnv(data=context.data, config=runtime_env_cfg)
        metrics = _run_training_episode(env=env, model=model, config=train_cfg, rng=rng)

        validation_reward = None
        if episode % train_cfg.eval_every == 0:
            val_rewards = []
            for val_ctx in split_val:
                eval_metrics = evaluate_policy(
                    model=model,
                    data=val_ctx.data,
                    env_config=runtime_env_cfg,
                    config=EvaluationConfig(episodes=1, deterministic=True, seed=train_cfg.seed),
                )
                val_rewards.append(float(eval_metrics["total_reward"]))
            validation_reward = float(np.mean(val_rewards)) if val_rewards else None

        ranking_score = (
            float(validation_reward)
            if validation_reward is not None
            else float(metrics.get("total_reward", 0.0))
        )
        if ranking_score > best_score:
            best_score = ranking_score
            best_weights = model.weights.copy()
            best_bias = model.bias.copy()

        history.append(
            {
                "episode": episode,
                "context_symbol": context.symbol,
                "context_timeframe": context.timeframe,
                "train_total_reward": float(metrics.get("total_reward", 0.0)),
                "train_final_equity": float(metrics.get("final_equity", 0.0)),
                "validation_total_reward": validation_reward,
            }
        )

    model.weights = best_weights
    model.bias = best_bias

    model_path = Path(train_cfg.model_output_path)
    metadata = {
        "training_mode": "multi_context",
        "contexts_loaded": len(contexts),
        "symbols": sorted({ctx.symbol for ctx in contexts}),
        "timeframes": sorted({ctx.timeframe for ctx in contexts}),
        "training_config": asdict(train_cfg),
        "multi_context_config": asdict(cfg),
        "best_score": float(best_score),
    }
    save_policy_model(model, str(model_path), metadata=metadata)

    test_context_metrics: List[Dict[str, Any]] = []
    test_rewards = []
    test_drawdowns = []
    reloaded_model, _ = load_policy_model(str(model_path))
    for test_ctx in split_test:
        eval_metrics = evaluate_policy(
            model=reloaded_model,
            data=test_ctx.data,
            env_config=runtime_env_cfg,
            config=EvaluationConfig(episodes=1, deterministic=True, seed=train_cfg.seed),
        )
        test_context_metrics.append(
            {
                "symbol": test_ctx.symbol,
                "timeframe": test_ctx.timeframe,
                "rows": len(test_ctx.data),
                "metrics": eval_metrics,
            }
        )
        test_rewards.append(float(eval_metrics["total_reward"]))
        test_drawdowns.append(float(eval_metrics["max_drawdown_pct"]))

    summary: Dict[str, Any] = {
        "mode": "multi_context",
        "model_path": str(model_path),
        "best_score": float(best_score),
        "contexts_loaded": len(contexts),
        "contexts_metadata": meta,
        "indicator_columns": shared_indicators,
        "train_history": history,
        "test_metrics": {
            "total_reward": float(np.mean(test_rewards)) if test_rewards else 0.0,
            "max_drawdown_pct": float(np.mean(test_drawdowns)) if test_drawdowns else 0.0,
            "contexts": test_context_metrics,
        },
    }

    if train_cfg.save_report:
        summary["report_files"] = save_training_report(summary, train_cfg.report_output_dir)

    return summary
