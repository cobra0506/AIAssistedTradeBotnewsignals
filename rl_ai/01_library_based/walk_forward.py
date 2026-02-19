from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import RLTradingConfig
from .dataset_split import SplitConfig
from .train_pipeline import TrainingConfig, train_with_split


@dataclass(frozen=True)
class WalkForwardConfig:
    window_rows: int = 3000
    step_rows: int = 1000
    max_windows: int = 6
    min_window_rows: int = 500
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    def validate(self):
        if self.window_rows < self.min_window_rows:
            raise ValueError("window_rows must be >= min_window_rows")
        if self.step_rows < 1:
            raise ValueError("step_rows must be >= 1")
        if self.max_windows < 1:
            raise ValueError("max_windows must be >= 1")
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train/validation/test ratios must sum to 1.0, got {total}")


def run_walk_forward_validation(
    data: pd.DataFrame,
    training_config: TrainingConfig,
    env_config: Optional[RLTradingConfig] = None,
    config: Optional[WalkForwardConfig] = None,
) -> Dict[str, Any]:
    if data is None or data.empty:
        raise ValueError("data is empty")

    wf_cfg = config or WalkForwardConfig()
    wf_cfg.validate()

    rows = len(data)
    windows: List[Dict[str, Any]] = []
    cursor = 0
    window_id = 0
    base_model_path = Path(training_config.model_output_path)

    while cursor + wf_cfg.min_window_rows <= rows and window_id < wf_cfg.max_windows:
        end = min(cursor + wf_cfg.window_rows, rows)
        window_df = data.iloc[cursor:end].copy().reset_index(drop=True)
        if len(window_df) < wf_cfg.min_window_rows:
            break

        split_cfg = SplitConfig(
            train_ratio=wf_cfg.train_ratio,
            validation_ratio=wf_cfg.validation_ratio,
            test_ratio=wf_cfg.test_ratio,
            min_rows_per_split=max(10, int(min(200, len(window_df) * 0.05))),
        )

        window_model = base_model_path.with_name(
            f"{base_model_path.stem}_wf_{window_id}{base_model_path.suffix}"
        )
        local_train_cfg = replace(
            training_config,
            model_output_path=str(window_model),
            save_report=False,
        )

        summary = train_with_split(
            data=window_df,
            split_config=split_cfg,
            training_config=local_train_cfg,
            env_config=env_config,
        )

        test_metrics = summary.get("test_metrics", {})
        windows.append(
            {
                "window_id": window_id,
                "start_row": int(cursor),
                "end_row": int(end),
                "rows": int(len(window_df)),
                "model_path": summary.get("model_path"),
                "best_score": float(summary.get("best_score", 0.0)),
                "test_metrics": test_metrics,
            }
        )

        window_id += 1
        cursor += wf_cfg.step_rows

    total_reward_values = [
        float(item.get("test_metrics", {}).get("total_reward", 0.0)) for item in windows
    ]
    drawdown_values = [
        float(item.get("test_metrics", {}).get("max_drawdown_pct", 0.0)) for item in windows
    ]

    return {
        "windows_run": len(windows),
        "average_total_reward": float(np.mean(total_reward_values)) if total_reward_values else 0.0,
        "average_max_drawdown_pct": float(np.mean(drawdown_values)) if drawdown_values else 0.0,
        "windows": windows,
    }


def walk_forward_passes(
    summary: Dict[str, Any],
    min_average_reward: float = 0.0,
    max_average_drawdown_pct: float = 0.30,
) -> bool:
    if not isinstance(summary, dict):
        return False
    if int(summary.get("windows_run", 0)) < 1:
        return False

    avg_reward = float(summary.get("average_total_reward", 0.0))
    avg_drawdown = float(summary.get("average_max_drawdown_pct", 1.0))
    return avg_reward >= min_average_reward and avg_drawdown <= max_average_drawdown_pct
