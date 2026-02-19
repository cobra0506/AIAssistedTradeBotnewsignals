from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class DeepActionResult:
    action: int
    probabilities: np.ndarray


class StableBaselinesPolicyAdapter:
    """
    Adapter that gives Stable-Baselines models the same interface
    used by the existing strategy/evaluation code.
    """

    def __init__(self, sb3_model: Any, seed: int = 42):
        self._model = sb3_model
        self._rng = np.random.default_rng(seed)

        obs_space = getattr(sb3_model, "observation_space", None)
        act_space = getattr(sb3_model, "action_space", None)

        if obs_space is None or not hasattr(obs_space, "shape") or not obs_space.shape:
            raise ValueError("Stable-Baselines model has no valid observation_space.shape")
        if act_space is None or not hasattr(act_space, "n"):
            raise ValueError("Stable-Baselines model has no discrete action space")

        self.observation_size = int(obs_space.shape[0])
        self.action_size = int(act_space.n)

    def _normalize_observation(self, observation) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if int(obs.size) != int(self.observation_size):
            raise ValueError(
                f"Observation size mismatch. Expected {self.observation_size}, got {obs.size}"
            )
        return obs.reshape(1, -1)

    @staticmethod
    def _normalize_action(action_value) -> int:
        if isinstance(action_value, np.ndarray):
            if action_value.size == 0:
                return 0
            return int(action_value.reshape(-1)[0])
        if isinstance(action_value, (list, tuple)):
            if len(action_value) == 0:
                return 0
            return int(action_value[0])
        return int(action_value)

    def action_probabilities(self, observation) -> np.ndarray:
        # Stable-Baselines does not expose normalized action probabilities
        # for all algorithms. We return a uniform vector for compatibility.
        _ = self._normalize_observation(observation)
        if self.action_size <= 0:
            return np.array([1.0], dtype=np.float32)
        return np.ones(self.action_size, dtype=np.float32) / float(self.action_size)

    def sample_action(
        self, observation, rng: Optional[np.random.Generator] = None
    ) -> DeepActionResult:
        del rng
        obs = self._normalize_observation(observation)
        action, _state = self._model.predict(obs, deterministic=False)
        normalized_action = self._normalize_action(action)
        return DeepActionResult(
            action=normalized_action,
            probabilities=self.action_probabilities(observation),
        )

    def greedy_action(self, observation) -> int:
        obs = self._normalize_observation(observation)
        action, _state = self._model.predict(obs, deterministic=True)
        return self._normalize_action(action)
