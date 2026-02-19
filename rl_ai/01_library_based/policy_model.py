from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ActionResult:
    action: int
    probabilities: np.ndarray


class LinearSoftmaxPolicy:
    """
    Small linear policy for discrete action selection.

    This keeps Phase 2 dependency-light (NumPy only) while still
    giving us a real train/evaluate/save/load flow.
    """

    def __init__(self, observation_size: int, action_size: int, seed: int = 42):
        if observation_size <= 0:
            raise ValueError("observation_size must be > 0")
        if action_size <= 1:
            raise ValueError("action_size must be > 1")

        self.observation_size = int(observation_size)
        self.action_size = int(action_size)
        self.rng = np.random.default_rng(seed)

        # Small random init to keep early softmax stable.
        self.weights = self.rng.normal(
            loc=0.0,
            scale=0.01,
            size=(self.observation_size, self.action_size),
        ).astype(np.float32)
        self.bias = np.zeros(self.action_size, dtype=np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp_logits = np.exp(np.clip(shifted, -50.0, 50.0))
        denominator = float(np.sum(exp_logits))
        if denominator <= 0 or not np.isfinite(denominator):
            return np.ones_like(logits, dtype=np.float32) / float(len(logits))
        return (exp_logits / denominator).astype(np.float32)

    def action_probabilities(self, observation) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if obs.size != self.observation_size:
            raise ValueError(
                f"Observation size mismatch. Expected {self.observation_size}, got {obs.size}"
            )

        logits = obs @ self.weights + self.bias
        return self._softmax(logits.astype(np.float64))

    def sample_action(self, observation, rng: Optional[np.random.Generator] = None) -> ActionResult:
        probabilities = self.action_probabilities(observation)
        generator = rng if rng is not None else self.rng
        action = int(generator.choice(self.action_size, p=probabilities))
        return ActionResult(action=action, probabilities=probabilities)

    def greedy_action(self, observation) -> int:
        probabilities = self.action_probabilities(observation)
        return int(np.argmax(probabilities))
