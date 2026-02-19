import json
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .policy_model import LinearSoftmaxPolicy


def save_policy_model(
    model: LinearSoftmaxPolicy,
    model_path: str,
    metadata: Dict[str, Any] | None = None,
) -> Path:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata_json = json.dumps(metadata or {}, sort_keys=True)
    np.savez(
        path,
        weights=model.weights.astype(np.float32),
        bias=model.bias.astype(np.float32),
        observation_size=np.int32(model.observation_size),
        action_size=np.int32(model.action_size),
        metadata_json=np.array(metadata_json),
    )
    return path


def _load_metadata_sidecar(model_path: Path) -> Dict[str, Any]:
    sidecar_path = Path(f"{model_path}.metadata.json")
    if not sidecar_path.exists():
        return {}
    try:
        return json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_linear_policy_model(path: Path) -> Tuple[LinearSoftmaxPolicy, Dict[str, Any]]:
    """Load legacy NumPy linear policy format (.npz)."""

    with np.load(path, allow_pickle=False) as payload:
        observation_size = int(payload["observation_size"])
        action_size = int(payload["action_size"])
        weights = np.asarray(payload["weights"], dtype=np.float32)
        bias = np.asarray(payload["bias"], dtype=np.float32)
        metadata_json = str(payload["metadata_json"])

    if weights.shape != (observation_size, action_size):
        raise ValueError(
            f"Invalid weights shape {weights.shape}, expected {(observation_size, action_size)}"
        )
    if bias.shape != (action_size,):
        raise ValueError(f"Invalid bias shape {bias.shape}, expected {(action_size,)}")

    model = LinearSoftmaxPolicy(
        observation_size=observation_size,
        action_size=action_size,
        seed=42,
    )
    model.weights = weights
    model.bias = bias

    metadata = json.loads(metadata_json) if metadata_json else {}
    return model, metadata


def _load_deep_policy_model(path: Path) -> Tuple[Any, Dict[str, Any]]:
    """
    Load Stable-Baselines deep policy format (.zip).

    This keeps backward compatibility for existing .npz models while allowing
    larger deep RL policies for users who install optional dependencies.
    """
    try:
        sb3_module = import_module("stable_baselines3")
        adapter_module = import_module("rl_ai.01_library_based.deep_policy_adapter")
    except Exception as exc:
        raise ImportError(
            "Deep RL model loading requires optional packages. "
            "Install with: pip install torch stable-baselines3 gymnasium"
        ) from exc

    StableBaselinesPolicyAdapter = getattr(adapter_module, "StableBaselinesPolicyAdapter")
    candidate_algorithms = ("PPO", "A2C", "DQN")
    load_errors: Dict[str, str] = {}

    for algorithm_name in candidate_algorithms:
        algorithm_class = getattr(sb3_module, algorithm_name, None)
        if algorithm_class is None:
            continue
        try:
            deep_model = algorithm_class.load(str(path), device="cpu")
            adapter = StableBaselinesPolicyAdapter(deep_model)
            metadata = _load_metadata_sidecar(path)
            metadata.setdefault("algorithm", algorithm_name.lower())
            metadata.setdefault("model_backend", "stable_baselines3")
            metadata.setdefault("model_format", "zip")
            return adapter, metadata
        except Exception as exc:
            load_errors[algorithm_name] = str(exc)

    error_summary = "; ".join(f"{name}: {msg}" for name, msg in load_errors.items())
    raise ValueError(f"Failed to load deep RL model from {path}. {error_summary}")


def load_policy_model(model_path: str) -> Tuple[Any, Dict[str, Any]]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if path.suffix.lower() == ".zip":
        return _load_deep_policy_model(path)

    return _load_linear_policy_model(path)
