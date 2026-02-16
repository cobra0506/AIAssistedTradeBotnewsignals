import json
from pathlib import Path
from typing import Any, Dict, Optional


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    tmp.replace(path)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_generation_checkpoint(run_dir: Path, generation: int, payload: Dict[str, Any]) -> Path:
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / f"gen_{generation:04d}.json"
    save_json(checkpoint_path, payload)
    save_json(checkpoints_dir / "latest.json", payload)
    return checkpoint_path


def load_latest_checkpoint(run_dir: Path) -> Optional[Dict[str, Any]]:
    return load_json(run_dir / "checkpoints" / "latest.json")
