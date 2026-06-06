from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dirs(config: dict[str, Any]) -> None:
    for key in ("artifacts", "models", "plots", "baselines"):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
