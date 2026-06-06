"""MLOps model registry — versioned artifacts and promotion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.domain.entities import ModelArtifact


class ModelRegistry:
    def __init__(self, registry_dir: str | Path):
        self.root = Path(registry_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"models": [], "production": None}
        return json.loads(self.index_path.read_text())

    def _save_index(self, index: dict[str, Any]) -> None:
        self.index_path.write_text(json.dumps(index, indent=2))

    def register(self, artifact: ModelArtifact, promote: bool = True) -> None:
        index = self._load_index()
        entry = artifact.to_dict()
        index["models"].append(entry)
        if promote:
            index["production"] = artifact.version
        self._save_index(index)

        version_dir = self.root / artifact.version
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "metadata.json").write_text(json.dumps(entry, indent=2))

    def get_production(self) -> dict[str, Any] | None:
        index = self._load_index()
        version = index.get("production")
        if not version:
            return None
        meta_path = self.root / version / "metadata.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        for m in index.get("models", []):
            if m.get("version") == version:
                return m
        return None

    def list_versions(self) -> list[str]:
        index = self._load_index()
        return [m["version"] for m in index.get("models", [])]
