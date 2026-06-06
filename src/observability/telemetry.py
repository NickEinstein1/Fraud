"""Structured observability across AI layers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.architecture.layers import layer_for_stage
from src.domain.enums import AILayer, PipelineStage


class Telemetry:
    """Append-only event log and run manifest writer."""

    def __init__(self, config: dict[str, Any], run_id: str):
        self.run_id = run_id
        self.log_dir = Path(config["paths"]["artifacts"]) / "telemetry"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.log_dir / f"{run_id}_events.jsonl"
        self._events: list[dict[str, Any]] = []

    def emit(
        self,
        stage: PipelineStage,
        message: str,
        payload: dict[str, Any] | None = None,
        layer: AILayer | None = None,
    ) -> None:
        event = {
            "run_id": self.run_id,
            "stage": stage.value,
            "layer": (layer or layer_for_stage(stage)).value,
            "message": message,
            "payload": payload or {},
        }
        self._events.append(event)
        with open(self.events_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def save_manifest(self, manifest: dict[str, Any]) -> Path:
        path = self.log_dir / f"{self.run_id}_manifest.json"
        path.write_text(json.dumps(manifest, indent=2))
        return path
