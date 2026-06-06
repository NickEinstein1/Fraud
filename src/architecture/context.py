"""Shared execution context passed top-down through orchestration stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.domain.entities import DriftAlert, ModelArtifact
from src.domain.enums import RetrainAction
from src.etl.pipeline import ETLResult
from src.engineering.features import EngineeringResult, FeatureEngineer
from src.inference.engine import InferenceEngine, TrainMetrics
from src.monitoring.drift import DriftMonitor


@dataclass
class PipelineContext:
    """Mutable state bag for a single pipeline run (or serving session)."""

    config: dict[str, Any]
    run_id: str
    stages_completed: list[str] = field(default_factory=list)

    # Data plane
    data_path: str | None = None

    # ML core artifacts
    etl_result: ETLResult | None = None
    eng_result: EngineeringResult | None = None
    engineer: FeatureEngineer | None = None
    engine: InferenceEngine | None = None
    train_metrics: TrainMetrics | None = None
    monitor: DriftMonitor | None = None

    # MLOps / governance
    model_artifact: ModelArtifact | None = None
    drift_alerts: list[DriftAlert] = field(default_factory=list)
    governance_action: RetrainAction = RetrainAction.CONTINUE

    # Paths written this run
    model_path: str | None = None
    baseline_path: str | None = None

    def record_stage(self, stage_name: str) -> None:
        self.stages_completed.append(stage_name)
