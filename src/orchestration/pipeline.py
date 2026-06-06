"""Top-down orchestrator: Business → … → Observability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.architecture.context import PipelineContext
from src.architecture.layers import AIWorldStack
from src.domain.entities import PipelineManifest
from src.domain.enums import PipelineStage, RetrainAction
from src.governance.policy import GovernancePolicy
from src.observability.telemetry import Telemetry
from src.orchestration.stages import (
    AnomalyModelStage,
    DataIngestStage,
    DriftBaselineStage,
    DriftCheckStage,
    ETLStage,
    EvaluateStage,
    FeatureEngineeringStage,
    RegisterStage,
    TrainStage,
)
from src.registry.model_card import generate_model_cards
from src.registry.model_registry import ModelRegistry
from src.utils.config import load_config


class FraudAIPipeline:
    """
    L3 Orchestration: executes the stage graph top-down and binds
    ML modules (L4) to data (L5), MLOps (L6), and telemetry (L7).
    """

    def __init__(self, config: dict[str, Any] | None = None, run_id: str | None = None):
        self.config = config or load_config()
        self.run_id = run_id or f"run_{uuid4().hex[:12]}"
        self.telemetry = Telemetry(self.config, self.run_id)
        self.registry = ModelRegistry(
            self.config.get("paths", {}).get("registry", "artifacts/registry")
        )
        self.governance = GovernancePolicy(self.config)

    @staticmethod
    def print_architecture() -> None:
        print(AIWorldStack.describe())

    def _training_stages(self, generate_if_missing: bool):
        return [
            DataIngestStage(generate_if_missing),
            ETLStage(),
            FeatureEngineeringStage(),
            TrainStage(),
            EvaluateStage(),
            DriftBaselineStage(),
            DriftCheckStage(),
            AnomalyModelStage(),
            RegisterStage(),
        ]

    def run_training(self, generate_if_missing: bool = True) -> PipelineManifest:
        ctx = PipelineContext(config=self.config, run_id=self.run_id)
        self.telemetry.emit(
            PipelineStage.DATA_INGEST,
            "Pipeline run started",
            {"run_id": self.run_id},
        )

        for stage in self._training_stages(generate_if_missing):
            stage.execute(ctx, self.telemetry)

        assert ctx.train_metrics is not None and ctx.model_artifact is not None
        reports = getattr(ctx, "_drift_reports", [])
        ctx.drift_alerts = self.governance.alerts_from_reports(reports)
        ctx.governance_action = self.governance.decide(
            ctx.train_metrics.roc_auc,
            ctx.drift_alerts,
        )

        self.telemetry.emit(
            PipelineStage.GOVERNANCE,
            f"Governance decision: {ctx.governance_action.value}",
            {
                "drift_alert_count": len(ctx.drift_alerts),
                "action": ctx.governance_action.value,
            },
        )
        ctx.record_stage(PipelineStage.GOVERNANCE.value)

        promote = ctx.governance_action != RetrainAction.BLOCK_SERVE
        self.registry.register(ctx.model_artifact, promote=promote)

        card_paths = generate_model_cards(
            self.config,
            version=ctx.model_artifact.version,
            metrics={
                "roc_auc": ctx.train_metrics.roc_auc,
                "f1": ctx.train_metrics.f1,
            },
            drift_alerts=[a.to_dict() for a in ctx.drift_alerts],
            governance_action=ctx.governance_action.value,
            run_id=self.run_id,
        )
        self.telemetry.emit(
            PipelineStage.REGISTER,
            "Model registered + model cards written",
            {
                "version": ctx.model_artifact.version,
                "promoted": promote,
                "model_cards": [str(p) for p in card_paths[:2]],
            },
        )

        manifest = PipelineManifest(
            run_id=self.run_id,
            stages_completed=ctx.stages_completed,
            metrics={
                "roc_auc": ctx.train_metrics.roc_auc,
                "f1": ctx.train_metrics.f1,
            },
            drift_alerts=ctx.drift_alerts,
            governance_action=ctx.governance_action.value,
            model_version=ctx.model_artifact.version,
        )
        manifest_path = self.telemetry.save_manifest(manifest.to_dict())

        summary_path = Path(self.config["paths"]["artifacts"]) / "run_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    **manifest.to_dict(),
                    "manifest_path": str(manifest_path),
                    "telemetry": str(self.telemetry.events_path),
                },
                indent=2,
            )
        )

        print("\n" + AIWorldStack.describe())
        print(f"\nRun ID: {self.run_id}")
        print(f"ROC-AUC: {ctx.train_metrics.roc_auc:.4f} | F1: {ctx.train_metrics.f1:.4f}")
        print(f"Governance: {ctx.governance_action.value}")
        print(f"Model version: {ctx.model_artifact.version}")
        print(f"Manifest: {manifest_path}")
        return manifest
