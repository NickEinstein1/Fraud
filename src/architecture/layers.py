"""Top-down mapping: pipeline stages → AI reference layers."""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.enums import AILayer, PipelineStage


STAGE_TO_LAYER: dict[PipelineStage, AILayer] = {
    PipelineStage.DATA_INGEST: AILayer.DATA,
    PipelineStage.ETL: AILayer.ML_CORE,
    PipelineStage.FEATURE_ENGINEERING: AILayer.ML_CORE,
    PipelineStage.TRAIN: AILayer.ML_CORE,
    PipelineStage.EVALUATE: AILayer.ML_CORE,
    PipelineStage.DRIFT_BASELINE: AILayer.ML_CORE,
    PipelineStage.DRIFT_CHECK: AILayer.ML_CORE,
    PipelineStage.GOVERNANCE: AILayer.BUSINESS,
    PipelineStage.REGISTER: AILayer.MLOPS,
    PipelineStage.ANOMALY_MODEL: AILayer.ML_CORE,
}


def layer_for_stage(stage: PipelineStage) -> AILayer:
    return STAGE_TO_LAYER[stage]


@dataclass(frozen=True)
class LayerSpec:
    layer: AILayer
    name: str
    responsibility: str


AI_WORLD_STACK: tuple[LayerSpec, ...] = (
    LayerSpec(
        AILayer.BUSINESS,
        "Business & Risk",
        "Fraud SLAs, adversarial drift policy, retrain/block decisions",
    ),
    LayerSpec(
        AILayer.EXPERIENCE,
        "Experience",
        "REST API, batch CLI, analyst-facing predictions",
    ),
    LayerSpec(
        AILayer.ORCHESTRATION,
        "Orchestration",
        "Stage graph, shared context, run lifecycle",
    ),
    LayerSpec(
        AILayer.ML_CORE,
        "ML Core",
        "ETL, SMOTE, feature engineering, CatBoost, KS drift",
    ),
    LayerSpec(
        AILayer.DATA,
        "Data Plane",
        "Kaggle/PaySim ingestion, synthetic fallback, schemas",
    ),
    LayerSpec(
        AILayer.MLOPS,
        "MLOps",
        "Model registry, versioning, artifact lineage",
    ),
    LayerSpec(
        AILayer.OBSERVABILITY,
        "Observability",
        "Structured telemetry, manifests, audit trail",
    ),
)


class AIWorldStack:
    """Queryable view of the seven-layer AI architecture."""

    @staticmethod
    def layers() -> tuple[LayerSpec, ...]:
        return AI_WORLD_STACK

    @staticmethod
    def describe() -> str:
        lines = ["AI World Architecture (top → down):", ""]
        for i, spec in enumerate(AI_WORLD_STACK, start=1):
            lines.append(f"  L{i} [{spec.layer.value}] {spec.name}")
            lines.append(f"      → {spec.responsibility}")
        return "\n".join(lines)
