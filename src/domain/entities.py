"""Business and system entities crossing architectural layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DriftAlert:
    feature: str
    ks_statistic: float
    p_value: float
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "ks_statistic": self.ks_statistic,
            "p_value": self.p_value,
            "severity": self.severity,
        }


@dataclass
class ModelArtifact:
    model_id: str
    version: str
    path: str
    algorithm: str
    metrics: dict[str, float]
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "path": self.path,
            "algorithm": self.algorithm,
            "metrics": self.metrics,
            "created_at": self.created_at,
        }


@dataclass
class PredictionBatch:
    count: int
    fraud_rate_predicted: float
    mean_probability: float
    drifted_features: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "fraud_rate_predicted": self.fraud_rate_predicted,
            "mean_probability": self.mean_probability,
            "drifted_features": self.drifted_features,
        }


@dataclass
class SystemHealth:
    status: str
    model_version: str | None
    last_drift_check: str | None
    governance_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model_version": self.model_version,
            "last_drift_check": self.last_drift_check,
            "governance_action": self.governance_action,
        }


@dataclass
class PipelineManifest:
    run_id: str
    stages_completed: list[str]
    metrics: dict[str, Any]
    drift_alerts: list[DriftAlert]
    governance_action: str
    model_version: str | None
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stages_completed": self.stages_completed,
            "metrics": self.metrics,
            "drift_alerts": [a.to_dict() for a in self.drift_alerts],
            "governance_action": self.governance_action,
            "model_version": self.model_version,
            "created_at": self.created_at,
        }
