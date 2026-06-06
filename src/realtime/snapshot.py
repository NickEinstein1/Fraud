"""Real-time AI event payload broadcast to UI and WebSocket clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RealtimeSnapshot:
    """Single tick of the real-time fraud + drift AI pipeline."""

    batch_id: int
    transactions_in_batch: int
    total_processed: int
    fraud_detected: int
    fraud_rate_window: float
    mean_fraud_probability: float
    mean_anomaly_score: float | None
    drifted_features: list[str]
    top_drift: list[dict[str, Any]]
    drift_alerts: list[dict[str, Any]]
    governance_action: str
    model_version: str | None
    window_fill: int
    window_capacity: int
    blocked: bool
    timestamp: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "transactions_in_batch": self.transactions_in_batch,
            "total_processed": self.total_processed,
            "fraud_detected": self.fraud_detected,
            "fraud_rate_window": self.fraud_rate_window,
            "mean_fraud_probability": self.mean_fraud_probability,
            "mean_anomaly_score": self.mean_anomaly_score,
            "drifted_features": self.drifted_features,
            "top_drift": self.top_drift,
            "drift_alerts": self.drift_alerts,
            "governance_action": self.governance_action,
            "model_version": self.model_version,
            "window_fill": self.window_fill,
            "window_capacity": self.window_capacity,
            "blocked": self.blocked,
            "timestamp": self.timestamp,
        }
