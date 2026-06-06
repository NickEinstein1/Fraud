"""Business-layer governance: drift + model quality → operational actions."""

from __future__ import annotations

from typing import Any

from src.domain.entities import DriftAlert
from src.domain.enums import DriftSeverity, RetrainAction


class GovernancePolicy:
    """
    Encodes risk rules for an adversarial fraud environment.
    Top-down: business constraints drive serve/retrain decisions.
    """

    def __init__(self, config: dict[str, Any]):
        gov = config.get("governance", {})
        self.min_roc_auc = gov.get("min_roc_auc", 0.85)
        self.drift_features_alert = gov.get("drift_features_alert", 3)
        self.drift_features_block = gov.get("drift_features_block", 10)
        self.high_drift_p = gov.get("high_drift_p", 0.001)

    def classify_drift(self, p_value: float, drift_detected: bool) -> DriftSeverity:
        if not drift_detected:
            return DriftSeverity.NONE
        if p_value < self.high_drift_p:
            return DriftSeverity.HIGH
        if p_value < 0.01:
            return DriftSeverity.MEDIUM
        return DriftSeverity.LOW

    def alerts_from_reports(
        self,
        reports: list,
    ) -> list[DriftAlert]:
        alerts: list[DriftAlert] = []
        for r in reports:
            if not r.drift_detected:
                continue
            sev = self.classify_drift(r.p_value, r.drift_detected)
            alerts.append(
                DriftAlert(
                    feature=r.feature,
                    ks_statistic=r.statistic,
                    p_value=r.p_value,
                    severity=sev.value,
                )
            )
        return alerts

    def decide(
        self,
        roc_auc: float,
        drift_alerts: list[DriftAlert],
    ) -> RetrainAction:
        n_drift = len(drift_alerts)
        high_sev = sum(1 for a in drift_alerts if a.severity == DriftSeverity.HIGH.value)

        if roc_auc < self.min_roc_auc:
            return RetrainAction.RETRAIN_RECOMMENDED
        if n_drift >= self.drift_features_block or high_sev >= 5:
            return RetrainAction.BLOCK_SERVE
        if n_drift >= self.drift_features_alert:
            return RetrainAction.ALERT
        return RetrainAction.CONTINUE
