"""L2 Experience runtime: load production model, score, drift, governance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.domain.entities import PredictionBatch, SystemHealth
from src.domain.enums import RetrainAction
from src.engineering.features import FeatureEngineer
from src.etl.pipeline import ETLPipeline
from src.governance.policy import GovernancePolicy
from src.inference.engine import InferenceEngine
from src.monitoring.drift import DriftMonitor
from src.registry.model_registry import ModelRegistry
from src.api.schemas import get_feature_columns
from src.data.credit_dt import prepare_scoring_frame
from src.utils.config import load_config


class FraudServingRuntime:
    """Production-style serving path across experience + ML core + business layers."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()
        self.registry = ModelRegistry(
            self.config.get("paths", {}).get("registry", "artifacts/registry")
        )
        self.governance = GovernancePolicy(self.config)
        self.engine = InferenceEngine(self.config)
        self.engineer = FeatureEngineer(self.config)
        self.monitor = DriftMonitor(self.config)
        self._loaded = False
        self._production_version: str | None = None
        self._registry_roc_auc = 1.0
        self._governance_action = RetrainAction.CONTINUE

    def load(self) -> None:
        prod = self.registry.get_production()
        model_path = Path(self.config["paths"]["models"]) / "catboost_fraud.cbm"
        if prod and Path(prod.get("path", "")).exists():
            model_path = Path(prod["path"])
            self._production_version = prod.get("version")
            self._registry_roc_auc = float(prod.get("metrics", {}).get("roc_auc", 1.0))
        self.engine.load(model_path)
        self.engineer.load_artifacts(self.config["paths"]["artifacts"])
        baseline = Path(self.config["paths"]["baselines"]) / "feature_distributions.joblib"
        self.monitor.load_baseline(baseline)
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def score_batch(self, df: pd.DataFrame) -> tuple[pd.DataFrame, PredictionBatch]:
        self._ensure_loaded()
        df = prepare_scoring_frame(df, self.config)
        target = self.config["data"]["target_column"]
        feature_cols = get_feature_columns(self.config)
        feature_df = df[feature_cols] if all(c in df.columns for c in feature_cols) else df.drop(
            columns=[target], errors="ignore"
        )

        X = self.engineer.transform(feature_df)
        proba = self.engine.predict_proba(X)
        out = df.copy()
        out["fraud_probability"] = proba
        out["fraud_prediction"] = (proba >= 0.5).astype(int)

        reports = self.monitor.check(feature_df, list(feature_df.columns))
        alerts = self.governance.alerts_from_reports(reports)
        action = self.governance.decide(
            roc_auc=self._registry_roc_auc,
            drift_alerts=alerts,
        )
        # Scoring hold-out / test CSVs often drift vs train baseline — warn, don't hard-block UI
        if action == RetrainAction.BLOCK_SERVE and len(feature_df) >= self.monitor.drift_cfg.get(
            "min_samples", 100
        ):
            action = RetrainAction.ALERT
        self._governance_action = action

        batch = PredictionBatch(
            count=len(out),
            fraud_rate_predicted=float(out["fraud_prediction"].mean()),
            mean_probability=float(out["fraud_probability"].mean()),
            drifted_features=self.monitor.drifted_features(reports),
        )
        return out, batch

    def score_csv(self, path: str | Path) -> tuple[pd.DataFrame, PredictionBatch]:
        etl = ETLPipeline(self.config)
        df = etl.clean(etl.load(path))
        return self.score_batch(df)

    def health(self) -> SystemHealth:
        self._ensure_loaded()
        status = "healthy"
        if self._governance_action == RetrainAction.BLOCK_SERVE:
            status = "degraded"
        elif self._governance_action == RetrainAction.ALERT:
            status = "warning"
        return SystemHealth(
            status=status,
            model_version=self._production_version,
            last_drift_check=None,
            governance_action=self._governance_action.value,
        )
