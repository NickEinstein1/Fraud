"""Real-time AI engine: CatBoost fraud + Isolation Forest anomaly + live KS drift."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator

import numpy as np
import pandas as pd

from src.domain.enums import RetrainAction
from src.governance.policy import GovernancePolicy
from src.inference.anomaly import AnomalyDetector
from src.inference.engine import InferenceEngine
from src.engineering.features import FeatureEngineer
from src.monitoring.drift import DriftMonitor
from src.realtime.buffer import SlidingWindowBuffer
from src.realtime.snapshot import RealtimeSnapshot
from src.realtime.stream import TransactionStream
from src.registry.model_registry import ModelRegistry
from src.data.credit_dt import prepare_scoring_frame
from src.data.datasets import get_stream_data_path
from src.utils.config import load_config


class RealtimeFraudAIEngine:
    """
    L4 Real-Time AI Core (extends ML Core for streaming).

    Models:
      - CatBoost: supervised fraud probability
      - Isolation Forest: unsupervised anomaly score
      - KS Monitor: distributional drift vs training baseline

    Feeds L1 governance and L2/L7 rendering surfaces.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()
        rt = self.config.get("realtime", {})
        self.window_size = rt.get("window_size", 500)
        self.drift_interval = rt.get("drift_check_interval", 50)
        self.fraud_threshold = rt.get("fraud_threshold", 0.5)
        self.anomaly_enabled = rt.get("anomaly", {}).get("enabled", True)

        self.buffer = SlidingWindowBuffer(self.window_size)
        self.engine = InferenceEngine(self.config)
        self.engineer = FeatureEngineer(self.config)
        self.monitor = DriftMonitor(self.config)
        self.anomaly = AnomalyDetector(self.config)
        self.governance = GovernancePolicy(self.config)
        self.registry = ModelRegistry(
            self.config.get("paths", {}).get("registry", "artifacts/registry")
        )

        self._loaded = False
        self._batch_id = 0
        self._total_processed = 0
        self._governance_action = RetrainAction.CONTINUE
        self._production_version: str | None = None
        self._last_drift_check_at = 0
        self._feature_cols: list[str] = []
        self._history: list[RealtimeSnapshot] = []
        self._max_history = rt.get("history_size", 200)
        self._blocked = False
        self._registry_roc_auc = 1.0

    def load(self) -> None:
        prod = self.registry.get_production()
        model_path = Path(self.config["paths"]["models"]) / "catboost_fraud.cbm"
        if prod and Path(prod.get("path", "")).exists():
            model_path = Path(prod["path"])
            self._production_version = prod.get("version")
            self._registry_roc_auc = prod.get("metrics", {}).get("roc_auc", 1.0)

        self.engine.load(model_path)
        self.engineer.load_artifacts(self.config["paths"]["artifacts"])
        baseline = Path(self.config["paths"]["baselines"]) / "feature_distributions.joblib"
        self.monitor.load_baseline(baseline)

        from src.api.schemas import get_feature_columns

        self._feature_cols = get_feature_columns(self.config)
        if self.monitor.baseline:
            self._feature_cols = [c for c in self._feature_cols if c in self.monitor.baseline]

        if self.anomaly_enabled:
            self._ensure_anomaly_model()

        self._loaded = True

    def _ensure_anomaly_model(self) -> None:
        path = Path(self.config["paths"]["models"]) / "anomaly_iforest.joblib"
        if path.exists():
            self.anomaly.load(path)
            return
        self.anomaly_enabled = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def process_batch(self, df: pd.DataFrame) -> RealtimeSnapshot:
        self._ensure_loaded()
        if self._blocked:
            return self._empty_snapshot(blocked=True)

        df = prepare_scoring_frame(df, self.config)
        target = self.config["data"]["target_column"]
        feature_df = df.drop(columns=[target], errors="ignore")
        if not self._feature_cols:
            self._feature_cols = list(feature_df.columns)

        X = self.engineer.transform(feature_df)
        proba = self.engine.predict_proba(X)
        preds = (proba >= self.fraud_threshold).astype(int)

        anomaly_mean: float | None = None
        if self.anomaly_enabled and self.anomaly.model is not None:
            scores = self.anomaly.predict_score(X)
            anomaly_mean = float(np.mean(scores))

        self.buffer.append(feature_df)
        self._batch_id += 1
        self._total_processed += len(df)

        drifted: list[str] = []
        top_drift: list[dict[str, Any]] = []
        alerts: list[dict[str, Any]] = []

        if (
            len(self.buffer) >= self.monitor.drift_cfg.get("min_samples", 100)
            and self._total_processed - self._last_drift_check_at >= self.drift_interval
        ):
            self._last_drift_check_at = self._total_processed
            window_df = self.buffer.to_dataframe()
            reports = self.monitor.check(window_df, self._feature_cols)
            drifted = self.monitor.drifted_features(reports)
            summary = self.monitor.summary(reports)
            if not summary.empty:
                top = summary.head(5)
                top_drift = top.to_dict(orient="records")
            alert_objs = self.governance.alerts_from_reports(reports)
            alerts = [a.to_dict() for a in alert_objs]
            action = self.governance.decide(
                self._registry_roc_auc,
                alert_objs,
            )
            if action == RetrainAction.BLOCK_SERVE:
                action = RetrainAction.ALERT
            self._governance_action = action

        window_preds = preds  # current batch only for fraud count
        snap = RealtimeSnapshot(
            batch_id=self._batch_id,
            transactions_in_batch=len(df),
            total_processed=self._total_processed,
            fraud_detected=int(window_preds.sum()),
            fraud_rate_window=float(preds.mean()),
            mean_fraud_probability=float(proba.mean()),
            mean_anomaly_score=anomaly_mean,
            drifted_features=drifted,
            top_drift=top_drift,
            drift_alerts=alerts,
            governance_action=self._governance_action.value,
            model_version=self._production_version,
            window_fill=len(self.buffer),
            window_capacity=self.window_size,
            blocked=self._blocked,
        )
        self._history.append(snap)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
        return snap

    def _empty_snapshot(self, blocked: bool = False) -> RealtimeSnapshot:
        return RealtimeSnapshot(
            batch_id=self._batch_id,
            transactions_in_batch=0,
            total_processed=self._total_processed,
            fraud_detected=0,
            fraud_rate_window=0.0,
            mean_fraud_probability=0.0,
            mean_anomaly_score=None,
            drifted_features=[],
            top_drift=[],
            drift_alerts=[],
            governance_action=self._governance_action.value,
            model_version=self._production_version,
            window_fill=len(self.buffer),
            window_capacity=self.window_size,
            blocked=blocked,
        )

    def _default_stream_path(self) -> str:
        return get_stream_data_path(self.config)

    def stream(
        self,
        source_path: str | Path | None = None,
        max_batches: int | None = None,
        callback: Callable[[RealtimeSnapshot], None] | None = None,
    ) -> Iterator[RealtimeSnapshot]:
        path = source_path or self._default_stream_path()
        rt = self.config.get("realtime", {})
        stream = TransactionStream(
            path,
            batch_size=rt.get("stream_batch_size", 10),
            shuffle=True,
            random_state=self.config["data"]["random_state"],
        )
        count = 0
        for batch in stream.batches():
            if max_batches is not None and count >= max_batches:
                break
            snap = self.process_batch(batch)
            if callback:
                callback(snap)
            yield snap
            count += 1
            if self._blocked:
                break

    async def stream_async(
        self,
        source_path: str | Path | None = None,
        max_batches: int | None = None,
    ) -> AsyncIterator[RealtimeSnapshot]:
        path = source_path or self._default_stream_path()
        rt = self.config.get("realtime", {})
        tps = rt.get("transactions_per_second", 20)
        stream = TransactionStream(
            path,
            batch_size=rt.get("stream_batch_size", 10),
            shuffle=True,
            random_state=self.config["data"]["random_state"],
        )
        count = 0
        async for batch in stream.async_batches(tps):
            if max_batches is not None and count >= max_batches:
                break
            yield self.process_batch(batch)
            count += 1
            if self._blocked:
                break

    def latest_snapshot(self) -> RealtimeSnapshot | None:
        return self._history[-1] if self._history else None

    def history(self, n: int = 50) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._history[-n:]]

    def status(self) -> dict[str, Any]:
        snap = self.latest_snapshot()
        return {
            "loaded": self._loaded,
            "total_processed": self._total_processed,
            "governance_action": self._governance_action.value,
            "blocked": self._blocked,
            "model_version": self._production_version,
            "window_fill": len(self.buffer),
            "window_capacity": self.window_size,
            "anomaly_enabled": self.anomaly_enabled,
            "latest": snap.to_dict() if snap else None,
        }
