"""Orchestration stages — each maps to an AI layer and ML module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from src.architecture.context import PipelineContext
from src.data.datasets import dataset_kind
from src.data.synthetic import ensure_dataset
from src.domain.entities import ModelArtifact
from src.domain.enums import PipelineStage
from src.engineering.features import FeatureEngineer
from src.etl.pipeline import ETLPipeline
from src.inference.anomaly import AnomalyDetector
from src.inference.engine import InferenceEngine
from src.monitoring.drift import DriftMonitor
from src.observability.telemetry import Telemetry
from src.utils.config import ensure_dirs


class Stage(ABC):
    name: PipelineStage

    @abstractmethod
    def execute(self, ctx: PipelineContext, telemetry: Telemetry) -> None:
        ...


class DataIngestStage(Stage):
    name = PipelineStage.DATA_INGEST

    def __init__(self, generate_if_missing: bool = True):
        self.generate_if_missing = generate_if_missing

    def execute(self, ctx: PipelineContext, telemetry: Telemetry) -> None:
        ensure_dirs(ctx.config)
        if dataset_kind(ctx.config) == "credit_dt":
            cfg = ctx.config["data"]["credit_dt"]
            train_p = Path(cfg["train_path"])
            test_p = Path(cfg["test_path"])
            if not train_p.exists() or not test_p.exists():
                raise FileNotFoundError(
                    f"credit_dt files missing: {train_p} and/or {test_p}"
                )
            ctx.data_path = str(train_p)
            telemetry.emit(
                self.name,
                "credit_dt train/test loaded",
                {"train": str(train_p), "test": str(test_p), "dataset": "credit_dt"},
            )
        else:
            data_path = Path(ctx.config["data"]["path"])
            if self.generate_if_missing and not data_path.exists():
                telemetry.emit(
                    self.name,
                    "Generating synthetic dataset (Kaggle schema)",
                    {"path": str(data_path)},
                )
                ensure_dataset(data_path)
            ctx.data_path = str(data_path)
            telemetry.emit(self.name, "Data plane ready", {"path": ctx.data_path})
        ctx.record_stage(self.name.value)


class ETLStage(Stage):
    name = PipelineStage.ETL

    def execute(self, ctx: PipelineContext, telemetry: Telemetry) -> None:
        etl = ETLPipeline(ctx.config)
        ctx.etl_result = etl.run()
        r = ctx.etl_result
        telemetry.emit(
            self.name,
            "ETL + SMOTE complete",
            {
                "train_rows": len(r.y_train),
                "test_rows": len(r.y_test),
                "test_fraud_rate": float(r.y_test.mean()),
            },
        )
        ctx.record_stage(self.name.value)


class FeatureEngineeringStage(Stage):
    name = PipelineStage.FEATURE_ENGINEERING

    def execute(self, ctx: PipelineContext, telemetry: Telemetry) -> None:
        assert ctx.etl_result is not None
        engineer = FeatureEngineer(ctx.config)
        ctx.eng_result = engineer.fit_transform(
            ctx.etl_result.X_train,
            ctx.etl_result.X_test,
            ctx.etl_result.y_train,
        )
        ctx.engineer = engineer
        engineer.save_artifacts(ctx.config["paths"]["artifacts"])

        if ctx.eng_result.pca_2d is not None:
            plot_path = Path(ctx.config["paths"]["plots"]) / "pca_train.png"
            engineer.plot_pca(ctx.eng_result.pca_2d, ctx.etl_result.y_train, plot_path)
            telemetry.emit(self.name, "PCA visualization saved", {"path": str(plot_path)})

        telemetry.emit(self.name, "Feature engineering complete")
        ctx.record_stage(self.name.value)


class TrainStage(Stage):
    name = PipelineStage.TRAIN

    def execute(self, ctx: PipelineContext, telemetry: Telemetry) -> None:
        assert ctx.etl_result is not None and ctx.eng_result is not None
        engine = InferenceEngine(ctx.config)
        engine.train(ctx.eng_result.X_train, ctx.etl_result.y_train)
        ctx.engine = engine
        model_path = Path(ctx.config["paths"]["models"]) / "catboost_fraud.cbm"
        engine.save(model_path)
        ctx.model_path = str(model_path)
        telemetry.emit(self.name, "CatBoost model trained", {"path": ctx.model_path})
        ctx.record_stage(self.name.value)


class EvaluateStage(Stage):
    name = PipelineStage.EVALUATE

    def execute(self, ctx: PipelineContext, telemetry: Telemetry) -> None:
        assert ctx.etl_result is not None and ctx.eng_result is not None and ctx.engine
        metrics = ctx.engine.evaluate(ctx.eng_result.X_test, ctx.etl_result.y_test)
        ctx.train_metrics = metrics
        telemetry.emit(
            self.name,
            "Hold-out evaluation",
            {"roc_auc": metrics.roc_auc, "f1": metrics.f1},
        )
        ctx.record_stage(self.name.value)


class DriftBaselineStage(Stage):
    name = PipelineStage.DRIFT_BASELINE

    def execute(self, ctx: PipelineContext, telemetry: Telemetry) -> None:
        assert ctx.etl_result is not None
        monitor = DriftMonitor(ctx.config)
        monitor.fit_baseline(ctx.etl_result.X_train_raw)
        baseline_path = Path(ctx.config["paths"]["baselines"]) / "feature_distributions.joblib"
        monitor.save_baseline(baseline_path)
        ctx.monitor = monitor
        ctx.baseline_path = str(baseline_path)
        telemetry.emit(self.name, "KS baseline persisted", {"path": ctx.baseline_path})
        ctx.record_stage(self.name.value)


class DriftCheckStage(Stage):
    name = PipelineStage.DRIFT_CHECK

    def execute(self, ctx: PipelineContext, telemetry: Telemetry) -> None:
        assert ctx.etl_result is not None and ctx.monitor is not None
        reports = ctx.monitor.check(ctx.etl_result.X_test, ctx.etl_result.feature_names)
        drift_path = Path(ctx.config["paths"]["artifacts"]) / "drift_report.csv"
        ctx.monitor.summary(reports).to_csv(drift_path, index=False)
        drifted = ctx.monitor.drifted_features(reports)
        telemetry.emit(
            self.name,
            "Drift check on hold-out (serving proxy)",
            {"drifted_count": len(drifted), "drifted_sample": drifted[:5]},
        )
        ctx._drift_reports = reports
        ctx.record_stage(self.name.value)


class AnomalyModelStage(Stage):
    name = PipelineStage.ANOMALY_MODEL

    def execute(self, ctx: PipelineContext, telemetry: Telemetry) -> None:
        if not ctx.config.get("realtime", {}).get("anomaly", {}).get("enabled", True):
            telemetry.emit(self.name, "Anomaly model skipped (disabled)")
            ctx.record_stage(self.name.value)
            return
        assert ctx.etl_result is not None
        detector = AnomalyDetector(ctx.config)
        out_path = Path(ctx.config["paths"]["models"]) / "anomaly_iforest.joblib"
        X = ctx.etl_result.X_train_raw
        max_rows = ctx.config.get("realtime", {}).get("anomaly", {}).get("fit_max_rows", 8000)
        if len(X) > max_rows:
            X = X.sample(n=max_rows, random_state=ctx.config["data"]["random_state"])
        detector.fit(X.values)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        detector.save(out_path)
        telemetry.emit(
            self.name,
            "Isolation Forest trained for real-time AI",
            {"path": str(out_path), "rows": len(X)},
        )
        ctx.record_stage(self.name.value)


class RegisterStage(Stage):
    name = PipelineStage.REGISTER

    def execute(self, ctx: PipelineContext, telemetry: Telemetry) -> None:
        assert ctx.train_metrics is not None and ctx.model_path
        version = f"v_{uuid4().hex[:8]}"
        artifact = ModelArtifact(
            model_id="fraud_catboost",
            version=version,
            path=ctx.model_path,
            algorithm="CatBoost",
            metrics={"roc_auc": ctx.train_metrics.roc_auc, "f1": ctx.train_metrics.f1},
        )
        ctx.model_artifact = artifact

        if ctx.engine and ctx.etl_result:
            importance = InferenceEngine.feature_importance(
                ctx.engine.model, ctx.etl_result.feature_names
            )
            imp_path = Path(ctx.config["paths"]["artifacts"]) / "feature_importance.csv"
            importance.head(15).to_csv(imp_path, index=False)

        telemetry.emit(self.name, "Model artifact prepared", artifact.to_dict())
        ctx.record_stage(self.name.value)
