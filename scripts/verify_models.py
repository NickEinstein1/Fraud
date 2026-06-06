#!/usr/bin/env python3
"""Verify all trained models and scoring paths work."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.schemas import get_feature_columns
from src.data.credit_dt import prepare_scoring_frame
from src.data.datasets import dataset_kind
from src.inference.anomaly import AnomalyDetector
from src.inference.engine import InferenceEngine
from src.monitoring.drift import DriftMonitor
from src.realtime.ai_engine import RealtimeFraudAIEngine
from src.registry.model_registry import ModelRegistry
from src.serving.runtime import FraudServingRuntime
from src.utils.config import load_config

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    config = load_config(ROOT / "config.yaml")
    models_dir = ROOT / config["paths"]["models"]
    baselines_dir = ROOT / config["paths"]["baselines"]

    print("\n=== Artifact files ===")
    catboost_path = models_dir / "catboost_fraud.cbm"
    anomaly_path = models_dir / "anomaly_iforest.joblib"
    scaler_path = ROOT / config["paths"]["artifacts"] / "scaler.joblib"
    baseline_path = baselines_dir / "feature_distributions.joblib"

    check("CatBoost model", catboost_path.exists(), str(catboost_path))
    check("Isolation Forest", anomaly_path.exists(), str(anomaly_path))
    check("Feature scaler", scaler_path.exists(), str(scaler_path))
    check("Drift baseline", baseline_path.exists(), str(baseline_path))

    print("\n=== Model registry ===")
    registry = ModelRegistry(ROOT / config["paths"]["registry"])
    prod = registry.get_production()
    check("Production model registered", prod is not None, str(prod.get("version") if prod else ""))
    if prod:
        check("Registry ROC-AUC >= min", prod.get("metrics", {}).get("roc_auc", 0) >= 0.85)

    print("\n=== Batch scoring (FraudServingRuntime) ===")
    runtime = FraudServingRuntime(config)
    runtime.load()
    if dataset_kind(config) == "credit_dt":
        sample = pd.read_csv(ROOT / config["data"]["credit_dt"]["test_path"], nrows=150)
        sample = prepare_scoring_frame(sample, config)
        target_col = "Class"
    else:
        sample = pd.read_csv(ROOT / config["data"]["path"], nrows=150)
        target_col = config["data"]["target_column"]
    feature_cols = get_feature_columns(config)
    feature_df = sample[feature_cols]
    out, summary = runtime.score_batch(feature_df)
    check("Batch score rows", len(out) == 150)
    check("Has fraud_probability", "fraud_probability" in out.columns)
    check("Has predictions", "fraud_prediction" in out.columns)
    frauds = int(out["fraud_prediction"].sum())
    print(f"       → {frauds}/150 flagged in sample")

    print("\n=== Manual single-row scoring ===")
    row = {c: float(sample[c].iloc[0]) for c in feature_cols}
    one = pd.DataFrame([row])[feature_cols]
    out1, _ = runtime.score_batch(one)
    prob = float(out1["fraud_probability"].iloc[0])
    check("Single row probability in [0,1]", 0.0 <= prob <= 1.0, f"prob={prob:.6f}")

    print("\n=== Real-time AI engine ===")
    rt = RealtimeFraudAIEngine(config)
    rt.load()
    check("Realtime engine loaded", rt._loaded)
    snaps = list(rt.stream(max_batches=5))
    check("Realtime stream ticks", len(snaps) == 5, f"last gov={snaps[-1].governance_action}")

    print("\n=== Drift monitor ===")
    monitor = DriftMonitor(config)
    monitor.load_baseline(baseline_path)
    reports = monitor.check(feature_df, feature_cols)
    check("KS reports generated", len(reports) > 0)

    print("\n=== CatBoost direct load ===")
    engine = InferenceEngine(config)
    engine.load(catboost_path)
    check("CatBoost predict", len(engine.predict_proba(feature_df.values)) == 150)

    if anomaly_path.exists():
        det = AnomalyDetector(config)
        det.load(anomaly_path)
        scores = det.predict_score(feature_df.values)
        check("Anomaly scores", len(scores) == 150)

    if target_col in sample.columns:
        known_fraud = sample[sample[target_col] == 1]
        known_legit = sample[sample[target_col] == 0]
        if len(known_fraud) > 0:
            fo, _ = runtime.score_batch(known_fraud[feature_cols])
            check(
                "Detects known fraud",
                int(fo["fraud_prediction"].sum()) >= max(1, len(known_fraud) // 2),
                f"{int(fo['fraud_prediction'].sum())}/{len(known_fraud)} flagged",
            )
        if len(known_legit) > 0:
            lo, _ = runtime.score_batch(known_legit[feature_cols].head(10))
            false_pos = int(lo["fraud_prediction"].sum())
            check("Low false positives on legit", false_pos <= 2, f"{false_pos}/10 flagged")

    print("\n" + "=" * 50)
    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        return 1
    print("All checks passed — models ready for UI, API, and real-time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
