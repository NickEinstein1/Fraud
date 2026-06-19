#!/usr/bin/env python3
"""Compute and save full model metrics snapshot for documentation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.etl.pipeline import ETLPipeline
from src.engineering.features import FeatureEngineer
from src.inference.engine import InferenceEngine
from src.utils.config import load_config

OUT = ROOT / "docs" / "model_metrics_snapshot.json"


def main() -> None:
    config = load_config()
    etl = ETLPipeline(config)
    result = etl.run()
    eng = FeatureEngineer(config)
    eng_result = eng.fit_transform(result.X_train, result.X_test, result.y_train)

    engine = InferenceEngine(config)
    engine.train(eng_result.X_train, result.y_train)
    metrics = engine.evaluate(eng_result.X_test, result.y_test)

    proba = engine.predict_proba(eng_result.X_test)
    preds = (proba >= 0.5).astype(int)
    y = result.y_test.values

    fi = engine.feature_importance(engine.model, result.feature_names)

    threshold_sweep = {}
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p = (proba >= t).astype(int)
        threshold_sweep[str(t)] = {
            "precision": float(precision_score(y, p, zero_division=0)),
            "recall": float(recall_score(y, p, zero_division=0)),
            "f1": float(f1_score(y, p, zero_division=0)),
            "flagged": int(p.sum()),
        }

    snapshot = {
        "roc_auc": float(metrics.roc_auc),
        "f1": float(metrics.f1),
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "average_precision": float(average_precision_score(y, proba)),
        "confusion_matrix": metrics.confusion.tolist(),
        "classification_report": metrics.report,
        "train_rows_after_smote": len(result.y_train),
        "test_rows": len(result.y_test),
        "test_fraud_count": int(result.y_test.sum()),
        "test_fraud_rate": float(result.y_test.mean()),
        "feature_count": len(result.feature_names),
        "catboost_tree_count": engine.model.tree_count_,
        "catboost_depth": config["model"]["catboost"]["depth"],
        "catboost_iterations": config["model"]["catboost"]["iterations"],
        "catboost_learning_rate": config["model"]["catboost"]["learning_rate"],
        "threshold_sweep": threshold_sweep,
        "feature_importance": fi.to_dict(orient="records"),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snapshot, indent=2))
    print(f"Metrics saved: {OUT}")
    print(f"ROC-AUC: {snapshot['roc_auc']:.4f}  F1: {snapshot['f1']:.4f}  Recall: {snapshot['recall']:.4f}")


if __name__ == "__main__":
    main()
