"""CatBoost inference engine for fraud classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@dataclass
class TrainMetrics:
    roc_auc: float
    f1: float
    report: str
    confusion: np.ndarray


class InferenceEngine:
    """Trains and serves CatBoost fraud classifiers."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_cfg = config["model"]["catboost"]
        self.model: CatBoostClassifier | None = None

    def _build_model(self) -> CatBoostClassifier:
        return CatBoostClassifier(
            iterations=self.model_cfg.get("iterations", 500),
            learning_rate=self.model_cfg.get("learning_rate", 0.05),
            depth=self.model_cfg.get("depth", 6),
            eval_metric=self.model_cfg.get("eval_metric", "AUC"),
            auto_class_weights=self.model_cfg.get("auto_class_weights", "Balanced"),
            random_seed=self.config["data"]["random_state"],
            verbose=self.model_cfg.get("verbose", 100),
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_val: np.ndarray | None = None,
        y_val: pd.Series | None = None,
    ) -> CatBoostClassifier:
        self.model = self._build_model()
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = Pool(X_val, y_val)
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            use_best_model=eval_set is not None,
        )
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: pd.Series) -> TrainMetrics:
        proba = self.predict_proba(X)
        preds = (proba >= 0.5).astype(int)
        return TrainMetrics(
            roc_auc=roc_auc_score(y, proba),
            f1=f1_score(y, preds),
            report=classification_report(y, preds),
            confusion=confusion_matrix(y, preds),
        )

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(out))

    def load(self, path: str | Path) -> None:
        self.model = CatBoostClassifier()
        self.model.load_model(str(path))

    @staticmethod
    def feature_importance(model: CatBoostClassifier, feature_names: list[str]) -> pd.DataFrame:
        scores = model.get_feature_importance()
        return (
            pd.DataFrame({"feature": feature_names, "importance": scores})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
