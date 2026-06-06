"""Secondary AI model: Isolation Forest for unsupervised anomaly signal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Complements CatBoost with anomaly scores in real-time streams."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        rt = config.get("realtime", {}).get("anomaly", {})
        self.contamination = rt.get("contamination", 0.002)
        self.n_estimators = rt.get("n_estimators", 100)
        self.model: IsolationForest | None = None

    def fit(self, X: np.ndarray) -> IsolationForest:
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.config["data"]["random_state"],
            n_jobs=-1,
        )
        self.model.fit(X)
        return self.model

    def fit_sample_csv(self, path: Path, feature_cols: list[str], max_rows: int = 8000) -> None:
        from src.data.credit_dt import prepare_scoring_frame
        from src.utils.config import load_config

        df = pd.read_csv(path, nrows=max_rows)
        df = prepare_scoring_frame(df, load_config())
        cols = [c for c in feature_cols if c in df.columns]
        X = df[cols].values
        self.fit(X)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Anomaly model not fitted.")
        # Higher = more anomalous (invert sklearn decision_function)
        raw = self.model.decision_function(X)
        return -raw

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("No anomaly model to save.")
        joblib.dump(self.model, path)

    def load(self, path: str | Path) -> None:
        self.model = joblib.load(path)
