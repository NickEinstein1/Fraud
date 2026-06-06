"""Feature drift detection via Kolmogorov-Smirnov tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class DriftReport:
    feature: str
    statistic: float
    p_value: float
    drift_detected: bool


class DriftMonitor:
    """Compares serving-time feature distributions to training baseline."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.drift_cfg = config["monitoring"]["drift"]
        self.alpha = self.drift_cfg.get("alpha", 0.05)
        self.baseline: dict[str, np.ndarray] | None = None

    def fit_baseline(self, X: pd.DataFrame) -> None:
        self.baseline = {col: X[col].dropna().values for col in X.columns}

    def save_baseline(self, path: str | Path) -> None:
        if self.baseline is None:
            raise RuntimeError("Baseline not fitted.")
        joblib.dump(self.baseline, path)

    def load_baseline(self, path: str | Path) -> None:
        self.baseline = joblib.load(path)

    def check(
        self,
        X_serving: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> list[DriftReport]:
        if self.baseline is None:
            raise RuntimeError("Baseline not fitted or loaded.")

        cols = feature_names or list(self.baseline.keys())
        min_samples = self.drift_cfg.get("min_samples", 100)
        reports: list[DriftReport] = []

        for col in cols:
            if col not in X_serving.columns or col not in self.baseline:
                continue
            ref = self.baseline[col]
            cur = X_serving[col].dropna().values
            if len(cur) < min_samples:
                continue
            stat, p_value = ks_2samp(ref, cur)
            reports.append(
                DriftReport(
                    feature=col,
                    statistic=float(stat),
                    p_value=float(p_value),
                    drift_detected=p_value < self.alpha,
                )
            )
        return reports

    def summary(self, reports: list[DriftReport]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "feature": r.feature,
                    "ks_statistic": r.statistic,
                    "p_value": r.p_value,
                    "drift_detected": r.drift_detected,
                }
                for r in reports
            ]
        ).sort_values("p_value")

    def drifted_features(self, reports: list[DriftReport]) -> list[str]:
        return [r.feature for r in reports if r.drift_detected]
