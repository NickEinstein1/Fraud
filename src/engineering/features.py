"""Feature scaling and PCA for modeling and visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class EngineeringResult:
    X_train: np.ndarray
    X_test: np.ndarray
    pca_2d: np.ndarray | None


class FeatureEngineer:
    """Scales features and optionally fits PCA for 2D plots."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.eng_cfg = config["engineering"]
        self.scaler = StandardScaler()
        self.pca: PCA | None = None

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
    ) -> EngineeringResult:
        if self.eng_cfg.get("scale_features", True):
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values

        pca_2d = None
        if self.eng_cfg.get("pca", {}).get("enabled", True):
            n_comp = self.eng_cfg["pca"].get("n_components", 2)
            self.pca = PCA(n_components=n_comp, random_state=self.config["data"]["random_state"])
            pca_2d = self.pca.fit_transform(X_train_scaled)

        return EngineeringResult(
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            pca_2d=pca_2d,
        )

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.eng_cfg.get("scale_features", True):
            return self.scaler.transform(X)
        return X.values

    def save_artifacts(self, base_path: str | Path) -> None:
        base = Path(base_path)
        base.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, base / "scaler.joblib")
        if self.pca is not None:
            joblib.dump(self.pca, base / "pca.joblib")

    def load_artifacts(self, base_path: str | Path) -> None:
        base = Path(base_path)
        self.scaler = joblib.load(base / "scaler.joblib")
        pca_path = base / "pca.joblib"
        if pca_path.exists():
            self.pca = joblib.load(pca_path)

    def plot_pca(
        self,
        pca_2d: np.ndarray,
        y: pd.Series,
        output_path: str | Path,
    ) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plot_df = pd.DataFrame({"PC1": pca_2d[:, 0], "PC2": pca_2d[:, 1], "Class": y.values})
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Class", alpha=0.5, s=12)
        plt.title("PCA projection (training set)")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
