"""ETL: load, clean, impute, and balance fraud data with SMOTE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from src.data.credit_dt import save_feature_manifest
from src.data.datasets import dataset_kind, load_single_table, prepare_for_etl


@dataclass
class ETLResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    X_train_raw: pd.DataFrame  # pre-SMOTE; use as drift monitoring baseline


class ETLPipeline:
    """Handles ingestion, cleaning, and class balancing."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.etl_cfg = config["etl"]
        self.data_cfg = config["data"]

    def load(self, path: str | None = None) -> pd.DataFrame:
        if dataset_kind(self.config) == "credit_dt":
            from src.data.credit_dt import load_credit_dt_splits

            train_df, _ = load_credit_dt_splits(self.config)
            return train_df
        path = path or self.data_cfg["path"]
        return load_single_table(self.config) if path == self.data_cfg["path"] else pd.read_csv(path)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.etl_cfg.get("drop_duplicates", True):
            out = out.drop_duplicates()
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if out[col].isna().any():
                out[col] = out[col].fillna(out[col].median())
        return out

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        target = self.data_cfg["target_column"]
        feature_cols = [c for c in df.columns if c != target]
        X = df[feature_cols]
        y = df[target]
        return train_test_split(
            X,
            y,
            test_size=self.data_cfg["test_size"],
            random_state=self.data_cfg["random_state"],
            stratify=y,
        )

    def apply_smote(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        smote_cfg = self.etl_cfg.get("smote", {})
        if not smote_cfg.get("enabled", True):
            return X, y

        smote = SMOTE(
            sampling_strategy=smote_cfg.get("sampling_strategy", "auto"),
            k_neighbors=smote_cfg.get("k_neighbors", 5),
            random_state=self.data_cfg["random_state"],
        )
        X_res, y_res = smote.fit_resample(X, y)
        return (
            pd.DataFrame(X_res, columns=X.columns),
            pd.Series(y_res, name=y.name),
        )

    def _xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        target = self.data_cfg["target_column"]
        feature_cols = [c for c in df.columns if c != target]
        return df[feature_cols], df[target]

    def run(self, df: pd.DataFrame | None = None) -> ETLResult:
        if df is not None:
            cleaned = self.clean(df)
            X_train, X_test, y_train, y_test = self.split(cleaned)
        elif dataset_kind(self.config) == "credit_dt":
            train_df, test_df = prepare_for_etl(self.config)
            assert train_df is not None and test_df is not None
            train_clean = self.clean(train_df)
            test_clean = self.clean(test_df)
            X_train, y_train = self._xy(train_clean)
            X_test, y_test = self._xy(test_clean)
        else:
            raw = self.load()
            cleaned = self.clean(raw)
            X_train, X_test, y_train, y_test = self.split(cleaned)

        X_train_bal, y_train_bal = self.apply_smote(X_train, y_train)
        save_feature_manifest(self.config, list(X_train.columns))
        return ETLResult(
            X_train=X_train_bal,
            X_test=X_test,
            y_train=y_train_bal,
            y_test=y_test,
            feature_names=list(X_train.columns),
            X_train_raw=X_train,
        )
