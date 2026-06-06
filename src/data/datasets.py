"""Dataset routing: Kaggle creditcard vs credit_dt train/test."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.credit_dt import (
    CREDIT_DT_FEATURE_COLUMNS,
    featurize_credit_dt,
    load_credit_dt_splits,
    save_feature_manifest,
)
from src.data.synthetic import ensure_dataset


def dataset_kind(config: dict[str, Any]) -> str:
    return config.get("data", {}).get("dataset", "kaggle_creditcard")


def get_stream_data_path(config: dict[str, Any]) -> str:
    """CSV path for real-time / batch replay (matches trained dataset)."""
    if dataset_kind(config) == "credit_dt":
        return config["data"]["credit_dt"]["test_path"]
    return config["data"]["path"]


def load_single_table(config: dict[str, Any]) -> pd.DataFrame:
    """Kaggle-style single CSV."""
    from pathlib import Path

    path = Path(config["data"]["path"])
    if not path.exists():
        ensure_dataset(path)
    return pd.read_csv(path)


def prepare_for_etl(config: dict[str, Any]) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Returns (full_df, None) for kaggle split-in-ETL,
    or (train_df, test_df) for credit_dt pre-split.
    """
    kind = dataset_kind(config)
    if kind == "credit_dt":
        train_df, test_df = load_credit_dt_splits(config)
        save_feature_manifest(config, CREDIT_DT_FEATURE_COLUMNS)
        return train_df, test_df
    return None, None
