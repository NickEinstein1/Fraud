"""Shared helpers for the Streamlit scoring UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.api.schemas import get_feature_columns
from src.serving.runtime import FraudServingRuntime
from src.utils.config import load_config


def feature_columns():
    return get_feature_columns()


def get_runtime() -> FraudServingRuntime:
    runtime = FraudServingRuntime(load_config())
    runtime.load()
    return runtime


def score_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], str]:
    from src.data.credit_dt import prepare_scoring_frame

    config = load_config()
    df = prepare_scoring_frame(df, config)
    runtime = get_runtime()
    cols = feature_columns()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out, summary = runtime.score_batch(df[cols])
    health = runtime.health()
    result = out[cols + ["fraud_probability", "fraud_prediction"]].copy()
    return result, summary.to_dict(), health.governance_action


def load_csv(path: Path, max_rows: int | None = 500) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, nrows=max_rows)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
