"""Shared API schemas for fraud scoring."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# Default Kaggle column order (excluding Class); overridden by feature_columns.json
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]


def get_feature_columns(config: dict | None = None) -> list[str]:
    if config is None:
        from src.utils.config import load_config

        config = load_config()
    from src.data.credit_dt import load_feature_manifest

    return load_feature_manifest(config)


class ScoreRequest(BaseModel):
    """
    Transaction fields for credit_dt: use strings for category/gender/state
    (e.g. \"shopping_net\", \"M\", \"NY\"). Numeric fields are amounts/coordinates.
    Raw CSV columns (trans_date_trans_time, is_fraud) are also accepted on batch paths.
    """

    transactions: list[dict[str, Any]] = Field(..., min_length=1)


class FilePathRequest(BaseModel):
    path: str = Field(..., description="Path relative to project root, e.g. data/creditcard.csv")
    max_rows: int | None = Field(default=500, ge=1, le=100_000)


class ScoreResponse(BaseModel):
    predictions: list[dict[str, Any]]
    batch_summary: dict[str, Any]
    governance_action: str
    rows_scored: int = 0
