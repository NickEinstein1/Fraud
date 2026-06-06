"""Data helpers for the UI (sample CSV rows for manual scoring)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.data.credit_dt import row_for_manual_form

router = APIRouter(prefix="/v1/data", tags=["Data"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_data_path(path: str) -> Path:
    candidate = (PROJECT_ROOT / path).resolve()
    root = PROJECT_ROOT.resolve()
    if not str(candidate).startswith(str(root)):
        raise HTTPException(status_code=400, detail="Path must stay inside the project directory")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    if candidate.suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    return candidate


@router.get("/csv-row")
def get_csv_row(
    path: str = Query(..., description="CSV path relative to project root"),
    row_index: int = Query(0, ge=0, description="0-based data row (after header)"),
) -> dict:
    """
    Return one raw CSV row as manual-entry fields.
    Use this to score the same transaction as in fraudTest.csv with matching probability.
    """
    csv_path = _resolve_data_path(path)
    try:
        row = pd.read_csv(csv_path, skiprows=range(1, row_index + 1), nrows=1).iloc[0]
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=f"Row {row_index} not found in {path}") from exc

    payload = row_for_manual_form(row)
    payload["row_index"] = row_index
    payload["path"] = path
    payload["note"] = (
        "is_fraud_actual is the dataset label (ground truth), not the model score. "
        "Submit these fields via Score transaction to reproduce batch scoring for this row."
    )
    return payload
