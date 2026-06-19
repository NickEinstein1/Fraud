"""Scoring routes: manual JSON, file path, and file upload."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.schemas import FilePathRequest, ScoreRequest, ScoreResponse, get_feature_columns
from src.serving.runtime import FraudServingRuntime
from src.utils.paths import resolve_csv_path

router = APIRouter(prefix="/v1/score", tags=["Scoring"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
_runtime: FraudServingRuntime | None = None


def get_runtime() -> FraudServingRuntime:
    global _runtime
    if _runtime is None:
        _runtime = FraudServingRuntime()
        _runtime.load()
    return _runtime


def _build_response(
    out: pd.DataFrame,
    summary_dict: dict[str, Any],
    governance: str,
    *,
    features_used: list[dict[str, Any]] | None = None,
    ground_truth: list[int] | None = None,
) -> ScoreResponse:
    preds = out[["fraud_probability", "fraud_prediction"]].to_dict(orient="records")
    if ground_truth:
        for i, actual in enumerate(ground_truth):
            if i >= len(preds):
                break
            preds[i]["is_fraud_actual"] = int(actual)
            preds[i]["model_agrees_with_label"] = int(actual) == preds[i]["fraud_prediction"]
        summary_dict["has_ground_truth"] = True
        summary_dict["label_note"] = (
            "is_fraud_actual is the CSV truth label (1 = known fraud). "
            "fraud_prediction is the model output — they often differ."
        )
        actual_fraud = sum(ground_truth)
        model_hits = sum(
            1 for i, a in enumerate(ground_truth) if i < len(preds) and preds[i]["fraud_prediction"] == 1
        )
        correct_on_fraud = sum(
            1
            for i, a in enumerate(ground_truth)
            if a == 1 and i < len(preds) and preds[i]["fraud_prediction"] == 1
        )
        summary_dict["actual_fraud_in_batch"] = actual_fraud
        summary_dict["model_flagged_count"] = model_hits
        if actual_fraud:
            summary_dict["recall_on_labeled_fraud"] = round(correct_on_fraud / actual_fraud, 4)
    if features_used:
        summary_dict["features_used_sample"] = features_used
    return ScoreResponse(
        predictions=preds,
        batch_summary=summary_dict,
        governance_action=governance,
        rows_scored=len(out),
    )


def _score_dataframe(df: pd.DataFrame) -> ScoreResponse:
    from src.data.credit_dt import extract_ground_truth, prepare_scoring_frame

    runtime = get_runtime()
    ground_truth = extract_ground_truth(df)
    df = prepare_scoring_frame(df, runtime.config)
    feature_cols = get_feature_columns(runtime.config)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {missing}. Required: {feature_cols}",
        )
    feature_df = df[feature_cols].copy()
    try:
        out, summary = runtime.score_batch(feature_df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    health = runtime.health()
    summary_dict = summary.to_dict()
    summary_dict["drifted_features"] = summary.drifted_features
    sample = feature_df.head(1).round(6).to_dict(orient="records")
    return _build_response(
        out,
        summary_dict,
        health.governance_action,
        features_used=sample,
        ground_truth=ground_truth,
    )


@router.post("", response_model=ScoreResponse)
def score_transactions(request: ScoreRequest) -> ScoreResponse:
    """Score one or more transactions from JSON (manual entry / API clients)."""
    return _score_dataframe(pd.DataFrame(request.transactions))


@router.post("/from-path", response_model=ScoreResponse)
def score_from_path(body: FilePathRequest) -> dict[str, Any]:
    """Score transactions from a CSV path on the server (e.g. data/creditcard.csv)."""
    path = resolve_csv_path(body.path)
    df = pd.read_csv(path, nrows=body.max_rows)
    response = _score_dataframe(df)
    payload = response.model_dump()
    payload["source"] = {"type": "path", "path": body.path, "rows_read": len(df)}
    return payload


@router.post("/from-upload", response_model=ScoreResponse)
async def score_from_upload(
    file: UploadFile = File(...),
    max_rows: int = 500,
) -> dict[str, Any]:
    """Score transactions from an uploaded CSV file."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file")
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)",
        )
    try:
        df = pd.read_csv(io.BytesIO(content), nrows=max_rows)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc
    response = _score_dataframe(df)
    payload = response.model_dump()
    payload["source"] = {"type": "upload", "filename": file.filename, "rows_read": len(df)}
    return payload
