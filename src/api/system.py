"""System info for the web UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from src.api.deps import get_runtime
from src.api.schemas import get_feature_columns
from src.data.credit_dt import feature_schema_for_ui, get_featurizer
from src.data.datasets import dataset_kind, get_stream_data_path
from src.registry.model_registry import ModelRegistry
from src.utils.config import load_config

router = APIRouter(prefix="/v1/system", tags=["System"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@router.get("/info")
def system_info() -> dict[str, Any]:
    config = load_config()
    registry = ModelRegistry(config["paths"]["registry"])
    prod = registry.get_production()
    manifest_path = Path(config["paths"]["artifacts"]) / "feature_columns.json"

    defaults: dict[str, float | str] = {
        "amt": 127.16,
        "lat": 40.0,
        "long": -74.0,
        "merch_lat": 40.01,
        "merch_long": -74.01,
        "city_pop": 500000.0,
        "unix_time": 1_325_376_018.0,
        "trans_hour": 12.0,
        "trans_dow": 3.0,
        "distance_km": 1.5,
        "category": "shopping_net",
        "gender": "M",
        "state": "NY",
    }

    try:
        health = get_runtime().health().to_dict()
    except Exception as exc:
        health = {"status": "offline", "error": str(exc)}

    featurizer_meta = None
    feature_schema: list[dict[str, Any]] = []
    if dataset_kind(config) == "credit_dt":
        try:
            featurizer_meta = get_featurizer(config).to_dict()
            feature_schema = feature_schema_for_ui(config)
        except Exception:
            pass

    return {
        "dataset": dataset_kind(config),
        "feature_columns": get_feature_columns(config),
        "feature_schema": feature_schema,
        "feature_defaults": defaults,
        "scoring_pipeline": [
            "Parse raw CSV or JSON (strings for category, gender, state).",
            "Derive trans_hour / trans_dow from trans_date_trans_time when present.",
            "Compute distance_km = Haversine(cardholder lat/long, merchant merch_lat/merch_long).",
            "Label-encode categoricals with maps fit on fraudTrain (saved in artifacts).",
            "StandardScaler → CatBoost predict_proba → fraud threshold 0.5.",
        ],
        "featurizer": featurizer_meta,
        "target_column": config["data"].get("target_column", "Class"),
        "paths": {
            "stream": get_stream_data_path(config),
            "train": config.get("data", {}).get("credit_dt", {}).get("train_path"),
            "test": config.get("data", {}).get("credit_dt", {}).get("test_path"),
            "fallback": config["data"].get("path"),
        },
        "production_model": prod,
        "health": health,
        "manifest_exists": manifest_path.exists(),
    }
