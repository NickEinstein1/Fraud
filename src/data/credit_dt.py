"""Load and featurize fraudTrain / fraudTest from data/credit_dt/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Columns fed to StandardScaler + CatBoost (saved to artifacts/feature_columns.json)
CREDIT_DT_FEATURE_COLUMNS = [
    "amt",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "city_pop",
    "unix_time",
    "trans_hour",
    "trans_dow",
    "distance_km",
    "category",
    "gender",
    "state",
]

CATEGORICAL_COLUMNS = ("category", "gender", "state")

# Filled from training medians when manual scoring omits location
GEO_COLUMNS = ("lat", "long", "merch_lat", "merch_long")
NUMERIC_DEFAULT_COLUMNS = (
    "amt",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "city_pop",
    "unix_time",
    "trans_hour",
    "trans_dow",
)

PII_AND_DROP = [
    "Unnamed: 0",
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "first",
    "last",
    "street",
    "city",
    "zip",
    "job",
    "dob",
    "trans_num",
    "is_fraud",
]

FEATURIZER_ARTIFACT = "credit_dt_featurizer.json"

FEATURE_HELP: dict[str, dict[str, str]] = {
    "amt": {
        "label": "Transaction amount ($)",
        "role": "Numeric input to CatBoost after standardization.",
    },
    "lat": {
        "label": "Cardholder latitude",
        "role": "Used with long + merchant coordinates to compute distance_km (Haversine).",
    },
    "long": {
        "label": "Cardholder longitude",
        "role": "Used with lat + merchant coordinates to compute distance_km (Haversine).",
    },
    "merch_lat": {
        "label": "Merchant latitude",
        "role": "Merchant location; combined into distance_km and passed as a direct model feature.",
    },
    "merch_long": {
        "label": "Merchant longitude",
        "role": "Merchant location; combined into distance_km and passed as a direct model feature.",
    },
    "city_pop": {
        "label": "City population",
        "role": "Numeric context for transaction locale.",
    },
    "unix_time": {
        "label": "Unix timestamp",
        "role": "Transaction time as seconds since epoch.",
    },
    "trans_hour": {
        "label": "Hour of day (0–23)",
        "role": "Derived from trans_date_trans_time when scoring raw CSVs; otherwise supplied.",
    },
    "trans_dow": {
        "label": "Day of week (0=Mon)",
        "role": "Derived from trans_date_trans_time when scoring raw CSVs; otherwise supplied.",
    },
    "distance_km": {
        "label": "Cardholder ↔ merchant distance (km)",
        "role": "Computed: Haversine(lat, long, merch_lat, merch_long). Flags geographically implausible purchases.",
    },
    "category": {
        "label": "Merchant category",
        "role": "Categorical (e.g. shopping_net, travel); label-encoded with training vocabulary.",
    },
    "gender": {
        "label": "Cardholder gender",
        "role": "Categorical F/M; label-encoded with training vocabulary.",
    },
    "state": {
        "label": "US state code",
        "role": "Categorical (e.g. NY, CA); label-encoded with training vocabulary.",
    },
}


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in km between (lat1,lon1) and (lat2,lon2)."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def compute_distance_km(df: pd.DataFrame) -> np.ndarray:
    return _haversine_km(
        pd.to_numeric(df["lat"], errors="coerce").values,
        pd.to_numeric(df["long"], errors="coerce").values,
        pd.to_numeric(df["merch_lat"], errors="coerce").values,
        pd.to_numeric(df["merch_long"], errors="coerce").values,
    )


class CreditDtFeaturizer:
    """
    Deterministic featurization: same math at train and score time.
    Categoricals use fixed label maps fit on training data (not per-batch pandas codes).
    """

    def __init__(
        self,
        categorical_maps: dict[str, dict[str, int]] | None = None,
        numeric_defaults: dict[str, float] | None = None,
    ):
        self.categorical_maps: dict[str, dict[str, int]] = categorical_maps or {}
        self.numeric_defaults: dict[str, float] = numeric_defaults or {}

    def fit(self, raw_df: pd.DataFrame) -> CreditDtFeaturizer:
        defaults: dict[str, float] = {}
        for col in NUMERIC_DEFAULT_COLUMNS:
            if col in raw_df.columns:
                defaults[col] = float(pd.to_numeric(raw_df[col], errors="coerce").median())
        self.numeric_defaults = defaults

        maps: dict[str, dict[str, int]] = {}
        for col in CATEGORICAL_COLUMNS:
            if col not in raw_df.columns:
                continue
            raw_vals = sorted(raw_df[col].astype(str).str.strip().unique())
            maps[col] = {v: i for i, v in enumerate(raw_vals)}
            maps[col]["__unknown__"] = len(raw_vals)
        self.categorical_maps = maps
        return self

    def encode_label(self, col: str, value: Any) -> float:
        if col not in self.categorical_maps:
            return float(pd.to_numeric(value, errors="coerce") or 0.0)
        m = self.categorical_maps[col]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return float(m["__unknown__"])
        s = str(value).strip()
        if s in m:
            return float(m[s])
        # Case-insensitive match for gender/state
        for k, code in m.items():
            if k != "__unknown__" and k.upper() == s.upper():
                return float(code)
        return float(m["__unknown__"])

    def _apply_numeric_defaults(self, out: pd.DataFrame) -> pd.DataFrame:
        """Use training medians for missing manual-entry fields (especially geo)."""
        for col in NUMERIC_DEFAULT_COLUMNS:
            default = self.numeric_defaults.get(col, 0.0)
            if col not in out.columns:
                out[col] = default
            else:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)
        if all(c in out.columns for c in GEO_COLUMNS):
            out["distance_km"] = compute_distance_km(out)
        return out

    def transform_raw(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline from fraudTrain/fraudTest CSV columns."""
        out = raw_df.copy()
        if "is_fraud" in out.columns:
            out["Class"] = pd.to_numeric(out["is_fraud"], errors="coerce").fillna(0).astype(int)

        ts = pd.to_datetime(out["trans_date_trans_time"], errors="coerce")
        out["trans_hour"] = ts.dt.hour.fillna(0).astype(int)
        out["trans_dow"] = ts.dt.dayofweek.fillna(0).astype(int)

        for col in ("amt", "lat", "long", "merch_lat", "merch_long", "city_pop", "unix_time"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        out["distance_km"] = compute_distance_km(out)

        for col in CATEGORICAL_COLUMNS:
            if col in out.columns:
                out[col] = out[col].apply(lambda v: self.encode_label(col, v))

        cols = CREDIT_DT_FEATURE_COLUMNS + (["Class"] if "Class" in out.columns else [])
        return out[[c for c in cols if c in out.columns]]

    def transform_scoring_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score-time transform: raw CSV, partial manual rows, or pre-numeric rows.
        Recomputes distance_km and re-encodes string categoricals when needed.
        """
        if is_raw_credit_dt(df):
            return self.transform_raw(df)

        out = df.copy()

        if "trans_date_trans_time" in out.columns:
            ts = pd.to_datetime(out["trans_date_trans_time"], errors="coerce")
            out["trans_hour"] = ts.dt.hour.fillna(out.get("trans_hour", 0)).astype(int)
            out["trans_dow"] = ts.dt.dayofweek.fillna(out.get("trans_dow", 0)).astype(int)

        out = self._apply_numeric_defaults(out)

        for col in CATEGORICAL_COLUMNS:
            if col not in out.columns:
                continue
            series = out[col]
            if series.dtype == object or series.map(_looks_like_text_label).any():
                out[col] = series.apply(lambda v: self.encode_label(col, v))
            else:
                # Already numeric codes from a prior featurization pass
                out[col] = pd.to_numeric(series, errors="coerce")

        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "categorical_maps": self.categorical_maps,
            "numeric_defaults": self.numeric_defaults,
            "feature_columns": CREDIT_DT_FEATURE_COLUMNS,
            "derived_features": {
                "distance_km": {
                    "formula": "2 * R * arcsin(sqrt(sin²(Δlat/2) + cos(lat1)*cos(lat2)*sin²(Δlon/2))), R=6371 km",
                    "inputs": ["lat", "long", "merch_lat", "merch_long"],
                },
                "trans_hour": {"formula": "hour(trans_date_trans_time)", "inputs": ["trans_date_trans_time"]},
                "trans_dow": {"formula": "dayofweek(trans_date_trans_time)", "inputs": ["trans_date_trans_time"]},
            },
            "feature_help": FEATURE_HELP,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CreditDtFeaturizer:
        return cls(
            categorical_maps=payload.get("categorical_maps", {}),
            numeric_defaults=payload.get("numeric_defaults", {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> CreditDtFeaturizer | None:
        if not path.exists():
            return None
        return cls.from_dict(json.loads(path.read_text()))


def _looks_like_text_label(v: Any) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    s = str(v).strip()
    if not s:
        return False
    try:
        float(s)
        return False
    except ValueError:
        return True


_featurizer_cache: CreditDtFeaturizer | None = None


def featurizer_artifact_path(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["artifacts"]) / FEATURIZER_ARTIFACT


def fit_featurizer_from_train(config: dict[str, Any]) -> CreditDtFeaturizer:
    cfg = config["data"]["credit_dt"]
    train_path = Path(cfg["train_path"])
    max_rows = cfg.get("max_train_rows")
    raw = pd.read_csv(train_path, nrows=max_rows)
    featurizer = CreditDtFeaturizer().fit(raw)
    featurizer.save(featurizer_artifact_path(config))
    return featurizer


def get_featurizer(config: dict[str, Any], *, refit: bool = False) -> CreditDtFeaturizer:
    global _featurizer_cache
    path = featurizer_artifact_path(config)
    if not refit:
        if _featurizer_cache is not None:
            return _featurizer_cache
        loaded = CreditDtFeaturizer.load(path)
        if loaded is not None and loaded.categorical_maps and loaded.numeric_defaults:
            _featurizer_cache = loaded
            return loaded
    featurizer = fit_featurizer_from_train(config)
    _featurizer_cache = featurizer
    return featurizer


def featurize_credit_dt(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Transform raw credit_dt rows into model-ready numeric features + optional Class."""
    if config is None:
        from src.utils.config import load_config

        config = load_config()
    featurizer = get_featurizer(config)
    return featurizer.transform_raw(df)


def load_credit_dt_splits(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load fraudTrain.csv and fraudTest.csv with optional row caps."""
    cfg = config["data"]["credit_dt"]
    train_path = Path(cfg["train_path"])
    test_path = Path(cfg["test_path"])

    max_train = cfg.get("max_train_rows")
    max_test = cfg.get("max_test_rows")

    train_raw = pd.read_csv(train_path, nrows=max_train)
    test_raw = pd.read_csv(test_path, nrows=max_test)

    featurizer = get_featurizer(config, refit=True)
    return featurizer.transform_raw(train_raw), featurizer.transform_raw(test_raw)


def save_feature_manifest(config: dict[str, Any], columns: list[str] | None = None) -> Path:
    cols = columns or CREDIT_DT_FEATURE_COLUMNS
    path = Path(config["paths"]["artifacts"]) / "feature_columns.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": config["data"].get("dataset", "credit_dt"),
        "feature_columns": cols,
        "target_column": config["data"].get("target_column", "Class"),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def is_raw_credit_dt(df: pd.DataFrame) -> bool:
    return "is_fraud" in df.columns and "amt" in df.columns and "merchant" in df.columns


def prepare_scoring_frame(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Featurize raw or manual credit_dt rows before scoring (stable encodings + distance)."""
    if config is None:
        from src.utils.config import load_config

        config = load_config()
    if dataset_kind_from_config(config) != "credit_dt":
        return df
    featurizer = get_featurizer(config)
    out = featurizer.transform_scoring_input(df)
    feature_cols = CREDIT_DT_FEATURE_COLUMNS
    for col in feature_cols:
        if col not in out.columns:
            out[col] = featurizer.numeric_defaults.get(col, 0.0)
    return out[feature_cols + [c for c in out.columns if c == "Class"]]


def extract_ground_truth(df: pd.DataFrame) -> list[int] | None:
    """Actual fraud labels from CSV (is_fraud / Class), not model output."""
    if "is_fraud" in df.columns:
        return pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int).tolist()
    if "Class" in df.columns:
        return pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int).tolist()
    return None


def row_for_manual_form(raw_row: pd.Series) -> dict[str, Any]:
    """Map a raw credit_dt CSV row to manual-entry field values."""
    out: dict[str, Any] = {}
    for col in (
        "amt",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "city_pop",
        "unix_time",
        "category",
        "gender",
        "state",
    ):
        if col in raw_row.index and pd.notna(raw_row[col]):
            out[col] = raw_row[col]
    if "trans_date_trans_time" in raw_row.index:
        ts = pd.to_datetime(raw_row["trans_date_trans_time"], errors="coerce")
        if pd.notna(ts):
            out["trans_hour"] = int(ts.hour)
            out["trans_dow"] = int(ts.dayofweek)
    if "is_fraud" in raw_row.index:
        out["is_fraud_actual"] = int(raw_row["is_fraud"])
    return out


def dataset_kind_from_config(config: dict[str, Any]) -> str:
    return config.get("data", {}).get("dataset", "kaggle_creditcard")


def load_feature_manifest(config: dict[str, Any]) -> list[str]:
    path = Path(config["paths"]["artifacts"]) / "feature_columns.json"
    if path.exists():
        return json.loads(path.read_text())["feature_columns"]
    from src.api.schemas import FEATURE_COLUMNS

    return list(FEATURE_COLUMNS)


def feature_schema_for_ui(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Field metadata for the Sentinel manual form."""
    featurizer = get_featurizer(config)
    schema: list[dict[str, Any]] = []
    for col in CREDIT_DT_FEATURE_COLUMNS:
        help_meta = FEATURE_HELP.get(col, {})
        entry: dict[str, Any] = {
            "name": col,
            "type": "number",
            "label": help_meta.get("label", col),
            "description": help_meta.get("role", ""),
        }
        if col in CATEGORICAL_COLUMNS and col in featurizer.categorical_maps:
            labels = [k for k in featurizer.categorical_maps[col] if k != "__unknown__"]
            entry["type"] = "categorical"
            entry["options"] = sorted(labels)
        if col in GEO_COLUMNS:
            entry["optional"] = True
            entry["group"] = "location"
            entry["description"] = (
                "Optional. Leave blank to use typical training values; "
                "for CSV rows, import a row to copy real coordinates."
            )
        if col == "distance_km":
            entry["computed"] = True
            entry["hidden"] = True
            entry["description"] = "Auto-calculated from coordinates (Haversine km)."
        schema.append(entry)
    return schema
