"""Domain vocabulary for the fraud AI system."""

from enum import Enum


class AILayer(str, Enum):
    """Top-down AI reference architecture (highest → lowest concern)."""

    BUSINESS = "L1_business"
    EXPERIENCE = "L2_experience"
    ORCHESTRATION = "L3_orchestration"
    ML_CORE = "L4_ml_core"
    DATA = "L5_data"
    MLOPS = "L6_mlops"
    OBSERVABILITY = "L7_observability"


class PipelineStage(str, Enum):
    DATA_INGEST = "data_ingest"
    ETL = "etl"
    FEATURE_ENGINEERING = "feature_engineering"
    TRAIN = "train"
    EVALUATE = "evaluate"
    DRIFT_BASELINE = "drift_baseline"
    DRIFT_CHECK = "drift_check"
    GOVERNANCE = "governance"
    REGISTER = "register"
    ANOMALY_MODEL = "anomaly_model"


class DriftSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RetrainAction(str, Enum):
    CONTINUE = "continue"
    ALERT = "alert"
    RETRAIN_RECOMMENDED = "retrain_recommended"
    BLOCK_SERVE = "block_serve"
