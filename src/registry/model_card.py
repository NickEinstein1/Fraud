"""Model cards (L6 MLOps) — documentation for each trained AI model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ModelCardContext:
    version: str
    run_id: str | None
    metrics: dict[str, float]
    drift_alerts: list[dict[str, Any]]
    governance_action: str
    config: dict[str, Any]
    feature_importance_path: Path | None = None


def _utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class ModelCardWriter:
    """Generates Markdown model cards aligned with Google/Hugging Face model card practice."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_catboost(self, ctx: ModelCardContext) -> Path:
        path = self.output_dir / f"MODEL_CARD_catboost_{ctx.version}.md"
        cfg = ctx.config
        etl = cfg.get("etl", {})
        model = cfg.get("model", {}).get("catboost", {})
        gov = cfg.get("governance", {})

        top_features = ""
        if ctx.feature_importance_path and ctx.feature_importance_path.exists():
            import pandas as pd

            imp = pd.read_csv(ctx.feature_importance_path).head(10)
            top_features = "\n".join(
                f"| {r['feature']} | {r['importance']:.4f} |"
                for _, r in imp.iterrows()
            )

        drift_section = (
            "No significant hold-out drift flagged at train time."
            if not ctx.drift_alerts
            else "\n".join(
                f"- **{a.get('feature')}**: p={a.get('p_value', 0):.4f}, severity={a.get('severity')}"
                for a in ctx.drift_alerts[:10]
            )
        )

        body = f"""# Model Card: Fraud CatBoost Classifier

**Model ID:** `fraud_catboost`  
**Version:** `{ctx.version}`  
**Date:** {_utc_date()}  
**Run ID:** `{ctx.run_id or "n/a"}`  
**Status:** Production candidate — governance `{ctx.governance_action}`

---

## Model Details

| Field | Value |
|-------|-------|
| **Type** | Gradient boosted decision trees (binary classification) |
| **Library** | [CatBoost](https://catboost.ai/) |
| **Artifact** | `artifacts/models/catboost_fraud.cbm` |
| **Decision threshold** | 0.5 (configurable via `realtime.fraud_threshold`) |

### Intended Use

- **Primary:** Score credit-card–style transactions for fraud probability in batch or real-time streams.
- **Users:** Risk analysts, fraud ops, and automated pipelines via REST API or Streamlit UI.
- **Out of scope:** Not for legal identity verification, credit underwriting, or non–fraud-detection classification without retraining.

### Factors

- **Relevant:** Tabular PCA features (V1–V28), Time, Amount; severe class imbalance (~0.17% fraud on Kaggle ULB data).
- **Not relevant:** Raw merchant names, geolocation text, or categorical fields not in the training schema.

---

## Training Data

| Field | Value |
|-------|-------|
| **Dataset** | `{cfg.get("data", {}).get("path", "data/creditcard.csv")}` |
| **Target** | `{cfg.get("data", {}).get("target_column", "Class")}` (0=legit, 1=fraud) |
| **Split** | Stratified hold-out, `test_size={cfg.get("data", {}).get("test_size", 0.2)}` |
| **Balancing** | SMOTE on train only (`enabled={etl.get("smote", {}).get("enabled", True)}`) |
| **Features** | V1–V28, Time, Amount (30 numeric inputs after engineering) |

**Preprocessing:** Duplicate removal, median imputation, `StandardScaler` fit on train, persisted to `artifacts/scaler.joblib`.

---

## Training Procedure

| Hyperparameter | Value |
|----------------|-------|
| iterations | {model.get("iterations", 500)} |
| learning_rate | {model.get("learning_rate", 0.05)} |
| depth | {model.get("depth", 6)} |
| eval_metric | {model.get("eval_metric", "AUC")} |
| auto_class_weights | {model.get("auto_class_weights", "Balanced")} |

**Why CatBoost vs Random Forest:** Ordered boosting handles imbalance; strong AUC on tabular fraud; minimal categorical preprocessing if PaySim-style fields are added later.

---

## Evaluation

| Metric | Hold-out test |
|--------|----------------|
| **ROC-AUC** | {ctx.metrics.get("roc_auc", 0):.4f} |
| **F1** | {ctx.metrics.get("f1", 0):.4f} |

*Metrics reflect the dataset used at train time. Synthetic data may show near-perfect scores; re-evaluate on real Kaggle ULB data for reporting.*

### Top feature importance (train)

| Feature | Importance |
|---------|------------|
{top_features or "| — | Run training to generate `feature_importance.csv` |"}

---

## Drift & Monitoring

- **Method:** Kolmogorov–Smirnov two-sample tests vs pre-SMOTE training baseline.
- **Config:** `alpha={cfg.get("monitoring", {}).get("drift", {}).get("alpha", 0.05)}`, `min_samples={cfg.get("monitoring", {}).get("drift", {}).get("min_samples", 100)}`
- **Hold-out drift at train time:**

{drift_section}

---

## Governance (L1)

| Rule | Threshold |
|------|-----------|
| Min ROC-AUC | {gov.get("min_roc_auc", 0.85)} |
| Alert if drifted features ≥ | {gov.get("drift_features_alert", 3)} |
| Block serve if drifted features ≥ | {gov.get("drift_features_block", 10)} |

**Train-time decision:** `{ctx.governance_action}`

---

## Ethical Considerations & Limitations

- **False positives** block or flag legitimate customers — tune threshold for business cost.
- **Adversarial drift:** Fraudsters adapt; models decay without retraining and KS monitoring.
- **SMOTE artifacts:** Synthetic minority points in high-D PCA space may not reflect real fraud geometry.
- **Bias:** Training data may under-represent regions or card types not present in the source CSV.

---

## How to Use

```bash
# Batch / UI
python main.py --ui
python main.py --serve data/creditcard.csv

# API
POST /v1/score
```

---

## Changelog

| Version | Date | Notes |
|---------|------|-------|
| {ctx.version} | {_utc_date()} | Registered via FraudAIPipeline |
"""
        path.write_text(body.strip() + "\n")
        return path

    def write_anomaly(self, ctx: ModelCardContext) -> Path:
        path = self.output_dir / f"MODEL_CARD_anomaly_{ctx.version}.md"
        cfg = ctx.config
        rt = cfg.get("realtime", {}).get("anomaly", {})

        body = f"""# Model Card: Isolation Forest Anomaly Detector

**Model ID:** `fraud_anomaly_iforest`  
**Version:** `{ctx.version}` (paired with CatBoost `{ctx.version}`)  
**Date:** {_utc_date()}  
**Run ID:** `{ctx.run_id or "n/a"}`

---

## Model Details

| Field | Value |
|-------|-------|
| **Type** | Unsupervised anomaly detection |
| **Library** | scikit-learn `IsolationForest` |
| **Artifact** | `artifacts/models/anomaly_iforest.joblib` |

### Intended Use

- **Primary:** Secondary signal in **real-time** streams (`RealtimeFraudAIEngine`) — higher scores suggest novel/outlier transactions.
- **Not standalone:** Do not use alone for fraud decisions; combine with CatBoost and governance rules.

---

## Training Data

Same feature schema as CatBoost: V1–V28, Time, Amount.  
Trained on up to 8,000 rows from `{cfg.get("data", {}).get("path")}` (legit-heavy distribution).

| Hyperparameter | Value |
|----------------|-------|
| n_estimators | {rt.get("n_estimators", 100)} |
| contamination | {rt.get("contamination", 0.002)} |

---

## Evaluation

No supervised metrics (unsupervised). Monitor score distribution shift alongside KS drift on features.

---

## Limitations

- Assumes anomalies ≈ fraud; many outliers may be benign rare behavior.
- Contamination hyperparameter should match expected fraud rate order-of-magnitude.

---

## How to Use

```bash
python main.py --realtime
python main.py --dashboard
```
"""
        path.write_text(body.strip() + "\n")
        return path

    def write_index(self, version: str, card_paths: list[Path]) -> Path:
        index = self.output_dir / "README.md"
        lines = [
            "# Model Cards",
            "",
            f"**Production version:** `{version}`",
            "",
            "## Cards",
            "",
        ]
        for p in card_paths:
            lines.append(f"- [{p.name}]({p.name})")
        lines.extend(
            [
                "",
                "Regenerate after training: `python main.py --model-cards`",
                "",
            ]
        )
        index.write_text("\n".join(lines))
        return index


def generate_model_cards(
    config: dict[str, Any],
    version: str,
    metrics: dict[str, float],
    drift_alerts: list[dict[str, Any]],
    governance_action: str,
    run_id: str | None = None,
) -> list[Path]:
    """Write model cards to registry version dir and global model_cards folder."""
    registry_dir = Path(config["paths"]["registry"]) / version
    global_dir = Path(config["paths"]["artifacts"]) / "model_cards"
    registry_dir.mkdir(parents=True, exist_ok=True)

    imp_path = Path(config["paths"]["artifacts"]) / "feature_importance.csv"
    ctx = ModelCardContext(
        version=version,
        run_id=run_id,
        metrics=metrics,
        drift_alerts=drift_alerts,
        governance_action=governance_action,
        config=config,
        feature_importance_path=imp_path if imp_path.exists() else None,
    )

    writer_reg = ModelCardWriter(registry_dir)
    writer_global = ModelCardWriter(global_dir)

    paths: list[Path] = []
    for writer in (writer_reg, writer_global):
        paths.append(writer.write_catboost(ctx))
        paths.append(writer.write_anomaly(ctx))
        writer.write_index(version, [p for p in paths if p.parent == writer.output_dir])

    return paths
