# Adversarial Fraud Detection — AI World Platform

A **top-down AI system** for credit-card fraud: seven architectural layers from business policy to observability, with CatBoost + SMOTE + KS drift at the ML core.

## AI World Architecture (Top → Down)

| Layer | Name | This project |
| ----- | ---- | ------------ |
| **L1** | Business & Risk | Governance: alert / retrain / block serve |
| **L2** | Experience | FastAPI + batch CLI |
| **L3** | Orchestration | `FraudAIPipeline` stage graph |
| **L4** | ML Core | ETL · Engineering · CatBoost · KS drift |
| **L5** | Data Plane | `credit_dt` train/test or Kaggle CSV |
| **L6** | MLOps | Model registry & promotion |
| **L7** | Observability | Per-run telemetry & manifests |

Full design: **[docs/AI_WORLD_ARCHITECTURE.md](docs/AI_WORLD_ARCHITECTURE.md)** · ML detail: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

```text
                    ┌──────────────────┐
                    │ L1 Governance    │
                    └────────┬─────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   ┌──────────┐        ┌─────────────┐      ┌────────────┐
   │ L2 API   │        │ L3 Pipeline │      │ L7 Telemetry│
   └────┬─────┘        └──────┬──────┘      └────────────┘
        │                     │
        │              ┌──────┴──────┐
        │              ▼             ▼
        │         ┌─────────┐  ┌──────────┐
        └────────►│ L4 ML   │  │ L6 Registry│
                  │ Core    │  └──────────┘
                  └────┬────┘
                       ▼
                  ┌─────────┐
                  │ L5 Data │
                  └─────────┘
```

## Problem & ML Approach

| Challenge | Response |
| --------- | -------- |
| **Adversarial drift** | KS-tests + L1 governance actions |
| **~1000:1 imbalance** | SMOTE (train only) + CatBoost balanced weights |
| **Tabular fraud features** | Scaling; optional 2D PCA for visualization |

**CatBoost vs Random Forest:** Boosting corrects prior tree errors; native categorical handling; `auto_class_weights` for fraud skew. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick Start

```bash
cd "/Users/einstein/Documents/DATA SCIENCE/Final_Project"
source .venv/bin/activate   # or: python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# View the seven-layer stack
python main.py --architecture

# Train end-to-end (orchestrated) — uses data/credit_dt by default
python main.py

# Batch serve + drift + governance
python main.py --serve data/credit_dt/fraudTest.csv

# REST API (L2)
python main.py --api
# Docs: http://127.0.0.1:8000/docs

# Real-time AI — CatBoost + Isolation Forest + live KS drift
python main.py --realtime --max-batches 100

# User UI — manual entry, file path, CSV upload (Streamlit)
python main.py --ui

# Live real-time dashboard
python main.py --dashboard

# Professional web UI + API (Sentinel)
python main.py --api
# Open http://127.0.0.1:8000/
```

The **Sentinel** web app includes overview, single-transaction scoring, batch upload/path,
live WebSocket stream, and system diagnostics — custom CSS design at `/ui/`.

## Frontend: how scoring works

| Step | What happens |
| ---- | -------------- |
| 1 | You enter features or provide a CSV (UI or API). |
| 2 | **Feature engineering** scales values using the training scaler. |
| 3 | **CatBoost** returns `fraud_probability` and a fraud/legit flag. |
| 4 | For multiple rows, **KS drift** runs vs the training baseline. |
| 5 | **Governance** may show `alert` or block API if drift is severe. |

**credit_dt:** raw `fraudTrain.csv` / `fraudTest.csv` are featurized automatically.  
**Kaggle:** columns `V1`–`V28`, `Time`, `Amount`.

| Interface | Best for |
| --------- | -------- |
| `python main.py --ui` | Streamlit — forms, path, upload (no API server needed) |
| `python main.py --api` → `/ui/` | Browser HTML UI calling REST API |
| `POST /v1/score` | Programmatic / manual JSON |
| `POST /v1/score/from-path` | Server-side path e.g. `data/credit_dt/fraudTest.csv` |
| `POST /v1/score/from-upload` | Multipart CSV upload |

## Real-Time AI Models

| Model | Purpose in stream |
| ----- | ----------------- |
| **CatBoost** | Fraud probability per micro-batch |
| **Isolation Forest** | Anomaly score for novel attack patterns |
| **KS drift** | Sliding-window feature drift vs training baseline |

Configured under `realtime:` in `config.yaml`. See **[docs/AI_WORLD_ARCHITECTURE.md](docs/AI_WORLD_ARCHITECTURE.md#real-time-ai-subsystem-l4-extension)**.

## Model cards

Documentation for each trained model (intended use, data, metrics, limitations):

```bash
python main.py --model-cards    # regenerate from production registry
```

| Location | Contents |
| -------- | -------- |
| `artifacts/model_cards/` | Latest cards + index `README.md` |
| `artifacts/registry/{version}/` | Versioned cards per release |

API: `GET /v1/models/cards` · `GET /v1/models/cards/{filename}.md`

## Project Layout

```text
Final_Project/
├── main.py                      # CLI: train | serve | api | --architecture
├── config.yaml                  # Data, ML, governance, paths
├── src/
│   ├── domain/                  # L1 entities & enums
│   ├── architecture/            # Layer map + PipelineContext
│   ├── orchestration/           # L3 FraudAIPipeline + stages
│   ├── governance/              # L1 policy engine
│   ├── registry/                # L6 model registry
│   ├── observability/           # L7 telemetry
│   ├── serving/                 # L2 runtime
│   ├── api/                     # L2 FastAPI + WebSocket
│   ├── realtime/                # L4 real-time AI engine
│   ├── etl/ engineering/ inference/ monitoring/  # L4 batch ML
│   dashboard/                   # L2 Streamlit render
│   └── data/                    # L5
├── artifacts/
│   ├── registry/                # Version index
│   └── telemetry/               # Run events & manifests
└── docs/
    ├── AI_WORLD_ARCHITECTURE.md
    └── ARCHITECTURE.md
```

## Data

Default: **`data/credit_dt/fraudTrain.csv`** (train) and **`fraudTest.csv`** (test).  
Optional: [Kaggle `creditcard.csv`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — set `data.dataset: kaggle_creditcard` in `config.yaml`.

## Configuration

`config.yaml` sections: `data`, `etl`, `engineering`, `model`, `monitoring`, **`governance`**, `paths` (includes `registry`).

## License

Educational / coursework. Kaggle data subject to its own terms.
