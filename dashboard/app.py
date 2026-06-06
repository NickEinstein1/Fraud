"""
Fraud Detection — User Frontend (L2 Experience)

Run:  python main.py --ui
  or: streamlit run dashboard/app.py

Tabs:
  - Manual entry: type Time, Amount, V1–V28
  - File path: score a CSV on disk (e.g. data/creditcard.csv)
  - Upload: drag-and-drop CSV
  - How it works: system flow
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.scoring_utils import feature_columns, load_csv, project_root, score_dataframe
from src.architecture.layers import AIWorldStack
from src.data.datasets import dataset_kind
from src.utils.config import load_config

st.set_page_config(page_title="Fraud Detection UI", page_icon="🛡️", layout="wide")

config = load_config()
_ds = dataset_kind(config)
if _ds == "credit_dt":
    default_data = config["data"]["credit_dt"]["test_path"]
else:
    default_data = config["data"]["path"]


def render_manual_tab() -> None:
    st.subheader("Enter transaction features")
    cols = feature_columns()

    if _ds == "credit_dt":
        from src.data.credit_dt import feature_schema_for_ui

        st.caption(
            "Use text labels for **category**, **gender**, **state**. "
            "**distance_km** is recomputed from lat/long and merchant coordinates (Haversine)."
        )
        schema = feature_schema_for_ui(config)
        defaults = {
            "amt": 127.16,
            "lat": 40.0,
            "long": -74.0,
            "merch_lat": 40.01,
            "merch_long": -74.01,
            "city_pop": 500000.0,
            "unix_time": 1_325_376_018.0,
            "trans_hour": 12.0,
            "trans_dow": 3.0,
            "category": "shopping_net",
            "gender": "M",
            "state": "NY",
        }
        row: dict[str, float | str] = {}
        for field in schema:
            name = field["name"]
            if field.get("type") == "categorical":
                opts = field.get("options") or []
                default = defaults.get(name, opts[0] if opts else "")
                idx = opts.index(default) if default in opts else 0
                row[name] = st.selectbox(name, opts, index=idx, key=f"cdt_{name}")
            elif field.get("computed"):
                row[name] = st.number_input(
                    f"{name} (optional override)",
                    value=0.0,
                    key=f"cdt_{name}",
                    help=field.get("description", ""),
                )
            else:
                with st.expander(field.get("label", name), expanded=name in ("amt", "lat")):
                    row[name] = st.number_input(
                        name, value=float(defaults.get(name, 0.0)), key=f"cdt_{name}"
                    )
    else:
        st.caption("Kaggle schema: **V1–V28**, **Time**, **Amount**.")
        mode = st.radio(
            "Input mode", ["Essential (Time & Amount only)", "Full (all 30 features)"], horizontal=True
        )
        c1, c2 = st.columns(2)
        with c1:
            time_val = st.number_input("Time", value=0.0, format="%.6f")
        with c2:
            amount_val = st.number_input("Amount", value=149.62, format="%.4f")
        row = {f"V{i}": 0.0 for i in range(1, 29)}
        row.update({"Time": time_val, "Amount": amount_val})
        if not mode.startswith("Full"):
            pass
        else:
            with st.expander("V1 – V28", expanded=True):
                gc = st.columns(4)
                for i in range(1, 29):
                    key = f"V{i}"
                    with gc[(i - 1) % 4]:
                        row[key] = st.number_input(key, value=0.0, format="%.6f", key=f"manual_{key}")

    if st.button("Score transaction", type="primary", use_container_width=True):
        df = pd.DataFrame([row])
        try:
            result, summary, gov = score_dataframe(df)
            prob = float(result["fraud_probability"].iloc[0])
            pred = int(result["fraud_prediction"].iloc[0])

            m1, m2, m3 = st.columns(3)
            m1.metric("Fraud probability", f"{prob:.4f}")
            m2.metric("Prediction", "FRAUD" if pred == 1 else "LEGIT")
            m3.metric("Governance", gov.upper())

            if pred == 1 or prob > 0.5:
                st.error("This transaction is flagged as potential fraud.")
            else:
                st.success("This transaction appears legitimate.")

            st.json({"batch_summary": summary, "governance": gov})
        except Exception as exc:
            st.error(f"Scoring failed: {exc}")
            st.info("Train the model first: `python main.py`")


def render_file_path_tab() -> None:
    st.subheader("Score from CSV file path")
    st.caption("Path is relative to the project folder.")

    path_input = st.text_input(
        "File path",
        value=default_data,
        placeholder="data/creditcard.csv",
    )
    max_rows = st.slider("Max rows to score", 10, 5000, 200)

    if st.button("Load and score", type="primary", use_container_width=True):
        full_path = project_root() / path_input
        try:
            df = load_csv(full_path, max_rows=max_rows)
            st.write(f"Loaded **{len(df)}** rows from `{path_input}`")
            result, summary, gov = score_dataframe(df)

            frauds = int(result["fraud_prediction"].sum())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows scored", len(result))
            m2.metric("Fraud flags", frauds)
            m3.metric("Fraud rate", f"{frauds / len(result):.2%}")
            m4.metric("Governance", gov.upper())

            st.dataframe(
                result.sort_values("fraud_probability", ascending=False).head(50),
                use_container_width=True,
            )

            if summary.get("drifted_features"):
                st.warning(f"Drift detected on: {', '.join(summary['drifted_features'][:10])}")

            csv_bytes = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results CSV",
                data=csv_bytes,
                file_name="fraud_scores.csv",
                mime="text/csv",
            )
        except FileNotFoundError:
            st.error(f"File not found: `{full_path}`")
        except Exception as exc:
            st.error(f"Error: {exc}")


def render_upload_tab() -> None:
    st.subheader("Upload CSV file")
    if _ds == "credit_dt":
        st.caption("Raw **fraudTrain** / **fraudTest** CSV OK — featurized automatically.")
    else:
        st.caption("File must include: V1 … V28, Time, Amount (optional Class ignored)")

    uploaded = st.file_uploader("Choose CSV", type=["csv"])
    max_rows = st.slider("Max rows", 10, 5000, 200, key="upload_max")

    if uploaded is not None:
        df = pd.read_csv(uploaded, nrows=max_rows)
        st.write(f"Preview ({len(df)} rows)")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("Score uploaded file", type="primary", use_container_width=True):
            try:
                result, summary, gov = score_dataframe(df)
                st.success(f"Scored {len(result)} rows · Governance: {gov}")
                st.dataframe(
                    result.sort_values("fraud_probability", ascending=False).head(50),
                    use_container_width=True,
                )
                st.download_button(
                    "Download results",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_scores.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(str(exc))


def render_how_it_works() -> None:
    st.subheader("How this system works")
    st.markdown(
        """
        ### Top-down flow

        1. **You provide data** — manual form, file path, or upload (this UI, L2 Experience).
        2. **Feature engineering** — values are scaled with the same `StandardScaler` saved during training (L4).
        3. **CatBoost model** — outputs `fraud_probability` and a binary flag (L4 Inference).
        4. **Drift monitoring** — for batches, KS-tests compare feature distributions to the training baseline (L4 Monitoring).
        5. **Governance** — if too many features drift, the system may `alert` or `block_serve` (L1 Business).

        ### credit_dt dataset (default)

        - **Train:** `data/credit_dt/fraudTrain.csv` (~1.3M rows)
        - **Test:** `data/credit_dt/fraudTest.csv` (~555k rows)
        - Set `data.dataset: credit_dt` in `config.yaml` (already default)

        ### Before using the UI

        ```bash
        python main.py          # train on fraudTrain, evaluate on fraudTest
        python main.py --ui     # score fraudTest or upload raw CSV
        ```

        ### Alternative interfaces

        | Interface | Command |
        |-----------|---------|
        | This UI (Streamlit) | `python main.py --ui` |
        | Web UI (HTML) | `python main.py --api` then open http://127.0.0.1:8000/ui/ |
        | Real-time stream dashboard | `python main.py --dashboard` |
        | CLI batch | `python main.py --serve data/creditcard.csv` |
        """
    )
    st.code(AIWorldStack.describe())


def main() -> None:
    st.title("Financial Fraud Detection")
    st.caption("Enter values · open a file path · or upload CSV — powered by CatBoost + drift AI")

    tab_manual, tab_path, tab_upload, tab_help = st.tabs(
        ["Manual entry", "File path", "Upload CSV", "How it works"]
    )

    with tab_manual:
        render_manual_tab()
    with tab_path:
        render_file_path_tab()
    with tab_upload:
        render_upload_tab()
    with tab_help:
        render_how_it_works()

    with st.sidebar:
        st.subheader("System status")
        model_path = project_root() / config["paths"]["models"] / "catboost_fraud.cbm"
        if model_path.exists():
            st.success("Model loaded path exists")
        else:
            st.error("No model — run `python main.py`")
        st.caption(f"Default data: `{default_data}`")


if __name__ == "__main__":
    main()
