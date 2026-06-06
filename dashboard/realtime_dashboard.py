"""
Real-Time Fraud AI Dashboard (L2 Experience — visual render layer).

Run: streamlit run dashboard/realtime_dashboard.py
Requires: trained model (python main.py). Streams fraudTest.csv when dataset=credit_dt.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.architecture.layers import AIWorldStack
from src.data.datasets import dataset_kind, get_stream_data_path
from src.realtime.ai_engine import RealtimeFraudAIEngine
from src.utils.config import load_config

st.set_page_config(
    page_title="Real-Time Fraud AI",
    page_icon="🛡️",
    layout="wide",
)

config = load_config()


@st.cache_resource
def get_engine() -> RealtimeFraudAIEngine:
    engine = RealtimeFraudAIEngine(config)
    engine.load()
    return engine


def main() -> None:
    st.title("Real-Time Financial Fraud Detection")
    st.caption("AI models on AI World architecture — CatBoost · Isolation Forest · KS Drift")

    with st.sidebar:
        st.subheader("Stream controls")
        batch_size = st.slider("Batch size", 5, 50, config["realtime"].get("stream_batch_size", 10))
        tps = st.slider("Transactions / sec", 5, 100, config["realtime"].get("transactions_per_second", 25))
        max_ticks = st.slider("Dashboard ticks", 10, 500, 120)
        auto_run = st.toggle("Auto-stream", value=False)
        step = st.button("Process one batch", use_container_width=True)
        reset = st.button("Reset engine", use_container_width=True)

        stream_path = get_stream_data_path(config)
        st.caption(f"Dataset: **{dataset_kind(config)}**")
        st.text(f"Stream: {stream_path}")

        st.divider()
        with st.expander("AI World layers"):
            st.code(AIWorldStack.describe())

    if reset:
        st.cache_resource.clear()
        st.rerun()

    engine = get_engine()
    config["realtime"]["stream_batch_size"] = batch_size
    config["realtime"]["transactions_per_second"] = tps

    if "history" not in st.session_state:
        st.session_state.history = []

    col1, col2, col3, col4 = st.columns(4)
    m1, m2, m3, m4 = col1.empty(), col2.empty(), col3.empty(), col4.empty()
    chart_fraud = st.empty()
    chart_drift = st.empty()
    alert_box = st.empty()
    arch_cols = st.columns(7)

    layer_labels = [
        "L1 Risk",
        "L2 UI",
        "L3 Orch",
        "L4 AI",
        "L5 Data",
        "L6 MLOps",
        "L7 Obs",
    ]
    layer_status = {k: "idle" for k in layer_labels}

    def render_layers(active: str) -> None:
        for i, label in enumerate(layer_labels):
            color = "#22c55e" if label == active else "#334155"
            arch_cols[i].markdown(
                f"<div style='background:{color};padding:8px;border-radius:6px;"
                f"text-align:center;color:white;font-size:11px'>{label}</div>",
                unsafe_allow_html=True,
            )

    def process_tick() -> None:
        from src.realtime.stream import TransactionStream

        stream_path = get_stream_data_path(config)
        stream = TransactionStream(
            stream_path,
            batch_size=batch_size,
            shuffle=True,
            random_state=config["data"]["random_state"],
        )
        batch = next(stream.batches())
        layer_status["L5 Data"] = "active"
        snap = engine.process_batch(batch)
        st.session_state.history.append(snap.to_dict())
        if len(st.session_state.history) > max_ticks:
            st.session_state.history = st.session_state.history[-max_ticks:]

    if step:
        process_tick()

    if auto_run:
        process_tick()
        time.sleep(max(0.05, batch_size / tps))
        st.rerun()

    hist = st.session_state.history
    if hist:
        latest = hist[-1]
        df = pd.DataFrame(hist)

        m1.metric("Transactions processed", f"{latest['total_processed']:,}")
        m2.metric("Fraud prob (batch avg)", f"{latest['mean_fraud_probability']:.3f}")
        m3.metric("Frauds flagged (batch)", latest["fraud_detected"])
        m4.metric("Governance", latest["governance_action"].upper())

        chart_fraud.subheader("CatBoost fraud probability (stream)")
        chart_fraud.line_chart(df.set_index("batch_id")["mean_fraud_probability"])

        if df["mean_anomaly_score"].notna().any():
            chart_drift.subheader("Isolation Forest anomaly score")
            chart_drift.line_chart(df.set_index("batch_id")["mean_anomaly_score"])

        drift_count = df["drifted_features"].apply(len)
        st.subheader("KS drift — features flagged per tick")
        st.area_chart(drift_count.rename("drift_feature_count"))

        if latest["drifted_features"]:
            alert_box.warning(
                f"Drift detected on: {', '.join(latest['drifted_features'][:8])}"
                + (" …" if len(latest["drifted_features"]) > 8 else "")
            )
        else:
            alert_box.success("No feature drift in latest window check")

        if latest["blocked"]:
            alert_box.error("Serving BLOCKED by L1 governance — retrain recommended")

        render_layers("L4 AI")
    else:
        st.info("Click **Process one batch** or enable **Auto-stream** to start the real-time AI pipeline.")
        render_layers("L2 UI")

    st.divider()
    st.subheader("Latest AI snapshot (JSON)")
    st.json(hist[-1] if hist else engine.status())


if __name__ == "__main__":
    main()
