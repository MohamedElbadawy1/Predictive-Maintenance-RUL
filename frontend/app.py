"""
frontend/app.py — minimal Streamlit UI for the Predictive Maintenance
RUL API. Upload a CSV of raw engine cycle readings (one row per engine
per cycle, full history up to "now" — same shape Pipeline/predict.py
expects) and see each engine's predicted remaining useful life.

The FastAPI app must be running separately — this is a thin client, it
never loads the model itself:

    uvicorn api.api_main:app --reload
    streamlit run frontend/app.py
"""
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontend.api_client import check_health, check_readiness, predict

st.set_page_config(page_title="Predictive Maintenance — RUL", page_icon="🔧", layout="centered")

st.title("🔧 Predictive Maintenance — RUL Prediction")
st.caption(
    "Upload raw engine cycle readings and get remaining-useful-life "
    "predictions from the current MLflow champion model."
)

api_url = st.sidebar.text_input(
    "API base URL",
    value=os.environ.get("API_URL", "http://localhost:8000"),
).rstrip("/")

st.sidebar.markdown("### API status")
try:
    health = check_health(api_url)
    st.sidebar.success(f"API reachable ({health.get('status', 'unknown')})")
    try:
        ready = check_readiness(api_url)
        if ready.get("model_loaded"):
            st.sidebar.success("Champion model loaded")
        else:
            st.sidebar.info("Champion model not loaded yet (loads on first prediction)")
    except requests.RequestException:
        st.sidebar.warning("Could not check readiness")
except requests.RequestException as exc:
    st.sidebar.error(f"API not reachable at {api_url}")
    st.sidebar.caption(str(exc))

st.subheader("1. Upload engine readings")
st.caption(
    "CSV with one row per (engine, cycle): unit_number, time_in_cycles, "
    "operational_setting_1-3, sensor_1-21."
)

uploaded_file = st.file_uploader("CSV file", type=["csv"])

if uploaded_file is not None:

    try:
        readings_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()

    n_engines = readings_df["unit_number"].nunique() if "unit_number" in readings_df.columns else 0
    st.write(f"Loaded **{len(readings_df)}** rows across **{n_engines}** engine(s).")
    st.dataframe(readings_df.head(10), width="stretch")

    if st.button("Predict RUL", type="primary"):

        with st.spinner("Calling the API..."):
            try:
                result = predict(api_url, readings_df.to_dict(orient="records"))
            except requests.RequestException as exc:
                st.error(f"Prediction request failed: {exc}")
                st.stop()

        st.subheader("2. Predictions")
        preds_df = pd.DataFrame(result["predictions"])
        st.caption(f"Champion model version: {result.get('model_version', 'unknown')}")

        if preds_df["short_history_warning"].any():
            n_warned = int(preds_df["short_history_warning"].sum())
            st.warning(
                f"{n_warned} engine(s) have limited cycle history — "
                "their prediction may be less reliable."
            )

        st.dataframe(preds_df, width="stretch")
        st.bar_chart(preds_df.set_index("unit_number")["predicted_RUL"])

else:
    st.info("Upload a CSV of raw engine readings to get started.")
