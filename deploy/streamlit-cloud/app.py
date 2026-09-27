"""
deploy/streamlit-cloud/app.py — standalone entry point for a free
Streamlit Community Cloud deployment: same UI as frontend/app.py, but
calls EnsembleInferencePipeline directly in-process instead of making
HTTP requests to a separate FastAPI service.

Why a separate file rather than reusing frontend/app.py as-is:
Community Cloud runs exactly one process per app — there's nowhere to
run a second (FastAPI) service alongside it, and no equivalent of
deploy/render/start.sh's "background API + foreground frontend" trick,
since Community Cloud only starts the one Python file you point it at.
Folding the API call into a direct pipeline.predict() call sidesteps
that entirely: no second process, no port to coordinate, no API_KEY to
configure. It also means this deployment needs no card, no Dockerfile,
and no separate web service at all -- see README.md's "Deploying to
Streamlit Community Cloud" for the three-click setup this enables.

Deploy this file with Community Cloud's main file path set to
deploy/streamlit-cloud/app.py, and requirements-serving.txt (same one
deploy/render/ uses, minus fastapi/uvicorn, which aren't needed here)
as this folder's requirements.txt.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.exceptions.custom_exception import CustomException
from src.pipelines.ensemble_inference_pipeline import EnsembleInferencePipeline

st.set_page_config(page_title="Predictive Maintenance — RUL", page_icon="🔧", layout="centered")

st.title("🔧 Predictive Maintenance — RUL Prediction")
st.caption(
    "Upload raw engine cycle readings and get remaining-useful-life "
    "predictions from the LSTM+GRU ensemble (test MAE 16.39 on NASA "
    "C-MAPSS FD004 — see the project README)."
)


@st.cache_resource(show_spinner="Loading the LSTM+GRU ensemble...")
def load_pipeline() -> EnsembleInferencePipeline:
    # Cached across reruns/sessions for the life of the process --
    # otherwise every button click would reload both ONNX models from
    # disk. This is the direct-call equivalent of api/dependencies.py's
    # PipelineCache.
    return EnsembleInferencePipeline()


st.sidebar.markdown("### Model status")
try:
    pipeline = load_pipeline()
    st.sidebar.success(f"Ensemble loaded ({pipeline.version})")
except CustomException as exc:
    st.sidebar.error("Could not load the ensemble artifacts.")
    st.sidebar.caption(str(exc))
    st.stop()

st.subheader("1. Upload engine readings")
st.caption(
    "CSV with one row per (engine, cycle): unit_number, time_in_cycles, "
    "operational_setting_1-3, sensor_1-21. Each engine needs at least "
    f"{pipeline.window_size} rows of history."
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
    st.dataframe(readings_df.head(10), use_container_width=True)

    if st.button("Predict RUL", type="primary"):

        with st.spinner("Running inference..."):
            try:
                preds_df = pipeline.predict(readings_df)
            except CustomException as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

        st.subheader("2. Predictions")
        st.caption(f"Model version: {pipeline.version}")

        if preds_df["short_history_warning"].any():
            n_warned = int(preds_df["short_history_warning"].sum())
            st.warning(
                f"{n_warned} engine(s) have limited cycle history — "
                "their prediction may be less reliable."
            )

        st.dataframe(preds_df, use_container_width=True)
        st.bar_chart(preds_df.set_index("unit_number")["predicted_RUL"])

else:
    st.info("Upload a CSV of raw engine readings to get started.")
