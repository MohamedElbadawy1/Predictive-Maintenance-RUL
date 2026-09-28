"""
deploy/streamlit-cloud/app.py — standalone entry point for a free
Streamlit Community Cloud deployment: same UI as frontend/app.py, but
calls EnsembleInferencePipeline directly in-process instead of making
HTTP requests to a separate FastAPI service.

Why a separate file rather than reusing frontend/app.py as-is:
Community Cloud runs exactly one Python process per app — there's
nowhere to run a second (FastAPI) service alongside it. Folding the
API call into a direct pipeline.predict() call sidesteps that
entirely: no second process, no port to coordinate, no API_KEY to
configure, no Docker. See README.md's "Deploying to Streamlit
Community Cloud" for the full setup this enables.

Deploy this file with Community Cloud's main file path set to
deploy/streamlit-cloud/app.py — it finds requirements.txt and
sample_data.csv, both in this same folder, automatically.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.exceptions.custom_exception import CustomException
from src.pipelines.ensemble_inference_pipeline import EnsembleInferencePipeline

SAMPLE_DATA_PATH = Path(__file__).resolve().parent / "sample_data.csv"

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

# "Use sample data" is a plain st.button, which only returns True on the
# one rerun right after it's clicked -- Streamlit reruns the whole
# script on every interaction, including the "Predict RUL" click below,
# so relying on the button's own return value would silently forget the
# sample data was selected the moment someone clicks Predict. Session
# state is what makes the choice stick across those reruns.
if "data_source" not in st.session_state:
    st.session_state.data_source = None

col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader("CSV file", type=["csv"])
with col2:
    st.write("")
    st.write("")
    if st.button("Use sample data"):
        st.session_state.data_source = "sample"

if uploaded_file is not None:
    st.session_state.data_source = "upload"

readings_df = None
if st.session_state.data_source == "sample":
    readings_df = pd.read_csv(SAMPLE_DATA_PATH)
    st.caption(
        "Using the bundled sample — 3 real NASA C-MAPSS FD004 test engines "
        "(units 1, 5, 12), no upload needed."
    )
elif st.session_state.data_source == "upload" and uploaded_file is not None:
    try:
        readings_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()

if readings_df is not None:

    n_engines = readings_df["unit_number"].nunique() if "unit_number" in readings_df.columns else 0
    st.write(f"Loaded **{len(readings_df)}** rows across **{n_engines}** engine(s).")
    st.dataframe(readings_df.head(10), width="stretch")

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

        st.dataframe(preds_df, width="stretch")
        st.bar_chart(preds_df.set_index("unit_number")["predicted_RUL"])

else:
    st.info("Upload a CSV of raw engine readings, or click \"Use sample data\", to get started.")
