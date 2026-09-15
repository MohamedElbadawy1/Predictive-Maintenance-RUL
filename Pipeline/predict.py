"""
Pipeline/predict.py — CLI wrapper around MLflowInferencePipeline
(src/pipelines/inference_pipeline.py): use the champion model directly
from MLflow, from any machine with access to the same tracking store.

The actual class lives in src/, not here — see the docstring in
src/pipelines/inference_pipeline.py for why (a cross-folder import
from api/dependencies.py used to break depending on the checked-out
folder's exact case).

Usage:
    python Pipeline/predict.py --input path/to/raw_engine_data.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.pipelines.inference_pipeline import MLflowInferencePipeline, REGISTRY_MODEL_NAME

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV of raw engine readings")
    parser.add_argument("--registry-model-name", default=REGISTRY_MODEL_NAME)
    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)

    pipeline = MLflowInferencePipeline(registry_model_name=args.registry_model_name)
    predictions = pipeline.predict(raw_df)

    print(predictions.to_string(index=False))
