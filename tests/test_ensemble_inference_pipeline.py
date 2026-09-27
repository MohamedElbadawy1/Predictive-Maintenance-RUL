import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.deep_learning.data_prep import RAW_FEATURE_COLUMNS
from src.deep_learning.lstm_model import build_lstm_baseline
from src.deep_learning.gru_model import build_gru_baseline
from src.exceptions.custom_exception import CustomException
from src.preprocessing.regime_normalizer import RegimeNormalizer
from src.preprocessing.feature_scaler import FeatureScaler
from src.utils.constant import SENSOR_COLUMNS

# Reuses Pipeline/train_ensemble.py's own Keras -> ONNX conversion
# rather than duplicating it, so this test exercises the exact export
# path production artifacts go through, not a second, independent one
# that could silently drift out of sync with it.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Pipeline.train_ensemble import _export_to_onnx

WINDOW_SIZE = 6
N_FEATURES = len(RAW_FEATURE_COLUMNS)  # 25


def _make_synthetic_raw_df(engine_id: int, n_cycles: int, rng: np.random.Generator) -> pd.DataFrame:
    """A minimal but schema-complete raw engine-cycle dataframe -- same
    shape a real /predict request body maps to."""

    data = {
        "unit_number": [engine_id] * n_cycles,
        "time_in_cycles": list(range(1, n_cycles + 1)),
        "operational_setting_1": rng.uniform(0, 42, n_cycles),
        "operational_setting_2": rng.uniform(0, 0.84, n_cycles),
        "operational_setting_3": rng.uniform(60, 100, n_cycles),
    }
    for col in SENSOR_COLUMNS:
        data[col] = rng.uniform(1, 1000, n_cycles)

    return pd.DataFrame(data)


class TestEnsembleInferencePipeline(unittest.TestCase):
    """Builds a tiny, fully self-contained ensemble (random-weight LSTM
    + GRU exported to ONNX, a scaler/regime-normalizer fit on synthetic
    data) under a temp directory and patches the module-level path
    constants EnsembleInferencePipeline reads from -- no real training,
    no real CMAPSS data, no artifacts/ dependency."""

    def setUp(self):

        self.tmp_dir = Path(tempfile.mkdtemp())
        self.rng = np.random.default_rng(42)

        self.ensemble_dir = self.tmp_dir / "ensemble"
        self.ensemble_dir.mkdir(parents=True)

        # Fit real (tiny) preprocessing artifacts on synthetic training rows
        train_df = pd.concat(
            [_make_synthetic_raw_df(eid, 20, self.rng) for eid in range(1, 6)],
            ignore_index=True,
        )

        self.regime_normalizer = RegimeNormalizer(n_regimes=2, sensor_columns=SENSOR_COLUMNS, random_state=42)
        normalized = self.regime_normalizer.fit(train_df).transform(train_df)

        self.scaler = FeatureScaler()
        self.scaler.fit(normalized[RAW_FEATURE_COLUMNS])

        self.scaler_path = self.ensemble_dir / "feature_scaler.pkl"
        self.regime_path = self.ensemble_dir / "regime_normalizer.pkl"
        self.scaler.save(self.scaler_path)
        self.regime_normalizer.save(self.regime_path)

        # Tiny, untrained (random-weight) LSTM/GRU, exported to ONNX the
        # same way Pipeline/train_ensemble.py exports the real ones --
        # correctness of the *averaging and windowing plumbing* is what's
        # under test here, not prediction quality.
        lstm_model = build_lstm_baseline(window_size=WINDOW_SIZE, n_features=N_FEATURES, lstm_units=4)
        gru_model = build_gru_baseline(window_size=WINDOW_SIZE, n_features=N_FEATURES, gru_units=4)

        self.lstm_onnx_path = self.ensemble_dir / "lstm.onnx"
        self.gru_onnx_path = self.ensemble_dir / "gru.onnx"
        _export_to_onnx(lstm_model, self.lstm_onnx_path)
        _export_to_onnx(gru_model, self.gru_onnx_path)

        self.manifest_path = self.ensemble_dir / "manifest.json"
        self.manifest = {
            "seed": 42,
            "window_size": WINDOW_SIZE,
            "regime_aware": True,
            "feature_columns": RAW_FEATURE_COLUMNS,
            "lstm_onnx": self.lstm_onnx_path.name,
            "gru_onnx": self.gru_onnx_path.name,
        }
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f)

        self._patches = [
            patch("src.pipelines.ensemble_inference_pipeline.ENSEMBLE_MANIFEST_PATH", self.manifest_path),
            patch("src.pipelines.ensemble_inference_pipeline.ENSEMBLE_SCALER_PATH", self.scaler_path),
            patch("src.pipelines.ensemble_inference_pipeline.ENSEMBLE_REGIME_NORMALIZER_PATH", self.regime_path),
            patch("src.pipelines.ensemble_inference_pipeline.ENSEMBLE_DIR", self.ensemble_dir),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _load_pipeline(self):
        # Imported lazily (after patches are active), same reasoning as
        # api/dependencies.py's own lazy import.
        from src.pipelines.ensemble_inference_pipeline import EnsembleInferencePipeline
        return EnsembleInferencePipeline()

    def test_predicts_one_row_per_engine_with_full_history(self):

        pipeline = self._load_pipeline()
        raw_df = pd.concat(
            [_make_synthetic_raw_df(1, 10, self.rng), _make_synthetic_raw_df(2, 15, self.rng)],
            ignore_index=True,
        )

        result = pipeline.predict(raw_df)

        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["unit_number"]), {1, 2})
        self.assertFalse(result["short_history_warning"].any())
        self.assertTrue(result["predicted_RUL"].notna().all())
        self.assertIn("regime", result.columns)
        self.assertTrue((result["champion_version"] == "lstm_gru_ensemble-seed42").all())

    def test_prediction_is_average_of_lstm_and_gru(self):
        """The pipeline's output must equal (lstm_pred + gru_pred) / 2
        computed independently on the exact same scaled window -- not
        just 'some number', to actually pin down the averaging logic."""

        pipeline = self._load_pipeline()
        raw_df = _make_synthetic_raw_df(1, WINDOW_SIZE, self.rng)

        result = pipeline.predict(raw_df)

        normalized = self.regime_normalizer.transform(raw_df)
        scaled = self.scaler.transform(normalized[RAW_FEATURE_COLUMNS])
        X = scaled.to_numpy(dtype=np.float32)[-WINDOW_SIZE:][None, ...]

        lstm_pred = pipeline.lstm_session.run(None, {pipeline.lstm_input_name: X})[0].flatten()[0]
        gru_pred = pipeline.gru_session.run(None, {pipeline.gru_input_name: X})[0].flatten()[0]
        expected = (lstm_pred + gru_pred) / 2

        self.assertAlmostEqual(float(result.loc[0, "predicted_RUL"]), float(expected), places=4)

    def test_engine_with_short_history_is_flagged_not_dropped(self):

        pipeline = self._load_pipeline()
        raw_df = pd.concat(
            [_make_synthetic_raw_df(1, WINDOW_SIZE, self.rng), _make_synthetic_raw_df(2, WINDOW_SIZE - 1, self.rng)],
            ignore_index=True,
        )

        result = pipeline.predict(raw_df)
        result = result.set_index("unit_number")

        self.assertFalse(result.loc[1, "short_history_warning"])
        self.assertTrue(result.loc[2, "short_history_warning"])
        self.assertTrue(pd.isna(result.loc[2, "predicted_RUL"]))
        self.assertEqual(result.loc[2, "n_cycles_seen"], WINDOW_SIZE - 1)

    def test_every_engine_below_window_size_raises(self):

        pipeline = self._load_pipeline()
        raw_df = _make_synthetic_raw_df(1, WINDOW_SIZE - 1, self.rng)

        with self.assertRaises(CustomException):
            pipeline.predict(raw_df)

    def test_missing_required_column_raises(self):

        pipeline = self._load_pipeline()
        raw_df = _make_synthetic_raw_df(1, WINDOW_SIZE, self.rng).drop(columns=["sensor_5"])

        with self.assertRaises(CustomException):
            pipeline.predict(raw_df)

    def test_missing_manifest_raises_on_construction(self):

        self.manifest_path.unlink()

        with self.assertRaises(CustomException):
            self._load_pipeline()


if __name__ == "__main__":
    unittest.main()
