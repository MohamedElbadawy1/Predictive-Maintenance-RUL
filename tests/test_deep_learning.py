import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.deep_learning.lstm_model import build_lstm_baseline
from src.deep_learning.gru_model import build_gru_baseline
from src.deep_learning.dl_trainer import DLTrainer


class TestBuildLstmBaseline(unittest.TestCase):

    def test_output_shape(self):

        model = build_lstm_baseline(window_size=10, n_features=5)

        X = np.random.rand(4, 10, 5).astype("float32")
        preds = model.predict(X, verbose=0)

        self.assertEqual(preds.shape, (4, 1))

    def test_layer_stack(self):

        model = build_lstm_baseline(window_size=10, n_features=5, lstm_units=8, dropout=0.3)

        layer_types = [layer.__class__.__name__ for layer in model.layers]

        self.assertIn("LSTM", layer_types)
        self.assertIn("Dropout", layer_types)
        self.assertIn("Dense", layer_types)


class TestBuildGruBaseline(unittest.TestCase):

    def test_output_shape(self):

        model = build_gru_baseline(window_size=10, n_features=5)

        X = np.random.rand(4, 10, 5).astype("float32")
        preds = model.predict(X, verbose=0)

        self.assertEqual(preds.shape, (4, 1))

    def test_layer_stack(self):

        model = build_gru_baseline(window_size=10, n_features=5, gru_units=8, dropout=0.3)

        layer_types = [layer.__class__.__name__ for layer in model.layers]

        self.assertIn("GRU", layer_types)
        self.assertIn("Dropout", layer_types)
        self.assertIn("Dense", layer_types)


class TestDLTrainer(unittest.TestCase):

    def setUp(self):

        self.tmp_dir = Path(tempfile.mkdtemp())
        self.checkpoint_path = self.tmp_dir / "model.keras"

        rng = np.random.default_rng(42)
        self.X_train = rng.random((20, 6, 3)).astype("float32")
        self.y_train = rng.random(20).astype("float32") * 100
        self.X_val = rng.random((5, 6, 3)).astype("float32")
        self.y_val = rng.random(5).astype("float32") * 100

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_trainer(self):
        model = build_lstm_baseline(window_size=6, n_features=3, lstm_units=4)
        return DLTrainer(
            model,
            checkpoint_path=self.checkpoint_path,
            track_mlflow=False,
        )

    def test_train_returns_validation_metrics(self):

        trainer = self._make_trainer()

        metrics = trainer.train(
            self.X_train, self.y_train, self.X_val, self.y_val,
            max_epochs=1, batch_size=8, patience=1,
        )

        self.assertIn("MAE", metrics)

    def test_checkpoint_and_state_written(self):

        trainer = self._make_trainer()
        trainer.train(
            self.X_train, self.y_train, self.X_val, self.y_val,
            max_epochs=1, batch_size=8, patience=1,
        )

        self.assertTrue(self.checkpoint_path.exists())
        self.assertTrue(trainer.state_path.exists())

        with open(trainer.state_path) as f:
            state = json.load(f)

        self.assertEqual(state["last_completed_epoch"], 0)

    def test_resume_continues_from_last_epoch(self):

        trainer = self._make_trainer()
        trainer.train(
            self.X_train, self.y_train, self.X_val, self.y_val,
            max_epochs=2, batch_size=8, patience=5, resume=False,
        )

        with open(trainer.state_path) as f:
            state_after_first_run = json.load(f)
        self.assertEqual(state_after_first_run["last_completed_epoch"], 1)

        new_model = build_lstm_baseline(window_size=6, n_features=3, lstm_units=4)
        resumed_trainer = DLTrainer(
            new_model,
            checkpoint_path=self.checkpoint_path,
            track_mlflow=False,
        )
        resumed_trainer.train(
            self.X_train, self.y_train, self.X_val, self.y_val,
            max_epochs=4, batch_size=8, patience=5, resume=True,
        )

        with open(resumed_trainer.state_path) as f:
            state_after_resume = json.load(f)
        self.assertEqual(state_after_resume["last_completed_epoch"], 3)

    def test_predict_output_length(self):

        trainer = self._make_trainer()
        trainer.train(
            self.X_train, self.y_train, self.X_val, self.y_val,
            max_epochs=1, batch_size=8, patience=1,
        )

        preds = trainer.predict(self.X_val)

        self.assertEqual(len(preds), len(self.X_val))

    def test_save_and_load_roundtrip(self):

        trainer = self._make_trainer()
        trainer.train(
            self.X_train, self.y_train, self.X_val, self.y_val,
            max_epochs=1, batch_size=8, patience=1,
        )

        final_path = self.tmp_dir / "final_model.keras"
        trainer.save(final_path)

        loaded_trainer = DLTrainer.load(final_path)
        preds = loaded_trainer.predict(self.X_val)

        self.assertEqual(len(preds), len(self.X_val))


if __name__ == "__main__":
    unittest.main()
