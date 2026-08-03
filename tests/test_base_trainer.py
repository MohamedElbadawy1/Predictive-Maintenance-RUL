import unittest
import tempfile
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

from src.models.base_trainer import BaseTrainer


class DummyTrainer(BaseTrainer):

    def build_model(self):
        return LinearRegression()

    def train(self, X_train, y_train):
        self.model = self.build_model()
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)


class TestBaseTrainer(unittest.TestCase):

    def setUp(self):

        self.trainer = DummyTrainer()

        self.X = pd.DataFrame({
            "x": [1, 2, 3, 4, 5]
        })

        self.y = pd.Series(
            [2, 4, 6, 8, 10]
        )

    def test_train(self):

        self.trainer.train(
            self.X,
            self.y,
        )

        self.assertIsNotNone(
            self.trainer.model
        )

    def test_predict(self):

        self.trainer.train(
            self.X,
            self.y,
        )

        pred = self.trainer.predict(
            self.X
        )

        self.assertEqual(
            len(pred),
            len(self.X),
        )

    def test_save_load(self):

        self.trainer.train(
            self.X,
            self.y,
        )

        with tempfile.TemporaryDirectory() as tmp:

            path = f"{tmp}/model.pkl"

            self.trainer.save(path)

            model = self.trainer.load(path)

            self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()