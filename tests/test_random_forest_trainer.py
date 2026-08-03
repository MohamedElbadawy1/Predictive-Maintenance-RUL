import unittest
import pandas as pd

from src.models.random_forest_trainer import RandomForestTrainer


class TestRandomForestTrainer(unittest.TestCase):

    def setUp(self):

        self.X = pd.DataFrame({

            "a": [1,2,3,4,5,6],

            "b": [2,3,4,5,6,7]

        })

        self.y = pd.Series(
            [3,5,7,9,11,13]
        )

        self.trainer = RandomForestTrainer(
            n_estimators=10
        )

    def test_training(self):

        self.trainer.train(
            self.X,
            self.y,
        )

        self.assertIsNotNone(
            self.trainer.model
        )

    def test_prediction(self):

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


if __name__ == "__main__":
    unittest.main()