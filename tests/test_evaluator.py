import unittest
import numpy as np

from src.evaluation.evaluator import RegressionEvaluator
from src.exceptions.custom_exception import CustomException


class TestRegressionEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = RegressionEvaluator()

    def test_perfect_prediction(self):

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        metrics = self.evaluator.evaluate(y_true, y_pred)

        self.assertEqual(metrics["MAE"], 0.0)
        self.assertEqual(metrics["RMSE"], 0.0)
        self.assertEqual(metrics["MAPE"], 0.0)
        self.assertEqual(metrics["R2"], 1.0)

    def test_normal_prediction(self):

        y_true = np.array([10, 20, 30, 40])
        y_pred = np.array([12, 18, 28, 39])

        metrics = self.evaluator.evaluate(y_true, y_pred)

        self.assertIn("MAE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertIn("MAPE", metrics)
        self.assertIn("R2", metrics)

    def test_none_input(self):

        with self.assertRaises(CustomException):
            self.evaluator.evaluate(None, None)

    def test_empty_input(self):

        with self.assertRaises(CustomException):
            self.evaluator.evaluate([], [])

    def test_length_mismatch(self):

        with self.assertRaises(CustomException):
            self.evaluator.evaluate(
                [1, 2, 3],
                [1, 2],
            )

    def test_zero_values(self):

        y_true = np.array([0, 10, 20])
        y_pred = np.array([0, 12, 18])

        metrics = self.evaluator.evaluate(
            y_true,
            y_pred,
        )

        self.assertTrue(np.isfinite(metrics["MAPE"]))


if __name__ == "__main__":
    unittest.main()