import os
import tempfile
import unittest

import pandas as pd

from src.preprocessing.feature_scaler import FeatureScaler
from src.exceptions.custom_exception import CustomException


class TestFeatureScaler(unittest.TestCase):

    def setUp(self):

        self.df = pd.DataFrame({
            "sensor_1": [1, 2, 3, 4],
            "sensor_2": [5, 6, 7, 8],
            "sensor_3": [9, 10, 11, 12],
        })

    def test_fit_transform(self):

        scaler = FeatureScaler()

        scaled = scaler.fit_transform(self.df)

        self.assertEqual(scaled.shape, self.df.shape)

    def test_transform_preserves_columns(self):

        scaler = FeatureScaler()

        scaler.fit(self.df)

        scaled = scaler.transform(self.df)

        self.assertEqual(
            list(scaled.columns),
            list(self.df.columns)
        )

    def test_transform_preserves_index(self):

        scaler = FeatureScaler()

        scaler.fit(self.df)

        scaled = scaler.transform(self.df)

        self.assertTrue(
            scaled.index.equals(self.df.index)
        )

    def test_transform_before_fit(self):

        scaler = FeatureScaler()

        with self.assertRaises(CustomException):
            scaler.transform(self.df)

    def test_empty_dataframe(self):

        scaler = FeatureScaler()

        with self.assertRaises(CustomException):
            scaler.fit(pd.DataFrame())

    def test_save_scaler(self):

        scaler = FeatureScaler()

        scaler.fit(self.df)

        with tempfile.TemporaryDirectory() as tmp:

            path = os.path.join(tmp, "scaler.pkl")

            scaler.save(path)

            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()