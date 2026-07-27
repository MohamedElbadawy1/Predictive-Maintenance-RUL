import unittest

import pandas as pd

from src.preprocessing.data_splitter import DataSplitter
from src.preprocessing.feature_selector import FeatureSelector
from src.preprocessing.feature_scaler import FeatureScaler
from src.preprocessing.preparation_pipeline import DataPreparationPipeline


class TestPreparationPipeline(unittest.TestCase):

    def setUp(self):

        rows = []

        for engine in range(1, 11):
            for cycle in range(1, 6):

                rows.append({

                    "unit_number": engine,
                    "time_in_cycles": cycle,

                    "sensor_1": engine * cycle,
                    "sensor_2": engine + cycle,

                    "RUL": 100 - cycle,
                })

        self.df = pd.DataFrame(rows)

        self.pipeline = DataPreparationPipeline(

            splitter=DataSplitter(),

            selector=FeatureSelector(
                target_column="RUL",
                drop_columns=["unit_number"],
            ),

            scaler=FeatureScaler(),

        )

    def test_pipeline_runs(self):

        X_train, X_val, y_train, y_val = self.pipeline.prepare(self.df)

        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(X_val, pd.DataFrame)

        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(y_val, pd.Series)

    def test_target_removed(self):

        X_train, _, _, _ = self.pipeline.prepare(self.df)

        self.assertNotIn("RUL", X_train.columns)

    def test_unit_number_removed(self):

        X_train, _, _, _ = self.pipeline.prepare(self.df)

        self.assertNotIn("unit_number", X_train.columns)

    def test_matching_shapes(self):

        X_train, X_val, y_train, y_val = self.pipeline.prepare(self.df)

        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_val), len(y_val))


if __name__ == "__main__":
    unittest.main()