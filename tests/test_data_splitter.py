import unittest

import pandas as pd

from src.preprocessing.data_splitter import DataSplitter
from src.exceptions.custom_exception import CustomException


class TestDataSplitter(unittest.TestCase):

    def setUp(self):
        rows = []

        for engine in range(1, 11):
            for cycle in range(1, 6):
                rows.append({
                    "unit_number": engine,
                    "time_in_cycles": cycle,
                    "sensor_1": engine * cycle,
                    "RUL": 100 - cycle
                })

        self.df = pd.DataFrame(rows)

    def test_split_returns_two_dataframes(self):

        splitter = DataSplitter(test_size=0.2)

        train_df, val_df = splitter.split(self.df)

        self.assertIsInstance(train_df, pd.DataFrame)
        self.assertIsInstance(val_df, pd.DataFrame)

    def test_no_engine_overlap(self):

        splitter = DataSplitter(test_size=0.2)

        train_df, val_df = splitter.split(self.df)

        train_engines = set(train_df.unit_number.unique())
        val_engines = set(val_df.unit_number.unique())

        self.assertTrue(train_engines.isdisjoint(val_engines))

    def test_all_rows_preserved(self):

        splitter = DataSplitter(test_size=0.2)

        train_df, val_df = splitter.split(self.df)

        self.assertEqual(
            len(self.df),
            len(train_df) + len(val_df)
        )

    def test_empty_dataframe(self):

        splitter = DataSplitter()

        with self.assertRaises(CustomException):
            splitter.split(pd.DataFrame())

    def test_missing_engine_column(self):

        splitter = DataSplitter()

        df = self.df.drop(columns="unit_number")

        with self.assertRaises(CustomException):
            splitter.split(df)

    def test_invalid_test_size(self):

        with self.assertRaises(CustomException):

            splitter = DataSplitter(test_size=2)

            splitter.split(self.df)


if __name__ == "__main__":
    unittest.main()