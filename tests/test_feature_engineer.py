import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.preprocessing.feature_engineer import FeatureEngineer
from src.exceptions.custom_exception import CustomException


class TestFeatureEngineer(unittest.TestCase):

    def setUp(self):
        """
        Create a small synthetic dataset for testing.
        """

        self.df = pd.DataFrame(
            {
                "unit_number": [1, 1, 1, 2, 2, 2],
                "time_in_cycles": [1, 2, 3, 1, 2, 3],
                "sensor_1": [10, 20, 30, 100, 200, 300],
                "sensor_2": [1, 2, 3, 10, 20, 30],
            }
        )

        self.engineer = FeatureEngineer(
            sensor_columns=["sensor_1", "sensor_2"],
            rolling_window=2,
            lags=[1, 2],
        )

    ####################################################################
    # Basic Tests
    ####################################################################

    def test_transform_returns_new_dataframe(self):
        """
        transform() should return a new dataframe.
        """

        new_df = self.engineer.transform(self.df)

        self.assertIsNot(self.df, new_df)

    def test_original_dataframe_not_modified(self):
        """
        Original dataframe should remain unchanged.
        """

        original = self.df.copy(deep=True)

        self.engineer.transform(self.df)

        assert_frame_equal(self.df, original)

    def test_number_of_rows_not_changed(self):

        new_df = self.engineer.transform(self.df)

        self.assertEqual(len(self.df), len(new_df))

    ####################################################################
    # Generated Columns
    ####################################################################

    def test_generated_columns_exist(self):

        new_df = self.engineer.transform(self.df)

        expected_columns = [
            "sensor_1_roll_mean_2",
            "sensor_1_roll_std_2",
            "sensor_1_lag_1",
            "sensor_1_lag_2",
            "sensor_1_diff",
            "sensor_2_roll_mean_2",
            "sensor_2_roll_std_2",
            "sensor_2_lag_1",
            "sensor_2_lag_2",
            "sensor_2_diff",
        ]

        for column in expected_columns:
            self.assertIn(column, new_df.columns)

    ####################################################################
    # GroupBy Tests
    ####################################################################

    def test_lag_is_calculated_per_engine(self):
        """
        First row of every engine should have NaN lag.
        """

        new_df = self.engineer.transform(self.df)

        first_rows = (
            new_df
            .groupby("unit_number")
            .head(1)
        )

        self.assertTrue(first_rows["sensor_1_lag_1"].isna().all())
        self.assertTrue(first_rows["sensor_2_lag_1"].isna().all())

    def test_diff_is_calculated_per_engine(self):
        """
        First row of every engine should have NaN diff.
        """

        new_df = self.engineer.transform(self.df)

        first_rows = (
            new_df
            .groupby("unit_number")
            .head(1)
        )

        self.assertTrue(first_rows["sensor_1_diff"].isna().all())
        self.assertTrue(first_rows["sensor_2_diff"].isna().all())

    ####################################################################
    # Validation Tests
    ####################################################################

    def test_invalid_window(self):

        with self.assertRaises(CustomException):

            FeatureEngineer(
                sensor_columns=["sensor_1"],
                rolling_window=0,
            ).transform(self.df)

    def test_invalid_lag(self):

        with self.assertRaises(CustomException):

            FeatureEngineer(
                sensor_columns=["sensor_1"],
                lags=[-1],
            ).transform(self.df)

    def test_missing_sensor_column(self):

        with self.assertRaises(CustomException):

            FeatureEngineer(
                sensor_columns=["sensor_100"],
            ).transform(self.df)

    def test_missing_group_column(self):

        with self.assertRaises(CustomException):

            FeatureEngineer(
                sensor_columns=["sensor_1"],
                group_column="engine",
            ).transform(self.df)

    def test_empty_dataframe(self):

        empty_df = pd.DataFrame()

        with self.assertRaises(CustomException):

            self.engineer.transform(empty_df)


if __name__ == "__main__":
    unittest.main()