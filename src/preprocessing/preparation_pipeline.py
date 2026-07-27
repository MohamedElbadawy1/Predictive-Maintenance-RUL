from typing import Tuple

import pandas as pd

from src.logger.logger import logger
from src.preprocessing.data_splitter import DataSplitter
from src.preprocessing.feature_scaler import FeatureScaler
from src.preprocessing.feature_selector import FeatureSelector


class DataPreparationPipeline:
    """
    Complete data preparation pipeline for traditional
    machine learning models.
    """

    def __init__(
        self,
        splitter: DataSplitter,
        selector: FeatureSelector,
        scaler: FeatureScaler,
    ):
        self.splitter = splitter
        self.selector = selector
        self.scaler = scaler

    def prepare(
        self,
        df: pd.DataFrame,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
    ]:

        logger.info("Starting Data Preparation Pipeline...")

        train_df, val_df = self.splitter.split(df)
        X_train, y_train = self.selector.transform(train_df)
        X_val, y_val = self.selector.transform(val_df)
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        logger.info("Data Preparation completed successfully.")

        return (
            X_train,
            X_val,
            y_train,
            y_val,
        )