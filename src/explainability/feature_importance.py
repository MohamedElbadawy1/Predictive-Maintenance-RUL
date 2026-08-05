import sys

import pandas as pd

from src.exceptions.custom_exception import CustomException
from src.logger.logger import logger


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance for tree-based machine learning models.
    """

    def get_importance(self,model,feature_names,) -> pd.DataFrame:

        logger.info("Computing feature importance...")

        if not hasattr(model, "feature_importances_"):
            raise CustomException(
                "The provided model does not support feature importance.",
                sys,
            )

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )

        importance_df = (
            importance_df
            .sort_values(
                by="importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        logger.info("Feature importance computed successfully.")

        return importance_df

    def get_top_features(self,importance_df: pd.DataFrame,top_n: int = 50,) -> pd.DataFrame:
        return importance_df.head(top_n)

    def get_zero_importance(self,importance_df: pd.DataFrame,) -> pd.DataFrame:

        return importance_df[
            importance_df["importance"] == 0
        ]

    def get_low_importance(self,importance_df: pd.DataFrame,threshold: float = 0.001,) -> pd.DataFrame:

        return importance_df[
            importance_df["importance"] < threshold
        ]