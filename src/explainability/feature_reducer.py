import json
import sys
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from src.exceptions.custom_exception import CustomException
from src.logger.logger import logger


class FeatureReducer:
    """
    Reduce feature dimensionality using a fitted model's feature
    importance, or an explicit list of features to keep.

    Exactly one reduction strategy must be provided:

    - keep_features      : keep only these exact feature names
    - remove_zero_only    : drop features with importance == 0
    - threshold           : drop features with importance < threshold
    - bottom_percent      : drop the lowest N% of features by importance

    Example
    -------
    reducer = FeatureReducer(threshold=0.001)

    X_train_reduced = reducer.fit_transform(
        X_train,
        importance_df,
    )

    X_val_reduced = reducer.transform(X_val)
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        remove_zero_only: bool = False,
        bottom_percent: Optional[float] = None,
        keep_features: Optional[List[str]] = None,
    ):

        strategies_set = sum(
            [
                threshold is not None,
                remove_zero_only,
                bottom_percent is not None,
                keep_features is not None,
            ]
        )

        if strategies_set != 1:
            raise CustomException(
                "Exactly one of threshold, remove_zero_only, "
                "bottom_percent, or keep_features must be provided.",
                sys,
            )

        self.threshold = threshold
        self.remove_zero_only = remove_zero_only
        self.bottom_percent = bottom_percent
        self.keep_features = keep_features

        self.selected_features_: Optional[List[str]] = None
        self.removed_features_: Optional[List[str]] = None

    def fit(
        self,
        X: pd.DataFrame,
        importance_df: Optional[pd.DataFrame] = None,
    ) -> "FeatureReducer":

        if self.keep_features is not None:

            selected = [
                col for col in X.columns
                if col in set(self.keep_features)
            ]

        else:

            if importance_df is None:
                raise CustomException(
                    "importance_df is required for importance-based "
                    "reduction strategies.",
                    sys,
                )

            selected = self._select_by_importance(
                X.columns,
                importance_df,
            )

        self.selected_features_ = selected
        self.removed_features_ = [
            col for col in X.columns
            if col not in set(selected)
        ]

        logger.info(
            f"FeatureReducer: kept {len(self.selected_features_)} "
            f"features, removed {len(self.removed_features_)}."
        )

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:

        if self.selected_features_ is None:
            raise CustomException(
                "FeatureReducer must be fit before calling transform().",
                sys,
            )

        return X[self.selected_features_]

    def fit_transform(
        self,
        X: pd.DataFrame,
        importance_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:

        return self.fit(X, importance_df).transform(X)

    def save_selected_features(
        self,
        path: Union[str, Path],
    ) -> None:

        if self.selected_features_ is None:
            raise CustomException(
                "FeatureReducer must be fit before saving the "
                "selected feature list.",
                sys,
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "n_selected": len(self.selected_features_),
            "n_removed": len(self.removed_features_),
            "selected_features": self.selected_features_,
            "removed_features": self.removed_features_,
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Selected feature list saved to {path}")

    @staticmethod
    def load_selected_features(
        path: Union[str, Path],
    ) -> List[str]:

        with open(path, "r") as f:
            payload = json.load(f)

        return payload["selected_features"]

    def _select_by_importance(
        self,
        columns: pd.Index,
        importance_df: pd.DataFrame,
    ) -> List[str]:

        if self.remove_zero_only:

            drop = set(
                importance_df.loc[
                    importance_df["importance"] == 0,
                    "feature",
                ]
            )

            return [col for col in columns if col not in drop]

        if self.threshold is not None:

            keep = set(
                importance_df.loc[
                    importance_df["importance"] >= self.threshold,
                    "feature",
                ]
            )

            return [col for col in columns if col in keep]

        if self.bottom_percent is not None:

            n_total = len(importance_df)
            n_drop = int(round(n_total * self.bottom_percent / 100))

            drop = set(
                importance_df
                .sort_values("importance", ascending=True)
                .head(n_drop)["feature"]
            )

            return [col for col in columns if col not in drop]

        raise CustomException(
            "No valid importance-based reduction strategy set.",
            sys,
        )
