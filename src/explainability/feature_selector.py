import re
from typing import Dict, Iterable, List

from src.logger.logger import logger

TIME_COLUMN = "time_in_cycles"

LAG_PATTERN = re.compile(r"_lag_\d+$")
ROLLING_PATTERN = re.compile(r"_roll_(mean|std)_\d+$")
DIFF_PATTERN = re.compile(r"_diff$")


class FeatureCategorySelector:
    """
    Group engineered features into categories based on their naming
    convention, and include/exclude whole categories at once.

    Categories
    ----------
    - "time"    : time_in_cycles
    - "raw"     : raw sensor / operational setting columns
    - "lag"     : *_lag_1, *_lag_2, ...
    - "rolling" : *_roll_mean_*, *_roll_std_*
    - "diff"    : *_diff (rate-of-change features)

    Example
    -------
    # Only raw + lag features
    features = FeatureCategorySelector.only(
        X_train.columns,
        categories=["raw", "lag"],
    )

    # Everything except rate-of-change features
    features = FeatureCategorySelector.exclude(
        X_train.columns,
        categories=["diff"],
    )
    """

    CATEGORIES = ("time", "raw", "lag", "rolling", "diff")

    @classmethod
    def categorize(
        cls,
        columns: Iterable[str],
    ) -> Dict[str, List[str]]:

        categories: Dict[str, List[str]] = {
            "time": [],
            "raw": [],
            "lag": [],
            "rolling": [],
            "diff": [],
        }

        for col in columns:

            if col == TIME_COLUMN:
                categories["time"].append(col)
            elif LAG_PATTERN.search(col):
                categories["lag"].append(col)
            elif ROLLING_PATTERN.search(col):
                categories["rolling"].append(col)
            elif DIFF_PATTERN.search(col):
                categories["diff"].append(col)
            else:
                categories["raw"].append(col)

        return categories

    @classmethod
    def only(
        cls,
        columns: Iterable[str],
        categories: List[str],
    ) -> List[str]:

        cls._validate_categories(categories)

        grouped = cls.categorize(columns)

        keep = set()
        for category in categories:
            keep.update(grouped[category])

        selected = [col for col in columns if col in keep]

        logger.info(
            f"FeatureCategorySelector.only({categories}): "
            f"kept {len(selected)} of {len(list(columns))} features."
        )

        return selected

    @classmethod
    def exclude(
        cls,
        columns: Iterable[str],
        categories: List[str],
    ) -> List[str]:

        cls._validate_categories(categories)

        columns = list(columns)
        grouped = cls.categorize(columns)

        drop = set()
        for category in categories:
            drop.update(grouped[category])

        selected = [col for col in columns if col not in drop]

        logger.info(
            f"FeatureCategorySelector.exclude({categories}): "
            f"dropped {len(drop)}, kept {len(selected)} features."
        )

        return selected

    @classmethod
    def _validate_categories(
        cls,
        categories: List[str],
    ) -> None:

        unknown = set(categories) - set(cls.CATEGORIES)

        if unknown:
            raise ValueError(
                f"Unknown feature categories: {unknown}. "
                f"Valid categories are: {cls.CATEGORIES}"
            )
