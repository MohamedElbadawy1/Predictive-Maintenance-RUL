import sys
from pathlib import Path
from typing import List, Optional, Union

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.exceptions.custom_exception import CustomException
from src.logger.logger import logger


class RegimeNormalizer:
    """
    Regime-aware sensor normalization for multi-operating-condition
    C-MAPSS subsets (FD004). The same sensor produces different
    "normal" values under different operating conditions (altitude,
    speed, throttle — the 3 operational settings) — a single global
    StandardScaler conflates that condition-driven variation with actual
    degradation signal. This detects the discrete operating regimes via
    K-Means on the operational settings, then fits a separate
    StandardScaler per regime for the sensor columns.

    Fit ONLY on training engines, exactly like FeatureScaler — both the
    regime detector (KMeans) and every per-regime scaler must never see
    validation or test rows during fitting, or the whole point (avoiding
    leakage) is defeated.

    Example
    -------
    normalizer = RegimeNormalizer(n_regimes=6)
    train_normalized = normalizer.fit_transform(train_df)
    val_normalized = normalizer.transform(val_df)
    test_normalized = normalizer.transform(test_df)  # predict(), never fit again
    """

    def __init__(
        self,
        n_regimes: int = 6,
        operational_columns: Optional[List[str]] = None,
        sensor_columns: Optional[List[str]] = None,
        random_state: int = 42,
    ):

        self.n_regimes = n_regimes
        self.operational_columns = operational_columns or [
            "operational_setting_1", "operational_setting_2", "operational_setting_3",
        ]
        self.sensor_columns = sensor_columns
        self.random_state = random_state

        # Operational settings have wildly different raw scales (e.g. in
        # FD004: setting_1 spans 0-42, setting_2 spans 0-0.84, setting_3
        # spans 60-100) — K-Means is distance-based, so clustering on raw
        # values would let the largest-magnitude setting dominate and the
        # smallest barely matter. Scale before clustering.
        self.setting_scaler = StandardScaler()
        self.kmeans: Optional[KMeans] = None
        self.regime_scalers = {}

        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "RegimeNormalizer":

        if self.sensor_columns is None:
            raise CustomException(
                "sensor_columns must be provided (either at construction "
                "or before calling fit()).",
                sys,
            )

        settings_scaled = self.setting_scaler.fit_transform(df[self.operational_columns])

        self.kmeans = KMeans(
            n_clusters=self.n_regimes,
            random_state=self.random_state,
            n_init=10,
        )
        regimes = self.kmeans.fit_predict(settings_scaled)

        regime_counts = pd.Series(regimes).value_counts().sort_index()
        logger.info(f"Detected regime row counts (train): {regime_counts.to_dict()}")

        self.regime_scalers = {}
        for regime_id in range(self.n_regimes):
            mask = regimes == regime_id
            if mask.sum() == 0:
                logger.warning(f"Regime {regime_id} has zero training rows — skipping.")
                continue
            scaler = StandardScaler()
            scaler.fit(df.loc[mask, self.sensor_columns])
            self.regime_scalers[regime_id] = scaler

        self._is_fitted = True

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        if not self._is_fitted:
            raise CustomException(
                "RegimeNormalizer must be fit before calling transform().",
                sys,
            )

        df = df.copy()

        settings_scaled = self.setting_scaler.transform(df[self.operational_columns])
        regimes = self.kmeans.predict(settings_scaled)  # predict, never fit, on val/test
        df["regime"] = regimes

        normalized = df[self.sensor_columns].to_numpy(copy=True)

        for regime_id, scaler in self.regime_scalers.items():
            mask = regimes == regime_id
            if mask.sum() == 0:
                continue
            normalized[mask] = scaler.transform(df.loc[mask, self.sensor_columns])

        df[self.sensor_columns] = normalized

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        return self.fit(df).transform(df)

    def save(self, path: Union[str, Path]) -> None:

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"RegimeNormalizer saved to {path}")

    @staticmethod
    def load(path: Union[str, Path]) -> "RegimeNormalizer":

        return joblib.load(path)
