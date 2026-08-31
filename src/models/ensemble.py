import json
from pathlib import Path
from typing import Dict, Optional, Union

import joblib
import numpy as np

from src.logger.logger import logger


class EnsembleModel:
    """
    Weighted average ensemble over several already-trained regressors
    (e.g. tuned CatBoost + XGBoost + LightGBM). Same save/load/predict
    shape as BaseTrainer, so it fits the existing project conventions
    rather than being a one-off script pattern.

    Weights default to equal (simple average) but can be set explicitly —
    e.g. weighted by each member's inverse validation MAE, so a more
    accurate member counts for more.

    Example
    -------
    ensemble = EnsembleModel({
        "catboost": catboost_model,
        "xgboost": xgboost_model,
        "lightgbm": lightgbm_model,
    })
    predictions = ensemble.predict(X_val)
    ensemble.save("artifacts/models/ensemble")

    reloaded = EnsembleModel.load("artifacts/models/ensemble")
    """

    def __init__(
        self,
        models: Dict[str, object],
        weights: Optional[Dict[str, float]] = None,
    ):

        if not models:
            raise ValueError("EnsembleModel needs at least one member model.")

        self.models = models

        if weights is None:
            equal_weight = 1.0 / len(models)
            self.weights = {name: equal_weight for name in models}
        else:
            missing = set(models) - set(weights)
            if missing:
                raise ValueError(f"Missing weights for: {missing}")
            total = sum(weights.values())
            self.weights = {name: w / total for name, w in weights.items()}  # normalize to sum to 1

        logger.info(f"EnsembleModel created: {list(self.models.keys())} | weights={self.weights}")

    @classmethod
    def from_inverse_mae(
        cls,
        models: Dict[str, object],
        val_mae: Dict[str, float],
    ) -> "EnsembleModel":
        """
        Weight each member by 1/MAE (a more accurate member — lower MAE —
        gets a larger share of the blend), instead of a plain average.
        """

        missing = set(models) - set(val_mae)
        if missing:
            raise ValueError(f"Missing val_mae for: {missing}")

        inverse = {name: 1.0 / val_mae[name] for name in models}

        return cls(models, weights=inverse)

    def predict(self, X) -> np.ndarray:

        prediction = np.zeros(len(X))

        for name, model in self.models.items():
            prediction += self.weights[name] * np.asarray(model.predict(X))

        return prediction

    def save(self, dir_path: Union[str, Path]) -> None:

        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            joblib.dump(model, dir_path / f"{name}.pkl")

        with open(dir_path / "weights.json", "w") as f:
            json.dump(self.weights, f, indent=2)

        logger.info(f"EnsembleModel saved to {dir_path}")

    @classmethod
    def load(cls, dir_path: Union[str, Path]) -> "EnsembleModel":

        dir_path = Path(dir_path)

        with open(dir_path / "weights.json", "r") as f:
            weights = json.load(f)

        models = {name: joblib.load(dir_path / f"{name}.pkl") for name in weights}

        logger.info(f"EnsembleModel loaded from {dir_path}")

        return cls(models, weights=weights)
