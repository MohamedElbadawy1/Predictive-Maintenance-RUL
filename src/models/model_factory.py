from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from src.exceptions.custom_exception import CustomException

import sys


class ModelFactory:
    """
    Factory responsible for creating regression models.
    """

    MODELS = {

        "random_forest": RandomForestRegressor,

        "xgboost": XGBRegressor,

        "lightgbm": LGBMRegressor,

        "catboost": CatBoostRegressor,
    }

    DEFAULT_PARAMETERS = {

        "random_forest": {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": -1,
        },

        "xgboost": {
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_estimators": 100,
        },

        "lightgbm": {
            "random_state": 42,
            "n_estimators": 100,
        },

        "catboost": {
            "random_state": 42,
            "verbose": False,
        },
    }

    @classmethod
    def create(
        cls,
        model_name: str,
        **kwargs,
    ):

        model_name = model_name.lower()

        if model_name not in cls.MODELS:

            raise CustomException(
                f"Unsupported model '{model_name}'.",
                sys,
            )

        params = cls.DEFAULT_PARAMETERS[model_name].copy()

        params.update(kwargs)

        return cls.MODELS[model_name](**params)

    @classmethod
    def available_models(cls):

        return list(cls.MODELS.keys())