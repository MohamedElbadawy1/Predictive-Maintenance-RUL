import time
from typing import Dict, Optional, Tuple

import optuna
import pandas as pd
from optuna.samplers import TPESampler

from src.evaluation.evaluator import RegressionEvaluator
from src.models.base_trainer import BaseTrainer
from src.models.model_factory import ModelFactory
from src.logger.logger import logger

# Each entry: (kind, low, high)
#   "int"       -> trial.suggest_int(low, high)
#   "float"     -> trial.suggest_float(low, high)
#   "float_log" -> trial.suggest_float(low, high, log=True)

CATBOOST_SEARCH_SPACE: Dict[str, Tuple[str, float, float]] = {
    "depth": ("int", 4, 10),
    "learning_rate": ("float_log", 0.01, 0.3),
    "iterations": ("int", 200, 1500),
    "l2_leaf_reg": ("float", 1.0, 10.0),
    "subsample": ("float", 0.5, 1.0),
    "random_strength": ("float", 0.0, 10.0),
}

XGBOOST_SEARCH_SPACE: Dict[str, Tuple[str, float, float]] = {
    "max_depth": ("int", 3, 10),
    "learning_rate": ("float_log", 0.01, 0.3),
    "n_estimators": ("int", 200, 1500),
    "reg_lambda": ("float", 1.0, 10.0),
    "subsample": ("float", 0.5, 1.0),
    "colsample_bytree": ("float", 0.5, 1.0),
}

LIGHTGBM_SEARCH_SPACE: Dict[str, Tuple[str, float, float]] = {
    "max_depth": ("int", 3, 10),
    "num_leaves": ("int", 15, 255),
    "learning_rate": ("float_log", 0.01, 0.3),
    "n_estimators": ("int", 200, 1500),
    "reg_lambda": ("float", 1.0, 10.0),
    "subsample": ("float", 0.5, 1.0),
    "colsample_bytree": ("float", 0.5, 1.0),
}

# Backward-compat alias — existing code imports this exact name.
DEFAULT_SEARCH_SPACE = CATBOOST_SEARCH_SPACE

_MODEL_DEFAULTS = {
    "catboost": {
        "search_space": CATBOOST_SEARCH_SPACE,
        "fixed_params": {"verbose": False},
    },
    "xgboost": {
        "search_space": XGBOOST_SEARCH_SPACE,
        "fixed_params": {"objective": "reg:squarederror"},
    },
    "lightgbm": {
        "search_space": LIGHTGBM_SEARCH_SPACE,
        "fixed_params": {"verbose": -1},
    },
}


class ModelTuner:
    """
    Hyperparameter optimization for any model in ModelFactory, using
    Optuna. Same mechanism for every model family — only the search
    space and a couple of fixed params (e.g. CatBoost's `verbose=False`
    vs. LightGBM's `verbose=-1`) differ, defined once in
    `_MODEL_DEFAULTS` rather than duplicated per model.

    Every trial trains the chosen model on the same train/validation
    split with a different hyperparameter combination, minimizing
    validation MAE. RMSE, R2, and training time are recorded per trial
    as secondary metrics. Every trial also gets automatically logged to
    MLflow (one run per trial) via BaseTrainer's built-in tracking.

    Example
    -------
    tuner = ModelTuner("xgboost", X_train, y_train, X_val, y_val)
    study = tuner.run(n_trials=30)
    best_params = tuner.best_params()
    trials_df = tuner.get_trials_dataframe()
    """

    def __init__(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        search_space: Optional[Dict[str, Tuple[str, float, float]]] = None,
        fixed_params: Optional[Dict] = None,
        random_state: int = 42,
        track_mlflow: bool = True,
    ):

        model_name = model_name.lower()
        if model_name not in _MODEL_DEFAULTS:
            raise ValueError(
                f"No tuner defaults for '{model_name}'. "
                f"Supported: {list(_MODEL_DEFAULTS.keys())}"
            )

        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        defaults = _MODEL_DEFAULTS[model_name]
        self.search_space = search_space or defaults["search_space"]
        self.fixed_params = {
            "random_state": random_state,
            **defaults["fixed_params"],
            **(fixed_params or {}),
        }
        self.random_state = random_state
        self.track_mlflow = track_mlflow

        self.evaluator = RegressionEvaluator()
        self.study: Optional[optuna.Study] = None
        self.trial_records = []

    def run(
        self,
        n_trials: int = 40,
        timeout: Optional[int] = None,
        show_progress_bar: bool = False,
    ) -> optuna.Study:

        sampler = TPESampler(seed=self.random_state)

        self.study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=f"{self.model_name}_rul_optimization",
        )

        logger.info(
            f"Starting Optuna search for {self.model_name}: {n_trials} trials, "
            f"search space = {list(self.search_space.keys())}"
        )

        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress_bar,
        )

        logger.info(
            f"[{self.model_name}] Optimization finished. "
            f"Best MAE = {self.study.best_value:.4f} at trial {self.study.best_trial.number}."
        )

        return self.study

    def get_trials_dataframe(self) -> pd.DataFrame:

        return pd.DataFrame(self.trial_records)

    def best_params(self) -> Dict:

        self._require_study()

        return {**self.fixed_params, **self.study.best_params}

    def best_trial_metrics(self) -> Dict:

        self._require_study()

        best_trial = self.study.best_trial

        return {
            "MAE": best_trial.value,
            "RMSE": best_trial.user_attrs["RMSE"],
            "R2": best_trial.user_attrs["R2"],
            "Training Time (s)": best_trial.user_attrs["training_time"],
        }

    def _objective(self, trial: optuna.Trial) -> float:

        params = self._suggest_params(trial)
        model_params = {**self.fixed_params, **params}

        model = ModelFactory.create(self.model_name, **model_params)
        trainer = BaseTrainer(
            model,
            track_mlflow=self.track_mlflow,
            run_name=f"{self.model_name}_optuna_trial_{trial.number}",
            tags={"model_family": self.model_name, "stage": "hyperparameter_search"},
        )

        start = time.time()
        # BaseTrainer.train() only auto-logs metrics when val data is
        # passed — using that here means every trial is fully logged
        # (params + metrics + model) with no separate logging call.
        metrics = trainer.train(self.X_train, self.y_train, self.X_val, self.y_val)
        training_time = time.time() - start

        trial.set_user_attr("RMSE", metrics["RMSE"])
        trial.set_user_attr("R2", metrics["R2"])
        trial.set_user_attr("training_time", round(training_time, 2))

        self.trial_records.append(
            {
                "trial": trial.number,
                **params,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "Training Time (s)": round(training_time, 2),
            }
        )

        logger.info(
            f"[{self.model_name}] Trial {trial.number} | MAE={metrics['MAE']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f} | R2={metrics['R2']:.4f} | "
            f"Time={training_time:.2f}s | params={params}"
        )

        return metrics["MAE"]

    def _suggest_params(self, trial: optuna.Trial) -> Dict:

        params = {}

        for name, (kind, low, high) in self.search_space.items():

            if kind == "int":
                params[name] = trial.suggest_int(name, int(low), int(high))
            elif kind == "float":
                params[name] = trial.suggest_float(name, low, high)
            elif kind == "float_log":
                params[name] = trial.suggest_float(name, low, high, log=True)
            else:
                raise ValueError(f"Unknown search space entry type: {kind}")

        return params

    def _require_study(self) -> None:

        if self.study is None:
            raise RuntimeError(
                "No study found. Call run() before requesting results."
            )


class CatBoostTuner(ModelTuner):
    """Thin convenience wrapper — identical to ModelTuner("catboost", ...).
    Kept for backward compatibility with existing scripts."""

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        search_space: Optional[Dict[str, Tuple[str, float, float]]] = None,
        fixed_params: Optional[Dict] = None,
        random_state: int = 42,
        track_mlflow: bool = True,
    ):
        super().__init__(
            "catboost", X_train, y_train, X_val, y_val,
            search_space=search_space, fixed_params=fixed_params,
            random_state=random_state, track_mlflow=track_mlflow,
        )


class XGBoostTuner(ModelTuner):
    """Convenience wrapper — identical to ModelTuner("xgboost", ...)."""

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        search_space: Optional[Dict[str, Tuple[str, float, float]]] = None,
        fixed_params: Optional[Dict] = None,
        random_state: int = 42,
        track_mlflow: bool = True,
    ):
        super().__init__(
            "xgboost", X_train, y_train, X_val, y_val,
            search_space=search_space, fixed_params=fixed_params,
            random_state=random_state, track_mlflow=track_mlflow,
        )


class LightGBMTuner(ModelTuner):
    """Convenience wrapper — identical to ModelTuner("lightgbm", ...)."""

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        search_space: Optional[Dict[str, Tuple[str, float, float]]] = None,
        fixed_params: Optional[Dict] = None,
        random_state: int = 42,
        track_mlflow: bool = True,
    ):
        super().__init__(
            "lightgbm", X_train, y_train, X_val, y_val,
            search_space=search_space, fixed_params=fixed_params,
            random_state=random_state, track_mlflow=track_mlflow,
        )
