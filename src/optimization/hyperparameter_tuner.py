import time
from typing import Callable, Dict, Optional, Tuple

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
DEFAULT_SEARCH_SPACE: Dict[str, Tuple[str, float, float]] = {
    "depth": ("int", 4, 10),
    "learning_rate": ("float_log", 0.01, 0.3),
    "iterations": ("int", 200, 1500),
    "l2_leaf_reg": ("float", 1.0, 10.0),
    "subsample": ("float", 0.5, 1.0),
    "random_strength": ("float", 0.0, 10.0),
}


class CatBoostTuner:
    """
    Hyperparameter optimization for CatBoost using Optuna.

    Every trial trains CatBoost on the same train/validation split with a
    different hyperparameter combination, minimizing validation MAE.
    RMSE, R2, and training time are recorded per trial as secondary metrics.

    Example
    -------
    tuner = CatBoostTuner(X_train, y_train, X_val, y_val)

    study = tuner.run(n_trials=40)

    best_params = tuner.best_params()
    trials_df = tuner.get_trials_dataframe()
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        search_space: Optional[Dict[str, Tuple[str, float, float]]] = None,
        fixed_params: Optional[Dict] = None,
        random_state: int = 42,
    ):

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.fixed_params = fixed_params or {
            "random_state": random_state,
            "verbose": False,
        }
        self.random_state = random_state

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
            study_name="catboost_rul_optimization",
        )

        logger.info(
            f"Starting Optuna search: {n_trials} trials, "
            f"search space = {list(self.search_space.keys())}"
        )

        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress_bar,
        )

        logger.info(
            f"Optimization finished. Best MAE = {self.study.best_value:.4f} "
            f"at trial {self.study.best_trial.number}."
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

        model = ModelFactory.create("catboost", **model_params)
        trainer = BaseTrainer(model)

        start = time.time()
        trainer.train(self.X_train, self.y_train)
        training_time = time.time() - start

        predictions = trainer.predict(self.X_val)
        metrics = self.evaluator.evaluate(self.y_val, predictions)

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
            f"Trial {trial.number} | MAE={metrics['MAE']:.4f} | "
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
