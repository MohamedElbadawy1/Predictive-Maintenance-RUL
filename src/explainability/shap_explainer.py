import shap

from src.logger.logger import logger


class SHAPExplainer:

    def __init__(
        self,
        model,
    ):

        self.model = model

        self.explainer = shap.TreeExplainer(model)

    def compute(
        self,
        X,
    ):

        logger.info("Computing SHAP values...")

        return self.explainer.shap_values(X)

    def summary_plot(
        self,
        shap_values,
        X,
    ):

        shap.summary_plot(
            shap_values,
            X,
        )

    def waterfall_plot(
        self,
        shap_values,
        index=0,
    ):

        shap.plots.waterfall(
            shap_values[index]
        )

    def dependence_plot(
        self,
        feature,
        shap_values,
        X,
    ):

        shap.dependence_plot(
            feature,
            shap_values,
            X,
        )