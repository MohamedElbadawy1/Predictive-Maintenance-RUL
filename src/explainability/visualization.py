from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd

from src.config.config import PLOTS_DIR
from src.logger.logger import logger


class ExplainabilityVisualizer:
    """
    Visualization utilities for model explainability.
    """

    def __init__(
        self,
        save_dir: Optional[Union[str, Path]] = None,
    ):

        self.save_dir = Path(save_dir) if save_dir is not None else PLOTS_DIR
        self.save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        save: bool = True,
    ):

        plot_df = importance_df.head(top_n)

        plt.figure(figsize=(10, 7))

        plt.barh(
            plot_df["feature"][::-1],
            plot_df["importance"][::-1],
        )

        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"Top {top_n} Feature Importance")

        plt.tight_layout()

        if save:

            filename = (
                self.save_dir
                / f"top_{top_n}_feature_importance.png"
            )

            plt.savefig(
                filename,
                dpi=300,
            )

            logger.info(
                f"Saved plot to {filename}"
            )

        plt.show()