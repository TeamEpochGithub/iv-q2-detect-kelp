"""Pretrain pipeline class."""
import time
from dataclasses import dataclass
from typing import Any

import dask.array as da
from sklearn.pipeline import Pipeline

from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator


@dataclass
class PretrainPipeline(Pipeline):
    """Class used to create the pretrain pipeline.

    :param steps: list of steps
    """

    steps: list[Any]

    def __post_init__(self) -> None:
        """Post init function."""
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, Any]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        # Use list comprehension to get the steps
        # if isinstance(self.steps[0], tuple):
        #     return self.steps
        # else:
        return [(str(step), step) for step in self.steps]

    def load_scaler(self, scaler_hash: str) -> None:
        """Load the scaler from the scaler hash.

        :param scaler_hash: The scaler hash
        """
        for step in self.steps:
            if hasattr(step, "load_scaler"):
                step.load_scaler(scaler_hash)

    def save_scaler(self, scaler_hash: str) -> list[tuple[str, Any]]:
        """Save the scaler to the scaler hash.

        :param scaler_hash: The scaler hash
        """
        for step in self.steps:
            if hasattr(step, "save_scaler"):
                step.save_scaler(scaler_hash)

    def fit_transform(self, X: da.Array, y: da.Array | None = None, **fit_params: dict[str, Any]) -> da.Array:
        """Fit and transform the data.

        :param X: Data to fit and transform
        :param y: Target data
        :param fit_params: Fit parameters
        """
        print_section_separator("Pretrain")
        start_time = time.time()
        X = super().fit_transform(X, y, **fit_params)
        logger.info(f"Fitted pretrain pipeline in {time.time() - start_time} seconds")
        return X

    def transform(self, X: da.Array) -> da.Array:
        """Transform the data.

        :param X: Data to transform
        """
        print_section_separator("Pretrain")
        start_time = time.time()
        X = super().transform(X)
        logger.info(f"Transformed pretrain pipeline in {time.time() - start_time} seconds")
        return X
