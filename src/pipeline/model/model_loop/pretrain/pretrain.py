"""Pretrain pipeline class."""
import time
from typing import Any

import dask.array as da
from sklearn.pipeline import Pipeline

from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator


class PretrainPipeline(Pipeline):
    """Class used to create the pretrain pipeline.

    :param steps: list of steps
    """

    def __init__(self, steps: list[Any]) -> None:
        """Initialize the PretrainPipeline.

        :param steps: list of steps
        """
        self.steps = steps
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, Any]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        return [(step.__class__.__name__, step) for step in self.steps]

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
