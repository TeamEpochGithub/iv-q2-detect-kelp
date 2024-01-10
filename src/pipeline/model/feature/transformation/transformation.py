"""TransformationPipeline."""
import time
from typing import Any

import dask.array as da
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator


class TransformationPipeline(Pipeline):
    """TransformationPipeline class extends the sklearn Pipeline class.

    :param transformations: list of transformations
    """

    def __init__(self, transformations: list[BaseEstimator]) -> None:
        """Initialize the TransformationPipeline.

        :param transformations: list of transformations
        """
        self.transformations = transformations
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, BaseEstimator | Pipeline]]:
        """Get the transformation pipeline steps.

        :return: list of steps
        """
        # Use list comprehension to get the steps
        return [(str(transformation), transformation) for transformation in self.transformations]

    def fit_transform(self, X: da.Array, y: da.Array | None = None, **fit_params: dict[str, Any]) -> da.Array:
        """Fit and transform the data.

        :param X: Data to fit and transform
        :param y: Target data
        :param fit_params: Fit parameters
        :return: Fitted and transformed data
        """
        logger.info("Fitting transformation pipeline")
        start_time = time.time()
        X = super().fit_transform(X, y, **fit_params)
        logger.info(f"Fitted complete transformation pipeline in {time.time() - start_time} seconds")
        return X

    def transform(self, X: da.Array) -> da.Array:
        """Transform the data.

        :param X: Data to transform
        :return: Transformed data
        """
        print_section_separator("Preprocessing - Transformation")
        logger.info("Transforming transformation pipeline")
        start_time = time.time()
        X = super().transform(X)
        logger.info(f"Transform of transformations complete in {time.time() - start_time} seconds")
        return X

    def __str__(self) -> str:
        """__str__ returns string representation of the TransformationPipeline.

        :return: String representation of the TransformationPipeline
        """
        return "TransformationPipeline"
