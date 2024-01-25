"""TransformationPipeline."""
import time
from dataclasses import dataclass
from typing import Any

import dask.array as da
from joblib import hash
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator


@dataclass
class TransformationPipeline(Pipeline):
    """TransformationPipeline class extends the sklearn Pipeline class.

    :param transformations: list of transformations
    """

    transformations: list[BaseEstimator | Pipeline]

    def __post_init__(self) -> None:
        """Post init function."""
        self.set_hash("")
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, BaseEstimator | Pipeline]]:
        """Get the transformation pipeline steps.

        :return: list of steps
        """
        # Use list comprehension to get the steps
        return [(str(transformation), transformation) for transformation in self.transformations]

    def set_hash(self, prev_hash: str) -> str:
        """set_hash function sets the hash for the pipeline.

        :param prev_hash: previous hash
        :return: hash
        """
        transformation_hash = prev_hash
        for transformation in self.transformations:
            transformation_hash = hash(str(transformation) + transformation_hash)

        self.prev_hash = transformation_hash
        return transformation_hash

    def fit_transform(self, X: da.Array, y: da.Array | None = None, **fit_params: dict[str, Any]) -> da.Array:
        """Fit and transform the data.

        :param X: Data to fit and transform
        :param y: Target data
        :param fit_params: Fit parameters
        :return: Fitted and transformed data
        """
        print_section_separator("Preprocessing - Transformations")
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
