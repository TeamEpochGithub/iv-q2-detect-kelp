"""PostProcessingPipeline class."""
import time
from dataclasses import dataclass
from typing import Any

import dask.array as da
from sklearn.pipeline import Pipeline

from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator


@dataclass
class PostProcessingPipeline(Pipeline):
    """PostProcessingPipeline is the class used to create the post processing pipeline."""

    post_processing_steps: list[Any]

    def __post_init__(self) -> None:
        """Post init function."""
        self.set_hash("")
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, Pipeline]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        # TODO(Jasper): Implement post processing pipeline steps
        return [(str(step), step) for step in self.post_processing_steps]

    def set_hash(self, prev_hash: str) -> str:
        """Set the hash.

        :param prev_hash: Previous hash
        :return: Hash
        """
        # TODO(Jasper): Implement post processing pipeline hash
        return prev_hash

    def fit_transform(self, X: da.Array, y: da.Array | None = None, **fit_params: dict[str, Any]) -> da.Array:
        """Fit and transform the data.

        :param X: Data to fit and transform
        :param y: Target data
        :param fit_params: Fit parameters
        """
        print_section_separator("Postprocessing")
        start_time = time.time()
        X = super().fit_transform(X, y, **fit_params)
        logger.info(f"Fitted postprocessing pipeline in {time.time() - start_time} seconds")
        return X

    def transform(self, X: da.Array) -> da.Array:
        """Transform the data.

        :param X: Data to transform
        """
        print_section_separator("Postprocessing")
        start_time = time.time()
        X = super().transform(X)
        logger.info(f"Transformed postprocessing pipeline in {time.time() - start_time} seconds")
        return X
