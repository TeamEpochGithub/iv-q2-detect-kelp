import time
from typing import Any

import dask.array as da
from sklearn.pipeline import Pipeline

from src.logging_utils.logger import logger
from src.pipeline.model.feature.column.column_block import ColumnBlockPipeline


class ColumnPipeline(Pipeline):
    """ColumnPipeline is the class used to create the column pipeline.

    :param columns: The columns
    """

    def __init__(self, columns: list[ColumnBlockPipeline]) -> None:
        """Initialize the class.

        :param columns: The columns
        """
        self.columns = columns
        self.path = ""
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, ColumnBlockPipeline]]:
        """Get the steps in the pipeline

        :return: List of steps in the pipeline
        """
        steps = []
        for column in self.columns:
            if self.path:
                column.set_path(self.path)
            steps.append((str(column), column))
        return steps

    def fit_transform(self, X: da.Array, y: da.Array | None = None, **fit_params: dict[str, Any]) -> da.Array:
        """Fit and transform the data

        :param X: Data to fit and transform
        :param y: Target data
        :param fit_params: Fit parameters
        :return: Fitted and transformed data
        """
        logger.info("Fitting column pipeline")
        start_time = time.time()
        X = super().fit_transform(X, y, **fit_params)
        logger.info(f"Fitted column pipeline in {time.time() - start_time} seconds")
        return X

    def __repr__(self) -> str:
        """Representation of the ColumnPipeline

        :return: String representation of the ColumnPipeline
        """
        return f"ColumnPipeline(columns={self.columns})"

    def __str__(self) -> str:
        """String representation of the ColumnPipeline

        :return: String representation of the ColumnPipeline
        """
        return "ColumnPipeline"

    def set_path(self, path: str) -> None:
        """Set the path

        :param path: path
        """
        self.path = path
        # Update the steps in the pipeline after changing the path
        self.steps = self._get_steps()
