"""ColumnPipeline is the class used to create the column pipeline."""
import time
from dataclasses import dataclass
from typing import Any

import dask.array as da
from sklearn.pipeline import Pipeline

from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator
from src.pipeline.model.feature.column.column_block import ColumnBlockPipeline


@dataclass
class ColumnPipeline(Pipeline):
    """ColumnPipeline is the class used to create the column pipeline.

    :param columns: The columns
    """

    columns: list[ColumnBlockPipeline]

    def __post_init__(self) -> None:
        """Post init function."""
        self.path = ""
        self.set_hash("")
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, ColumnBlockPipeline]]:
        """Get the steps in the pipeline.

        :return: List of steps in the pipeline
        """
        steps = []
        for column in self.columns:
            if self.path:
                column.set_path(self.path)
            steps.append((str(column), column))
        return steps

    def fit_transform(self, X: da.Array, y: da.Array | None = None, **fit_params: dict[str, Any]) -> da.Array:
        """Fit and transform the data.

        :param X: Data to fit and transform
        :param y: Target data
        :param fit_params: Fit parameters
        :return: Fitted and transformed data
        """
        print_section_separator("Preprocessing - Columns")
        logger.info("Fitting column pipeline")
        start_time = time.time()
        X = super().fit_transform(X, y, **fit_params)
        logger.info(f"Fitted column pipeline in {time.time() - start_time} seconds")
        return X

    def transform(self, X: da.Array) -> da.Array:
        """Transform the data.

        :param X: Data to transform
        :return: Transformed data
        """
        print_section_separator("Preprocessing - Columns")
        logger.info("Transforming column pipeline")
        start_time = time.time()
        X = super().transform(X)
        logger.info(f"Transform of columns complete in {time.time() - start_time} seconds")
        return X

    def set_path(self, path: str) -> None:
        """Set the path.

        :param path: path
        """
        self.path = path
        # Update the steps in the pipeline after changing the path
        self.steps = self._get_steps()

    def set_hash(self, prev_hash: str = "") -> str:
        """Set the hash.

        :param prev_hash: Previous hash
        :return: Hash
        """
        column_hash = prev_hash
        for column in self.columns:
            column_hash = column.set_hash(column_hash)

        self.prev_hash = column_hash
        return column_hash
