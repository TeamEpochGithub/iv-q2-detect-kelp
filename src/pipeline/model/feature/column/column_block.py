"""Column block pipeline."""
from dataclasses import dataclass
from typing import Any

import dask.array as da
from joblib import hash
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.logging_utils.logger import logger
from src.pipeline.caching.column import CacheColumnBlock


@dataclass
class ColumnBlockPipeline(Pipeline):
    """ColumnBlockPipeline extends the sklearn Pipeline class.

    :param column_block: column block
    :param cache_block: cache block
    """

    column_block: BaseEstimator
    cache_block: CacheColumnBlock | None = None

    def __post_init__(self) -> None:
        """Post init function."""
        self.path = ""
        self.prev_hash = ""
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, BaseEstimator | Pipeline]]:
        """Get the column block pipeline steps.

        :return: list of steps
        """
        steps = []
        if self.column_block:
            steps.append((str(self.column_block), self.column_block))
        if self.cache_block:
            if self.path:
                self.cache_block.set_path(self.path + "/" + str(self.column_block))
            steps.append((str(self.cache_block), self.cache_block))
        return steps

    def fit_transform(self, X: da.Array, y: da.Array | None = None, **fit_params: dict[str, Any]) -> da.Array:
        """Fit and transform the data.

        :param X: The data to fit and transform
        :param y: The target variable
        :param fit_params: The fit parameters
        :return: The transformed data
        """
        logger.debug(f"ColumnBlockPipeline fit_transform: {self.steps}")
        return super().fit_transform(X, y, **fit_params)

    def set_path(self, path: str) -> None:
        """Set the path.

        :param path: path
        """
        self.path = path
        # Update the steps in the pipeline after changing the path
        self.steps = self._get_steps()

    def set_hash(self, prev_hash: str = "") -> str:
        """set_hash function sets the hash for the pipeline.

        :param prev_hash: previous hash
        :return: hash
        """
        column_block_hash = hash(str(self.column_block) + prev_hash)

        self.prev_hash = column_block_hash

        return column_block_hash
