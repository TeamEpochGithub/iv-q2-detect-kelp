"""A pipeline step that caches a column to disk."""
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import dask.array as da
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_utils.logger import logger
from src.pipeline.caching.util.error import CachePipelineError
from src.pipeline.caching.util.store_raw import store_raw


@dataclass
class CacheColumnPipeline(BaseEstimator, TransformerMixin):
    """The caching column pipeline is responsible for loading and storing individual columns to disk.

    :param data_path: The path to the data
    :param column: The column to store
    """

    data_path: Path
    column: int = -1

    def __post_init__(self) -> None:
        """Check if the data path is defined."""
        if not self.data_path:
            raise CachePipelineError("data_path is required")

    def fit(self, X: da.Array, y: da.Array | None = None) -> Self:
        """Do nothing. This method only exists for compatibility with Scikit-Learn Pipelines.

        :param X: The data to fit
        :param y: The target variable
        :return: The same pipeline
        """
        return self

    def transform(self, X: da.Array, y: da.Array | None = None) -> da.Array:
        """Cache the column.

        :param X: The data to transform
        :param y: The target variable
        :return: The transformed data
        """
        # Load or store the data column
        logger.info("Loading or storing column")
        column = store_raw(self.data_path.as_posix(), X[:, self.column])

        # Create the new array
        return da.concatenate([X[:, : self.column], column[:, None]], axis=1)
