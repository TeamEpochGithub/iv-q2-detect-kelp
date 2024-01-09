"""The caching column block is responsible for loading and storing individual columns to disk."""
from typing import Self

import dask.array as da
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_utils.logger import logger
from src.pipeline.caching.util.error import CachePipelineError
from src.pipeline.caching.util.store_raw import store_raw


class CacheColumnBlock(BaseEstimator, TransformerMixin):
    """The caching column block is responsible for loading and storing individual columns to disk.

    :param data_path: The path to the data
    :param column: The column to store
    """

    def __init__(self, data_path: str | None = None, column: int = -1) -> None:
        """Initialize the caching column block.

        :param data_path: The path to the data
        :param column: The column to store
        """
        # Set values to self
        self.data_path = data_path
        self.column = column

    def fit(self, X: da.Array, y: da.Array | None = None) -> Self:
        """Fit the transformer.

        :param X: The data to fit
        :param y: The target variable
        :return: The fitted transformer
        """
        return self

    def transform(self, X: da.Array, y: da.Array | None = None) -> da.Array:
        """Transform the data.

        :param X: The data to transform
        :param y: The target variable
        :return: The transformed data
        """
        # Check if the data path is set
        if not self.data_path:
            if not X:
                logger.error("data_paths are required")
                raise CachePipelineError("data_path is required")

            return X

        # Load or store the data column
        column = store_raw(self.data_path, X[:, self.column])

        # Create the new array
        X_new = da.concatenate([X[:, : self.column], column[:, None]], axis=1)
        return X_new.rechunk()

    def get_data_path(self) -> str | None:
        """Get the data path.

        :return: The data path
        """
        return self.data_path

    def set_path(self, data_path: str) -> None:
        """Override the data path.

        :param data_path: The new data path
        """
        self.data_path = data_path

    def __str__(self) -> str:
        """__str__ returns string representation of the CacheColumnBlock.

        :return: String representation of the CacheColumnBlock
        """
        return "CacheColumnBlock"

    def __repr__(self) -> str:
        """__repr__ returns full representation of the CacheColumnBlock.

        :return: Representation of the CacheColumnBlock
        """
        return f"CacheColumnBlock(data_path={self.data_path}, column={self.column})"
