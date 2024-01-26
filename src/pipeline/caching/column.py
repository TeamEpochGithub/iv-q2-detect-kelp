"""The caching column block is responsible for loading and storing individual columns to disk."""
import sys
from dataclasses import dataclass

import dask.array as da
from sklearn.base import BaseEstimator, TransformerMixin

from src.pipeline.caching.util.error import CachePipelineError
from src.pipeline.caching.util.store_raw import store_raw

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class CacheColumnBlock(BaseEstimator, TransformerMixin):
    """The caching column block is responsible for loading and storing individual columns to disk.

    :param data_path: The path to the data
    :param column: The column to store
    """

    data_path: str | None = None
    column: int = -1

    def fit(self, X: da.Array, y: da.Array | None = None) -> Self:  # noqa: ARG002
        """Do nothing. Exists for Pipeline compatibility.

        :param X: UNUSED data to fit
        :param y: UNUSED target variable
        :return: Itself
        """
        return self

    def transform(self, X: da.Array) -> da.Array:
        """Transform the data.

        :param X: The data to transform
        :return: The transformed data
        """
        # Check if the data path is set
        if not self.data_path:
            if X is None:
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
