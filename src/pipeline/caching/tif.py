"""A pipeline step that caches a TIF image to disk."""
from dataclasses import dataclass
from typing import Self

import dask.array as da
from sklearn.base import BaseEstimator, TransformerMixin

from src.pipeline.caching.util.error import CachePipelineError
from src.pipeline.caching.util.store_raw import store_raw


@dataclass
class CacheTIFPipeline(BaseEstimator, TransformerMixin):
    """The caching pipeline is responsible for loading and storing the data to disk.

    :param data_path: The path to the data
    """

    data_path: str

    def __post_init__(self) -> None:
        """Check if the data path is defined."""
        if not self.data_path:
            raise CachePipelineError("data_path is required")

    def fit(self, X: da.Array, y: da.Array | None = None) -> Self:
        """Do nothing. This method only exists for compatibility with Scikit-Learn Pipelines.

        :param X: The data to fit.
        :param y: The target variable.
        :return: The same pipeline step.
        """
        return self

    def transform(self, X: da.Array, y: da.Array | None = None) -> da.Array:
        """Store the images to disk.

        :param X: The data to store.
        :param y: The target variable. Unused, but required for compatibility with Scikit-Learn Pipelines.
        :return: The transformed data.
        """
        return store_raw(self.data_path, X)
