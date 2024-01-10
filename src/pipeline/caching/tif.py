"""The caching tif pipeline is responsible for loading and storing all data to disk."""
import sys

import dask.array as da
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_utils.logger import logger
from src.pipeline.caching.util.error import CachePipelineError
from src.pipeline.caching.util.store_raw import store_raw

if sys.version_info < (3, 11):  # Self was added in Python 3.11
    from typing_extensions import Self
else:
    from typing import Self


class CacheTIFBlock(BaseEstimator, TransformerMixin):
    """The caching pipeline is responsible for loading and storing the data to disk.

    :param data_path: The path to the data
    """

    def __init__(self, data_path: str) -> None:
        """Initialize the caching pipeline.

        :param data_path: The path to the data
        """
        if not data_path:
            logger.error("data_path is required")
            raise CachePipelineError("data_path is required")

        # Set paths to self
        self.data_path = data_path

    def fit(self, X: da.Array, y: da.Array | None = None) -> Self:
        """Fit the data.

        :param X: The data to fit
        :param y: The target variable
        :return: The fitted pipeline
        """
        return self

    def transform(self, X: da.Array, y: da.Array | None = None) -> da.Array:
        """Transform the data.

        :param X: The data to transform
        :param y: The target variable
        :return: The transformed data
        """
        return store_raw(self.data_path, X)

    def __hash__(self) -> int:
        """Hash the class.

        :return: The hash value
        """
        # Get the hash of the class name
        hash_value = hash(self.__class__.__name__)

        # Combine the hash with the hashes of the other attributes
        for attr in self.__dict__:
            if attr != "data_path":
                hash_value ^= hash(self.__dict__[attr])

        return hash_value
