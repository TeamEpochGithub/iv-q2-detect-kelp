from typing import Any
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from src.logging_utils.logger import logger
import dask.array as da
from src.pipeline.caching.util.error import CachePipelineError
from rasterio.errors import NotGeoreferencedWarning

from src.pipeline.caching.util.store_raw import store_raw


class CacheColumnPipeline(BaseEstimator, TransformerMixin):
    """
    The caching pipeline is responsible for loading and storing the data to disk.
    :param data_paths: The paths to the data
    """

    def __init__(self, data_paths: list[str], column: int = -1) -> None:

        if not data_paths:
            logger.error("data_paths are required")
            raise CachePipelineError("data_path is required")

        # Set paths to self
        self.data_paths = data_paths

        # Set the column to self
        self.column = column

    def fit(self, X: Any, y: Any = None) -> Any:
        """
        :param X: The data to fit
        :param y: The target variable
        :return: The fitted pipeline
        """
        return self

    def transform(self, X: Any, y: Any = None) -> da.Array:
        """
        Transform the data.
        :param X: The data to transform
        :param y: The target variable
        :return: The transformed data
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

            # Load or store the data column
            logger.info("Loading or storing column")
            column = store_raw(self.data_paths, X[:, self.column])

            # Return the column with the original data
            X[:, self.column] = column.squeeze()
            return X
