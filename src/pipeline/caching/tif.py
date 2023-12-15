from typing import Any
from sklearn.base import BaseEstimator, TransformerMixin
from src.logging_utils.logger import logger
import dask.array as da
from src.pipeline.caching.util.error import CachePipelineError

from src.pipeline.caching.util.store_raw import store_raw


class CacheTIFPipeline(BaseEstimator, TransformerMixin):
    """
    The caching pipeline is responsible for loading and storing the data to disk.
    :param data_paths: The paths to the data
    """

    def __init__(self, ids, data_path: str) -> None:

        if not data_path:
            logger.error("data_path is required")
            raise CachePipelineError("data_path is required")

        # Set paths to self
        self.data_path = data_path
        self.ids = ids

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
        return store_raw(self.ids, self.data_path, X)
