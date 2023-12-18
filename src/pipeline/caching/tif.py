from typing import Self
from sklearn.base import BaseEstimator, TransformerMixin
from src.logging_utils.logger import logger
from src.pipeline.caching.util.error import CachePipelineError
import numpy.typing as npt
from src.pipeline.caching.util.store_raw import store_raw


class CacheTIFPipeline(BaseEstimator, TransformerMixin):
    """
    The caching pipeline is responsible for loading and storing the data to disk.
    :param data_path: The path to the data
    """

    def __init__(self, data_path: str) -> None:

        if not data_path:
            logger.error("data_path is required")
            raise CachePipelineError("data_path is required")

        # Set paths to self
        self.data_path = data_path

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> Self:
        """
        :param X: The data to fit
        :param y: The target variable
        :return: The fitted pipeline
        """
        return self

    def transform(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> npt.ArrayLike:
        """
        Transform the data.
        :param X: The data to transform
        :param y: The target variable
        :return: The transformed data
        """
        return store_raw(self.data_path, X)
