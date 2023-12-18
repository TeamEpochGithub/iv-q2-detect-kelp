import time
from typing import Any

import dask
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from src.logging_utils.logger import logger

from src.pipeline.caching.column import CacheColumnPipeline


class BandCopyPipeline():
    """
    This class creates a band copy pipeline.
    :param band: The band to copy
    :param processed_path: path to the processed data
    """

    def __init__(self, band: int, processed_path: str | None = None) -> None:
        """
        This class creates a band copy pipeline.
        :param band: The band to copy
        :param processed_path: path to the processed data
        """
        self.band = band
        if processed_path:
            self.processed_path = processed_path + '/band_copy_' + str(band)

    def get_pipeline(self) -> Pipeline:
        """
        This function creates the band copy pipeline.
        :return: The band copy pipeline
        """
        steps = []

        # Create the band copy pipeline
        steps.append(('band_copy', BandCopy(self.band)))

        # Create the cache column pipeline
        if self.processed_path:
            cache = ('cache', CacheColumnPipeline(
                self.processed_path, column=-1))
            steps.append(cache)

        pipeline_path = self.processed_path + "/pipeline/" if self.processed_path else None
        return Pipeline(steps=steps, memory=pipeline_path)


class BandCopy(BaseEstimator, TransformerMixin):
    """
    BandCopy is a transformer that copies a band.
    :param band: The band to copy
    """

    def __init__(self, band: int) -> None:
        """
        BandCopy is a transformer that copies a band.
        :param band: The band to copy
        """
        self.band = band

    def fit(self, X: Any, y: Any = None) -> Any:
        """
        Fit the transformer.
        :param X: The data to fit
        :param y: The target variable
        :return: The fitted transformer
        """
        return self

    def transform(self, X: Any, y: Any = None) -> Any:
        """
        Transform the data.
        :param X: The data to transform
        :param y: The target variable
        :return: The transformed data
        """
        # Take band and add it to the end
        copy_of_band = X[:, self.band].copy()

        # Set copy to dtype float32
        copy_of_band = copy_of_band.astype("float32")

        start_time = time.time()
        X = dask.array.concatenate([X, copy_of_band[:, None]], axis=1)
        logger.debug(f"dask concat time: {time.time() - start_time}s")
        return X
