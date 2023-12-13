import dask.array as da
from sklearn.base import BaseEstimator, TransformerMixin
import tifffile as tiff

from src.logging_utils.logger import logger
from src.pipeline.store.error import StorePipelineError


class StoreTIFPipeline(BaseEstimator, TransformerMixin):
    """
    The store pipeline is responsible for storing the data to disk.
    :param data_paths: The paths to the data
    """

    def __init__(self, data_paths: list[str]) -> None:

        if not data_paths:
            logger.error("data_paths are required")
            raise StorePipelineError("data_path is required")

        # Set paths to self
        self.data_paths = data_paths

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        logger.info("Storing data to disk")
        store_raw(self.data_paths, X)
        return X


def store_raw(data_paths: list[str], dask_array: da.Array) -> None:
    """
    This function stores the raw data to disk.
    :param data_paths: The paths of all the data
    :param dask_array: The dask array to store
    """

    # Check if the data paths are defined
    if not data_paths:
        logger.error("data_paths are required to store raw data")
        raise StorePipelineError("data_paths are required to store raw data")

    # Check if the dask array is defined
    if not dask_array:
        logger.error("dask_array is required to store raw data")
        raise StorePipelineError("dask_array is required to store raw data")

    # Check if the data paths are the same length as the dask array
    if len(data_paths) != len(dask_array):
        logger.error("data_paths and dask_array must be the same length")
        raise StorePipelineError("data_paths and dask_array must be the same length")

    # Iterate over the data paths and store the data
    for data_path, data in zip(data_paths, dask_array):
        # Write the data
        tiff.imwrite(data_path, data.compute())
