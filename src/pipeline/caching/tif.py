import os
from typing import Any
import warnings
import numpy as np
import rasterio
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from src.logging_utils.logger import logger
import dask.array as da
from src.pipeline.caching.error import CachePipelineError
import dask
from dask import delayed
from rasterio.errors import NotGeoreferencedWarning


class CacheTIFPipeline(BaseEstimator, TransformerMixin):
    """
    The caching pipeline is responsible for loading and storing the data to disk.
    :param data_paths: The paths to the data
    """

    def __init__(self, data_paths: list[str]) -> None:

        if not data_paths:
            logger.error("data_paths are required")
            raise CachePipelineError("data_path is required")

        # Set paths to self
        self.data_paths = data_paths

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
            return store_raw(self.data_paths, X)


def store_raw(data_paths: list[str], dask_array: da.Array) -> da.Array:
    """
    This function stores the raw data to disk.
    :param data_paths: The paths of all the data
    :param dask_array: The dask array to store
    :return: dask array
    """

    # Check if the data paths are defined
    if not data_paths:
        logger.error("data_paths are required to store raw data")
        raise CachePipelineError("data_paths are required to store raw data")

    # Check if all data paths exist and if they do, load them instead of creating them
    exists = [os.path.exists(data_path) for data_path in data_paths]

    # Check if all data paths exist
    if all(exists):
        logger.info("All data paths exist, loading data from disk")
        return parse_raw(data_paths)

    # Check if the dask array is defined
    if dask_array is None:
        logger.error("dask_array is required to store raw data")
        raise CachePipelineError("dask_array is required to store raw data")

    # Check if the data paths are the same length as the dask array
    if len(data_paths) != len(dask_array):
        logger.error("data_paths and dask_array must be the same length")
        raise CachePipelineError(
            "data_paths and dask_array must be the same length")

    # Iterate over the data paths and store the data
    logger.info("Storing data to disk")
    delayed_write_data = delayed(write_data)
    write_tasks = []
    for data_path, data in tqdm(zip(data_paths, dask_array), desc="Creating write tasks"):
        write_tasks.append(delayed_write_data(data_path, data))

    dask.compute(*write_tasks)

    logger.info("Finished storing data to disk")

    # Return the dask array
    return dask_array


def write_data(data_path: str, data: da.Array | np.ndarray[Any, Any]) -> None:
    """
    Function to write the data to disk, wrapper for rasterio.open to use in dask
    :param data_path: The path to write the data to
    :param data: The data to write
    """
    # Define the metadata for the raster file
    metadata = {
        'driver': 'GTiff',
        'height': data.shape[1],
        'width': data.shape[2],
        'count': data.shape[0],
        'dtype': str(data.dtype),
    }

    # Write the data to the raster file
    with rasterio.open(data_path, 'w', **metadata) as dst:
        dst.write(data)


def parse_raw(data_paths: list[str] = []) -> da.Array:
    """
    This function parses the raw data into a dask array.
    :param data_paths: The paths of all the data
    :return: dask array
    """
    if not data_paths:
        logger.error("data_paths are required to parse raw data")
        raise CachePipelineError("data_paths are required to parse raw data")

    # Define the delayed function
    dask_imread = dask.delayed(read_image)

    # Get the sample
    sample = read_image(data_paths[0])

    images = []
    for data_path in tqdm(data_paths, desc="Creating read tasks"):
        lazy_image = dask_imread(data_path)
        image = da.from_delayed(lazy_image, dtype=sample.dtype, shape=sample.shape)
        images.append(image)

    return da.stack(images, axis=0)


def read_image(image_path: str) -> da.Array:
    """
    Read the image from the image path.
    :param image_path: The path to the image
    :return: dask array
    """
    if not image_path:
        logger.error("image_path is required to read image")
        raise CachePipelineError("image_path is required to read image")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        with rasterio.open(image_path) as src:
            image_data = src.read()
            return image_data
