import os
import rasterio
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from src.logging_utils.logger import logger
import dask.array as da
import tifffile as tiff
from src.pipeline.caching.error import CachePipelineError
import dask
from dask import delayed


class CacheTIFPipeline(BaseEstimator, TransformerMixin):
    """
    The caching pipeline is responsible for loading and storing the data to disk.
    :param data_paths: The paths to the data
    """

    def __init__(self, data_paths: list[str]) -> None:

        if not data_paths:
            logger.error("data_paths are required")
            raise CacheTIFPipeline("data_path is required")

        # Set paths to self
        self.data_paths = data_paths

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return store_raw(self.data_paths, X)


def store_raw(data_paths: list[str], dask_array: da.Array) -> da.Array:
    """
    This function stores the raw data to disk.
    :param data_paths: The paths of all the data
    :param dask_array: The dask array to store
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
    # Iterate over the data paths and store the data
    logger.info("Storing data to disk")
    write_tasks = []
    for data_path, data in tqdm(zip(data_paths, dask_array), desc="Creating write tasks"):
        write_tasks.append(delayed(write_data)(data_path, data))

    dask.compute(*write_tasks)

    logger.info("Finished storing data to disk")

    # Return the dask array
    return dask_array


def write_data(data_path, data):
    # Write the data
    tiff.imwrite(data_path, data)


def parse_raw(data_paths: list[str] = None) -> da.array:
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

    # Image paths are the files in the data path

    # Get the image paths
    lazy_images = [dask_imread(image_path) for image_path in data_paths]

    # Check if there are images
    if not lazy_images:
        logger.error("No images found in data_path")
        raise CachePipelineError("No images found in data_path")

    # Get the sample
    sample = lazy_images[0].compute()

    # Create the dask array
    images = [da.from_delayed(
        lazy_image, dtype=sample.dtype, shape=sample.shape) for lazy_image in lazy_images]

    return da.stack(images, axis=0)


def read_image(image_path: str) -> da.array:
    """
    Read the image from the image path.
    :param image_path: The path to the image
    :return: dask array
    """
    if not image_path:
        logger.error("image_path is required to read image")
        raise CachePipelineError("image_path is required to read image")

    with rasterio.open(image_path) as src:
        return src.read()
