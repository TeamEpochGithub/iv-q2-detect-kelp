import glob
import time
import warnings
import numpy as np

from tqdm import tqdm
from src.logging_utils.logger import logger
import dask.array as da
import dask
from src.pipeline.caching.util.error import CachePipelineError
from src.pipeline.caching.util.parse_raw import parse_raw
import os
import skimage.io
from dask.delayed import delayed
import imageio.v3 as iio
from dask_image.imread import imread


def store_raw(ids, data_path: str, dask_array: da.Array) -> da.Array:
    """
    This function stores the raw data to disk.
    :param data_paths: The paths of all the data
    :param dask_array: The dask array to store
    :return: dask array
    """

    # Check if the data path is defined
    if not data_path:
        logger.error("data_path is required to store raw data")
        raise CachePipelineError("data_paths is required to store raw data")

    # Check if the data exists on disk
    if os.path.exists(data_path):
        # Check if path has any tif files
        if glob.glob(f"{data_path}/*.tif"):
            logger.info("Data already exists on disk")
            return imread(f"{data_path}/*.tif")
        elif glob.glob(f"{data_path}/*.npy"):
            logger.info("Data already exists on disk")
            return da.from_npy_stack(data_path)

    # Check if the dask array is defined
    if dask_array is None:
        logger.error("dask_array is required to store raw data")
        raise CachePipelineError("dask_array is required to store raw data")

    # Iterate over the data paths and store the data
    logger.info("Storing data to disk")
    start_time = time.time()

    # Create data_paths
    data_paths = [f"{data_path}/{id}.npy" for id in ids]

    # Iterate over the dask array and store each image
    dask.array.to_npy_stack(data_path, dask_array)

    end_time = time.time()
    logger.info(f"Storing data to disk took {end_time - start_time} seconds")
    logger.info("Finished storing data to disk")

    # Return the dask array
    return dask_array


def save_file(filename, arr):
    # Save array to numpy file
    np.save(filename, arr)
