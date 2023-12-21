import glob
import time
import numpy as np

from src.logging_utils.logger import logger
import dask.array as da
from src.pipeline.caching.util.error import CachePipelineError
import os
from dask_image.imread import imread


def store_raw(data_path: str, dask_array: da.Array) -> da.Array:
    """
    This function stores the raw data to disk.
    :param data_path: The path of all the data
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
            return imread(f"{data_path}/*.tif").transpose(0, 3, 1, 2)
        elif glob.glob(f"{data_path}/*.npy"):
            logger.info("Data already exists on disk")
            return da.from_npy_stack(data_path).astype(np.float32)

    # Check if the dask array is defined
    if dask_array is None:
        logger.error("dask_array is required to store raw data")
        raise CachePipelineError("dask_array is required to store raw data")

    # Iterate over the data paths and store the data
    logger.info("Storing data to disk")
    start_time = time.time()

    # Iterate over the dask array and store each image
    dask_array = dask_array.astype(np.float32)
    da.to_npy_stack(data_path, dask_array)

    end_time = time.time()
    logger.debug(
        f"Finished storing data to disk in: {end_time - start_time} seconds")

    # Return the dask array
    return dask_array
