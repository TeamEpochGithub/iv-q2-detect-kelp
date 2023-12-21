"""Module for storing raw data to disk."""

import time
from pathlib import Path

import dask.array as da
import numpy as np
from dask_image.imread import imread

from src.logging_utils.logger import logger
from src.pipeline.caching.util.error import CachePipelineError


def store_raw(data_path: str, dask_array: da.Array) -> da.Array:
    """Store the raw data to disk.

    :param data_path: The path of all the data
    :param dask_array: The dask array to store
    :return: dask array
    """
    # Check if the data path is defined
    if not data_path:
        raise CachePipelineError("data_paths is required to store raw data")

    # Check if the data exists on disk
    if Path(data_path).exists():
        # Check if path has any tif files
        if Path(data_path).glob("*.tif"):
            logger.info("Data already exists on disk")
            return imread(f"{data_path}/*.tif").transpose(0, 3, 1, 2)
        if Path(data_path).glob("*.npy"):
            logger.info("Data already exists on disk")
            return da.from_npy_stack(data_path).astype(np.float32)

    # Check if the dask array is defined
    if dask_array is None:
        raise CachePipelineError("dask_array is required to store raw data")

    # Iterate over the data paths and store the data
    logger.info("Storing data to disk")
    start_time = time.time()

    # Iterate over the dask array and store each image
    da.to_npy_stack(data_path, dask_array.astype(np.float32))

    end_time = time.time()
    logger.debug(f"Finished storing data to disk in: {end_time - start_time} seconds")

    # Return the dask array
    return dask_array
