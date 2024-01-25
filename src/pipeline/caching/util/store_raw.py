"""Module for storing raw data to disk."""
import glob
import os
import time

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
    if os.path.exists(data_path):
        array = _data_exists(data_path, dask_array)
        if array is not None:
            return array
        # Check if path has any tif files

    # Check if the dask array is defined
    if dask_array is None:
        raise CachePipelineError("dask_array is required to store raw data")

    # Iterate over the data paths and store the data
    start_time = time.time()

    # Iterate over the dask array and store each image
    dask_array = dask_array.astype(np.float32)
    # If path does not exist, create it
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if dask_array.ndim == 4:
        dask_array = dask_array.rechunk({0: "auto", 1: -1, 2: -1, 3: -1})
    elif dask_array.ndim == 3:
        dask_array = dask_array.rechunk({0: "auto", 1: -1, 2: -1})

    da.to_npy_stack(data_path, dask_array)
    dask_array = da.from_npy_stack(data_path)

    # Print type and shape of the data
    end_time = time.time()
    logger.info(f"Finished storing data: {dask_array.shape} to {data_path} in: {end_time - start_time} seconds")

    # Return the dask array
    return dask_array


def _data_exists(data_path: str, dask_array: da.Array) -> da.Array | None:
    """Check if the data exists on disk.

    :param data_path: The path of all the data
    :param dask_array: The dask array to store
    :return: dask array
    """
    array = None
    if glob.glob(f"{data_path}/*.tif"):
        logger.info(f"Loading tif data from {data_path}")
        array = imread(f"{data_path}/*.tif").transpose(0, 3, 1, 2)
    if glob.glob(f"{data_path}/*.npy"):
        logger.info(f"Loading npy data from {data_path}")
        array = da.from_npy_stack(data_path).astype(np.float32)

    if array is not None:
        # Check if the shape of the data on disk matches the shape of the dask array
        if array.shape != dask_array.shape:
            logger.warning(f"Shape of data on disk does not match shape of dask array, cache corrupt at {data_path}")
            raise CachePipelineError(
                f"Shape of data on disk ({array.shape}) does not match shape of dask array ({dask_array.shape})",
            )

        # Rechunk the array
        if array.ndim == 4:
            array = array.rechunk({0: "auto", 1: -1, 2: -1, 3: -1})
        elif array.ndim == 3:
            array = array.rechunk({0: "auto", 1: -1, 2: -1})

    return array
