import os
import dask
import rasterio
from sklearn.base import BaseEstimator, TransformerMixin
from src.pipeline.parse.error import ParsePipelineError
from src.logging_utils.logger import logger
import dask.array as da


class ParseTIFPipeline(BaseEstimator, TransformerMixin):
    """
    The parse pipeline is responsible for parsing disk data into a dask array.
    :param data_path: The path to the data
    """

    def __init__(self, data_path: str) -> None:

        if not data_path:
            logger.error("data_path is required")
            raise ParsePipelineError("data_path is required")

        # Set paths to self
        self.data_path = data_path

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        logger.info("Parsing raw data from tif files")
        return parse_raw(self.data_path)


def parse_raw(data_path: str) -> da.array:
    """
    This function parses the raw data into a dask array.
    :param data_path: The path to the data
    :return: dask array
    """
    if not data_path:
        logger.error("data_path is required to parse raw data")
        raise ParsePipelineError("data_path is required to parse raw data")

    # Define the delayed function
    dask_imread = dask.delayed(read_image)

    # Get the image paths
    lazy_images = [dask_imread(os.path.join(data_path, image_path))
                   for image_path in os.listdir(data_path)]

    # Check if there are images
    if not lazy_images:
        logger.error("No images found in data_path")
        raise ParsePipelineError("No images found in data_path")

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
        raise ParsePipelineError("image_path is required to read image")

    with rasterio.open(image_path) as src:
        return src.read()
