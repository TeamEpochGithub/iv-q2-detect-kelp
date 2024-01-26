"""Creates a new column by applying convolution to a channel."""

import sys
import time
from dataclasses import dataclass

import dask
import dask.array as da
import dask_image.ndfilters as dask_filter
import skimage
from sklearn.base import BaseEstimator, TransformerMixin

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from src.logging_utils.logger import logger


@dataclass
class Filter(BaseEstimator, TransformerMixin):
    """Filter applies.

    :param band: The band to copy
    """

    filters: list[skimage.filters] | list[dask_filter]
    channels: list[int]

    def __post_init__(self) -> None:

        # Check that channels is not None
        if self.filters is None:
            logger.error("Filters must be defined in the Convolutions Column step.")
            raise ValueError("Filters must be defined")

        # Check that kernel_types is not None
        if self.channels is None:
            logger.error("Channels must be defined in the Convolutions Column step.")
            raise ValueError("Channels must be defined")

        # Check if filters and channels are the same length
        if len(self.filters) != len(self.channels):
            logger.error("Filters and channels must be the same length in the Convolutions Column step.")
            raise ValueError("Filters and channels must be the same length")

    def fit(self, X: da.Array, y: da.Array | None = None) -> Self:  # noqa: ARG002
        """Do nothing. Exists for Pipeline compatibility.

        :param X: UNUSED data to fit.
        :param y: UNUSED target variable.
        :return: The fitted transformer.
        """
        return self

    def __repr__(self) -> str:

        total_args = ""
        for filter in self.filters:
            # Filter is a functools.partial object, grab the underlying function
            filter_name = str(filter.func.__name__)
            filter_args = str(filter.keywords)
            # Now filter_args is a dict, convert to string without ''
            filter_args = filter_args.replace("'", "")
            filter_args = filter_args.replace(":", "")
            # Now this is a function with a memory address, grab the name and arguments
            total_args += f"{filter_name}-{filter_args}"

        to_repr = f"Filter(filters={total_args},channels={self.channels})"
        return to_repr

    def transform(self, X: da.Array, y: da.Array | None = None) -> da.Array:  # noqa: ARG002
        """Transform the data.

        :param X: The data to transform.
        :param y: UNUSED target variable. Exists for Pipeline compatibility.
        :return: The transformed data.
        """

        start_time = time.time()

        # Loop through all the channels and apply the filter
        for i, (filter, channel) in enumerate(zip(self.filters, self.channels)):
            # Apply the filter
            filter_name = filter.__class__.__name__
            logger.info(f"Applying {filter_name} to channel {channel}")
            filtered_channel = filter(X[:, channel])

            # Set copy to dtype float32
            filtered_channel = filtered_channel.astype("float32")

            # Concatenate the filtered channel to the end of the array
            X = dask.array.concatenate([X, filtered_channel[:, None]], axis=1)

        logger.info(f"Filter transform complete in: {time.time() - start_time} seconds")
        return X
