"""Creates a new column by applying convolution to a channel."""

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import dask
import dask.array as da
import dask_image.ndfilters as dask_filter
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_utils.logger import logger

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


@dataclass(repr=False)
class Filter(BaseEstimator, TransformerMixin):
    """Filter applies.

    :param filters: A list of filters to apply to the data.
    :param channels: A list of channels to apply the filters to.
    """

    filters: list[Callable[..., npt.NDArray[np.float_] | da.Array]] | list[dask_filter]
    channels: list[int]

    def __post_init__(self) -> None:
        """Validate the filters & channels."""
        if self.filters is None:
            raise ValueError("Filters must be defined in the Convolutions Column step.")

        if self.channels is None:
            raise ValueError("Channels must be defined in the Convolutions Column step.")

        if len(self.filters) != len(self.channels):
            raise ValueError("Filters and channels must be the same length in the Convolutions Column step.")

    def fit(self, X: da.Array, y: da.Array | None = None) -> Self:  # noqa: ARG002
        """Do nothing. Exists for Pipeline compatibility.

        :param X: UNUSED data to fit.
        :param y: UNUSED target variable.
        :return: The fitted transformer.
        """
        return self

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        total_args = ""
        for image_filter in self.filters:
            # Filter is a functools.partial object, grab the underlying function
            filter_name = str(image_filter.func.__name__)  # type: ignore[union-attr]
            filter_args = str(image_filter.keywords)  # type: ignore[union-attr]
            # Now filter_args is a dict, convert to string without ''
            filter_args = filter_args.replace("'", "")
            filter_args = filter_args.replace(":", "")
            # Now this is a function with a memory address, grab the name and arguments
            total_args += f"{filter_name}-{filter_args}"

        return f"Filter(filters={total_args},channels={self.channels})"

    def transform(self, X: da.Array, y: da.Array | None = None) -> da.Array:  # noqa: ARG002
        """Transform the data.

        :param X: The data to transform.
        :param y: UNUSED target variable. Exists for Pipeline compatibility.
        :return: The transformed data.
        """
        start_time = time.time()

        # Loop through all the channels and apply the filter
        for image_filter, channel in zip(self.filters, self.channels, strict=False):
            # Apply the filter
            filter_name = image_filter.__class__.__name__
            logger.info(f"Applying {filter_name} to channel {channel}")
            filtered_channel = image_filter(X[:, channel])

            # Set copy to dtype float32
            filtered_channel = filtered_channel.astype("float32")

            # Concatenate the filtered channel to the end of the array
            X = dask.array.concatenate([X, filtered_channel[:, None]], axis=1)

        logger.info(f"Filter transform complete in: {time.time() - start_time} seconds")
        return X
