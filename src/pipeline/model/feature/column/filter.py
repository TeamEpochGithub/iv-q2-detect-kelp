"""Creates a new column by applying convolution to a channel."""

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import dask.array as da
import joblib
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

    filters: list[Callable[..., da.Array | npt.NDArray[np.float_]]]
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
        """Return the string representation of the filter.

        :return: The string representation of the filter.
        """
        total_args: list[str] = [image_filter_to_str(image_filter) for image_filter in self.filters]
        # Remove square brackets from self.channels and turn into string
        channels = str(self.channels).replace("[", "(").replace("]", ")")
        # Convert total_args to a string and hash it

        total_args = joblib.hash(str(total_args))
        return f"Filter(filters={''.join(total_args)},channels={channels})"

    def transform(self, X: da.Array, y: da.Array | None = None) -> da.Array:  # noqa: ARG002
        """Transform the data.

        :param X: The data to transform.
        :param y: UNUSED target variable. Exists for Pipeline compatibility.
        :return: The transformed data.
        """
        X = X.rechunk({0: "auto", 1: -1, 2: -1, 3: -1})

        return X.map_blocks(
            self.transform_chunk,
            dtype=np.float32,
            chunks=(X.chunks[0], (X.chunks[1][0] + len(self.filters),), X.chunks[2], X.chunks[3]),
            meta=np.array((), dtype=np.float32),
        )

    def transform_chunk(self, X: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """Transform a chunk of data.

        :param X: The data to transform.
        :return: The transformed data.
        """
        start_time = time.time()
        # Loop through all the channels and apply the filter
        for image_filter, channel in zip(self.filters, self.channels, strict=False):
            # Apply the filter
            filter_name = image_filter.func.__name__  # type: ignore[attr-defined]
            logger.info(f"Applying {filter_name} to channel {channel}")
            filtered_channel = np.empty_like(X[:, channel])
            for image in range(X.shape[0]):
                filtered_channel[image] = image_filter(X[image, channel])

            # Set copy to dtype float32
            filtered_channel = filtered_channel.astype("float32")
            # Concatenate the filtered channel to the end of the array
            X = np.concatenate([X, filtered_channel[:, None]], axis=1)
        logger.info(f"Filter transform complete in: {time.time() - start_time} seconds")
        return X


def image_filter_to_str(image_filter: Callable[..., da.Array | npt.NDArray[np.float_]]) -> str:
    """Convert an image filter to a string.

    :param image_filter: The image filter to convert.
    :return: The string representation of the image filter.
    """
    filter_name = image_filter.func.__name__  # type: ignore[attr-defined]
    filter_args = str(image_filter.keywords).replace("'", "").replace(":", "")  # type: ignore[attr-defined]
    return f"{filter_name}-{filter_args}"
