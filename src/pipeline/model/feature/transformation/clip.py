"""Pipeline step to clip features based on kelp ranges."""
import sys
import time
from dataclasses import dataclass, field

import dask.array as da
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from src.logging_utils.logger import logger

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class Clip(BaseEstimator, TransformerMixin):
    """Pipeline step to clip features based on kelp ranges."""

    feature_ranges: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize the transformer."""
        # Validate kelp_to_zero to make sure that list is of length 2 and that the first value is less than the second
        for i, band in enumerate(self.feature_ranges):
            if len(band) != 2:
                raise ValueError(f"Invalid kelp_to_zero list at index {i}. Tuple must be of length 2.")
            if band[0] > band[1]:
                raise ValueError(f"Invalid kelp_to_zero list at index {i}. First value must be less than second value.")

    def fit(self, X: da.Array, y: da.Array | None = None) -> Self:  # noqa: ARG002
        """Do nothing. Exists for Pipeline compatibility.

        :param X: UNUSED data to fit.
        :param y: UNUSED target variable.
        :return: The fitted transformer.
        """
        return self

    def transform(self, X: da.Array) -> da.Array:
        """Transform the data.

        :param X: The data to transform
        :return: The transformed data
        """
        time_start = time.time()

        X = X.astype(np.float32)
        X = self.clip(X)

        # Apply this function across the channels
        logger.info(f"Clipping complete in: {time.time() - time_start} seconds.")

        return X

    def clip(self, X: da.Array) -> da.Array:
        """Clip the values in the array according to the kelp ranges specified in the config.

        :param X: The data to clamp
        :return: The clamped data
        """
        # Apply this function across the channels
        results = []
        for c, band in tqdm(enumerate(self.feature_ranges)):
            low = band[0]
            high = band[1]
            modified_channel = np.clip(X[:, c, :, :], low, high)
            results.append(modified_channel)

        return da.stack(results, axis=1) if len(results) > 1 else X
