"""Pipeline step set all feature data outside the kelp region to zero."""
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
class SetOutsideRange(BaseEstimator, TransformerMixin):
    """Pipeline step set all feature data outside the kelp region to zero.

    :param ranges: A list of tuples of the form (low, high) where low and high are the lower and upper bounds of the kelp region.
    :param values: A list of values to set the data to if it is outside the kelp region.
    :param nan_to_zero: If true, convert all nan values to 0.
    :param nan_value: If nan_to_zero is true, convert all values equal to nan_value to 0.
    """

    ranges: list[list[float]] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    nan_to_zero: bool = True
    nan_value: int = 0

    def __post_init__(self) -> None:
        """Initialize the transformer."""
        # Validate ranges to make sure that list is of length 2 and that the first value is less than the second
        for i, band in enumerate(self.ranges):
            if len(band) != 2:
                logger.error(f"Invalid ranges list at index {i}. Tuple must be of length 2.")
                raise ValueError(f"Invalid ranges list at index {i}. Tuple must be of length 2.")
            if band[0] > band[1]:
                logger.error(f"Invalid ranges list at index {i}. First value must be less than second value.")
                raise ValueError(f"Invalid ranges list at index {i}. First value must be less than second value.")

        # Validate that ranges and values are equal length
        if len(self.ranges) != len(self.values):
            logger.error(f"Invalid ranges list. Ranges and values must be equal length.")
            raise ValueError(f"Invalid ranges list. Ranges and values must be equal length.")

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
        X = self.set_out_of_range(X)
        # Apply nan_to_zero if true
        if self.nan_to_zero:
            # If a custom nan value is set, convert that value to 0, else convert all nan values to 0
            if self.nan_value == 0:
                X = da.where(da.isnan(X), 0.0, X)
            else:
                X = da.where(self.nan_value == X, 0.0, X)

        logger.info(f"No Kelp Zero complete in: {time.time() - time_start} seconds.")

        return X

    def set_out_of_range(self, X: da.Array) -> da.Array:
        """Set all values outside of the kelp region to zero.

        :param X: The data to transform
        :return: Transformed dataset
        """

        # Create a function that applies the conditional operation
        def apply_conditions(x: da.Array, low: float, high: float, i: int) -> da.Array:
            return da.where((x < low) | (x > high), self.values[i], x)

        # Apply this function across the channels
        results = []
        for c, band in tqdm(enumerate(self.ranges)):
            low = band[0]
            high = band[1]
            modified_channel = apply_conditions(X[:, c, :, :], low, high, c)
            results.append(modified_channel)

        # Stack the modified channels back together
        return da.stack(results, axis=1) if len(results) > 1 else X
