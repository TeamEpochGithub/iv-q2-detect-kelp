"""Block to compute the offset value w.r.t. the median water value."""
import sys
from dataclasses import dataclass

import dask
import dask.array as da
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_utils.logger import logger

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


def compute_offset(band: da.Array, elevation: da.Array) -> da.Array:
    """Given a single band and elevation, compute the offset value.

    Uses median twice, separately for each axis, since dask doesn't support nanmedian for multiple axes.

    :param band: The band to compute the offset for (N,350,350)
    :param elevation: The elevation band  (N,350,350)
    :return: The offset value
    """
    # Set the land pixels to NaN
    water_masked = band.copy()
    water_masked[elevation > 1] = np.nan  # (N,350,350)

    # Check if all values are NaN, if so replace with 0
    nan_mask = da.isnan(water_masked).all(axis=(-2, -1))
    water_masked = da.where(nan_mask[:, None, None], 0, water_masked)

    # Compute the median for each image, ignoring NaNs (shape (N,))
    water_median = da.nanmedian(water_masked, axis=(-2, -1))

    # Compute the offset, expanding the median to the shape (N,1,1)
    return band - water_median[:, None, None]


@dataclass
class Offset(BaseEstimator, TransformerMixin):
    """Offset computes the difference between a pixel's value and the median value across all water of that given band.

    In pseudocode: ``Offset = X - median(X[water])``, where ``water = elevation < 1``

    :param band: Index of the band
    :param elevation: Index of the elevation band (default 6)
    """

    band: int
    elevation: int = 6

    def fit(self, X: da.Array, y: da.Array | None = None) -> Self:  # noqa: ARG002
        """Do nothing. Exists for Pipeline compatibility.

        :param X: UNUSED data to fit.
        :param y: UNUSED target variable.
        :return: The fitted transformer.
        """
        return self

    def transform(self, X: da.Array, y: da.Array | None = None) -> da.Array:  # noqa: ARG002
        """Transform the data.

        :param X: The data to transform.
        :param y: UNUSED target variable. Exists for Pipeline compatibility.
        :return: The transformed data.
        """
        logger.info("Computing offset...")
        result = compute_offset(X[:, self.band], X[:, self.elevation])
        X = dask.array.concatenate([X, result[:, None]], axis=1)
        return X.rechunk()
