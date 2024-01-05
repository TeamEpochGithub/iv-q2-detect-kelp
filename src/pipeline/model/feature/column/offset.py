"""Block to compute the offset value w.r.t. the median water value."""
from dataclasses import dataclass
from typing import Self

import dask
import dask.array as da
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def compute_offset(band: da.Array, elevation: da.Array) -> da.Array:
    """Given a single band and elevation, compute the offset value.

    :param band: The band to compute the offset for
    :param elevation: The elevation band
    :return: The offset value
    """
    # Set the land pixels to NaN
    water_masked = band.copy()
    water_masked[elevation > 1] = np.nan

    # Flatten (N,350,350) to (N,122500)
    water_masked = water_masked.reshape((water_masked.shape[0], -1))

    # Compute the median for each image, ignoring NaNs (shape (N,))
    water_median = da.nanmedian(water_masked, axis=-1)

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

    def fit(self, X: da.Array, y: da.Array | None = None) -> Self:
        """Fit the transformer.

        :param X: The data to fit
        :param y: The target variable
        :return: The fitted transformer
        """
        return self

    def transform(self, X: da.Array, y: da.Array | None = None) -> da.Array:
        """Transform the data.

        :param X: The data to transform
        :param y: The target variable
        :return: The transformed data
        """
        result = compute_offset(X[:, self.band], X[:, self.elevation])
        X = dask.array.concatenate([X, result[:, None]], axis=1)
        return X.rechunk()
