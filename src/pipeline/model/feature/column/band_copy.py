"""Pipeline step that copies a band."""

import time
from typing import Self

import dask
import dask.array as da
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_utils.logger import logger


class BandCopy(BaseEstimator, TransformerMixin):
    """BandCopy is a transformer that copies a band.

    :param band: The band to copy
    """

    def __init__(self, band: int) -> None:
        """BandCopy is a transformer that copies a band.

        :param band: The band to copy
        """
        self.band = band

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
        # Take band and add it to the end
        copy_of_band = X[:, self.band].copy()

        # Set copy to dtype float32
        copy_of_band = copy_of_band.astype("float32")

        start_time = time.time()
        X = dask.array.concatenate([X, copy_of_band[:, None]], axis=1)
        X = X.rechunk()
        logger.debug(f"dask concat time: {time.time() - start_time}s")
        return X

    def __str__(self) -> str:
        """Return the name of the transformer.

        :return: The name of the transformer
        """
        return f"BandCopy_{self.band}"


if __name__ == "__main__":
    # Test the band copy
    band_copy = BandCopy(1)
    X = da.from_array([[1, 2], [3, 4]])
    X = band_copy.transform(X)
    print(str(band_copy))
    # Print class of band copy
    print(band_copy.__class__.__name__)
    print(X.compute())
