"""Pipeline step that copies a band."""
import sys
import time
from dataclasses import dataclass

import dask
import dask.array as da
from sklearn.base import BaseEstimator, TransformerMixin

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from src.logging_utils.logger import logger


@dataclass
class BandCopy(BaseEstimator, TransformerMixin):
    """BandCopy is a transformer that copies a band.

    :param band: The band to copy
    """

    band: int

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
        # Take band and add it to the end
        copy_of_band = X[:, self.band].copy()

        # Set copy to dtype float32
        copy_of_band = copy_of_band.astype("float32")

        start_time = time.time()
        X = dask.array.concatenate([X, copy_of_band[:, None]], axis=1)

        if X.ndim == 3:
            X = X.rechunk({0: "auto", 1: -1, 2: -1})
        elif X.ndim == 4:
            X = X.rechunk({0: "auto", 1: -1, 2: -1, 3: -1})
        logger.info(f"BandCopy transform complete in: {time.time() - start_time} seconds")
        return X
