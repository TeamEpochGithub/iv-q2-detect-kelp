"""Block to compute the normalized difference between two bands. For e.g. NDVI, NDWI, etc."""
import sys
from dataclasses import dataclass

import dask
import dask.array as da
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_utils.logger import logger

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class NormDiff(BaseEstimator, TransformerMixin):
    """NormDiff computes (a-b)/(a+b) given the indices to bands A and B.

    Some examples, given the default band order of [SWIR,NIR,Red,Green,Blue]:

    - NDVI: Normalized Difference Vegetation Index = NormDiff(NIR, Red) = NormDiff(1, 2)
    - NDWI: Normalized Difference Water Index = NormDiff(Green, NIR) = NormDiff(3, 1)


    :param a: Index of the first band
    :param b: Index of the second band
    """

    a: int
    b: int

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
        # perform the normalized difference
        logger.info("Computing normalized difference...")
        result = (X[:, self.a] - X[:, self.b]) / (X[:, self.a] + X[:, self.b])
        # Fill nans with 0
        result = da.nan_to_num(result)
        X = dask.array.concatenate([X, result[:, None]], axis=1)
        return X.rechunk()
