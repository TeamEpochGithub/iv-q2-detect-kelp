"""Block to compute the difference of two bands after ofsetting. See src/pipeline/model/feature/column/offset.py for more details."""
from dataclasses import dataclass
from typing import Self

import dask
import dask.array as da
from sklearn.base import BaseEstimator, TransformerMixin

from src.pipeline.model.feature.column.offset import compute_offset


@dataclass
class OffsetDiff(BaseEstimator, TransformerMixin):
    """Offset diff computes the difference of two bands after computing their offset values.

    See src/pipeline/model/feature/column/offset.py for more details.

    Example: given the default band order of [SWIR,NIR,Red,Green,Blue]:

    - ODVI: Offset Difference Vegetation Index = Offset(NIR) - Offset(Red) = OffsetDiff(1, 2)


    :param a: Index of the first band
    :param b: Index of the second band
    :param elevation: Index of the elevation band (default 6)
    """

    a: int
    b: int
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
        offset_a = compute_offset(X[:, self.a], X[:, self.elevation])
        offset_b = compute_offset(X[:, self.b], X[:, self.elevation])
        result = offset_a - offset_b

        X = dask.array.concatenate([X, result[:, None]], axis=1)
        return X.rechunk()
