"""Block to compute the difference of two bands after ofsetting. See src/pipeline/model/feature/column/offset.py for more details."""
import sys
from dataclasses import dataclass

import dask
import dask.array as da
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_utils.logger import logger
from src.pipeline.model.feature.column.offset import compute_offset

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


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
        logger.info("Computing offset difference...")
        offset_a = compute_offset(X[:, self.a], X[:, self.elevation])
        offset_b = compute_offset(X[:, self.b], X[:, self.elevation])
        result = offset_a - offset_b

        X = dask.array.concatenate([X, result[:, None]], axis=1)
        return X.rechunk()
