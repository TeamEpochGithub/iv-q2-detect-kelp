"""A piepline step that divides the data by a number."""
from dataclasses import dataclass
from typing import Self

import dask.array as da
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class Divider(BaseEstimator, TransformerMixin):
    """Pipeline step to divide the data by a number.

    :param divider: The number to divide by
    """

    divider: float = 1

    def fit(self, X: da.Array, y: da.Array | None = None) -> Self:
        """Do nothing. This method only exists for compatibility with Scikit-Learn Pipelines.

        :param X: The data to fit
        :param y: The target variable
        :return: The fitted transformer
        """
        return self

    def transform(self, X: da.Array, y: da.Array | None = None) -> da.Array:
        """Divide the data.

        :param X: The data to transform
        :param y: The target variable
        :return: The transformed data
        """
        return (X / self.divider).astype(np.float32)
