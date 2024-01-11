"""Scaler block to fit and transform the data."""
import sys
import time
from dataclasses import dataclass

import dask.array as da
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_utils.logger import logger

if sys.version_info < (3, 11):  # Self was added in Python 3.11
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class ColumnSelection(BaseEstimator, TransformerMixin):
    """Block that selects a subset of columns from the data. Should probably be at the start of the pretrain pipeline.

    :param columns: Columns to select.
    """

    columns: list[int]

    def fit(self, X: da.Array, y: da.Array) -> Self:
        """Return self, no fitting necessary.

        :param X: Data to fit
        :param y: Target data
        """
        return self

    def transform(self, X: da.Array) -> da.Array:
        """Transform the data.

        :param X: Data to transform
        """
        logger.info("Selecting columns...")
        start_time = time.time()
        X = X[:, self.columns]
        X = X.rechunk({1: -1})
        logger.info("Selected columns in %s seconds", time.time() - start_time)
        return X