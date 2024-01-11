"""A piepline step that divides the data by a number."""
import sys
import time
from dataclasses import dataclass

import dask.array as da
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_utils.logger import logger

if sys.version_info < (3, 11):  # Self was added in Python 3.11
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class Divider(BaseEstimator, TransformerMixin):
    """Pipeline step to divide the data by a number.

    :param divider: The number to divide by
    """

    divider: int

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
        time_start = time.time()
        result = (X / self.divider).astype(np.float32)
        logger.info(f"Divider transform complete in: {time.time() - time_start} seconds.")
        return result
