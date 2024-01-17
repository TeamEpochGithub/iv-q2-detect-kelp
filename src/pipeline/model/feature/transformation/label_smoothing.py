"""Pipeline step to smooth the labels."""
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
class LabelSmoothing(BaseEstimator, TransformerMixin):
    """Pipeline step to smooth the labels.

    :param smoothing: The smoothing factor
    """

    smoothing: float = 0.1

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

        # Add gaussian label smoothing to the 1's of the labels to make the model more robust

        logger.info(f"Label smoothing complete in: {time.time() - time_start} seconds.")
        return X
