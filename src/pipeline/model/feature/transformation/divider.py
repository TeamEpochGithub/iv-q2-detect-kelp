from typing import Any
import dask.array as da
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Divider(BaseEstimator, TransformerMixin):
    """
    This class divides the data by a number.
    :param divider: The number to divide by
    """

    def __init__(self, divider: int = 1) -> None:
        self.divider = divider

    def fit(self, X: Any, y: Any = None) -> Any:
        """
        Fit the transformer.
        :param X: The data to fit
        :param y: The target variable
        :return: The fitted transformer
        """
        return self

    def transform(self, X: Any, y: Any = None) -> Any:
        """
        Transform the data.
        :param X: The data to transform
        :param y: The target variable
        :return: The transformed data
        """
        return division(X, self.divider)


def division(array: da.Array | np.ndarray, divider: int = 1) -> da.Array | np.ndarray:
    """
    This function divides the array by a number.
    :param array: The  array to divide
    :param divider: The number to divide by
    :return: The divided array
    """

    return array / divider
