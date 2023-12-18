from typing import Any
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
        return (X / self.divider).astype(np.float32)
