from typing import Self
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import dask.array as da


class Divider(BaseEstimator, TransformerMixin):
    """
    This class divides the data by a number.
    :param divider: The number to divide by
    """

    def __init__(self, divider: int = 1) -> None:
        """
        Initialize the divider.

        :param divider: The number to divide by
        """
        self.divider = divider

    def fit(self, X: da.Array, y: da.Array | None = None) -> Self:
        """
        Fit the transformer.

        :param X: The data to fit
        :param y: The target variable
        :return: The fitted transformer
        """
        return self

    def transform(self, X: da.Array, y: da.Array | None = None) -> da.Array:
        """
        Transform the data.

        :param X: The data to transform
        :param y: The target variable
        :return: The transformed data
        """
        return (X / self.divider).astype(np.float32)

    def __str__(self) -> str:
        """
        Return the name of the transformer.

        :return: The name of the transformer
        """
        return f"Divider_{self.divider}"


if __name__ == '__main__':
    # Test the divider
    divider = Divider(2)
    X = da.from_array(np.array([1, 2, 3, 4, 5]))
    X, y = divider.transform(X)
    print(str(divider))
    # Print class of divider
    print(divider.__class__.__name__)
    print(X.compute())