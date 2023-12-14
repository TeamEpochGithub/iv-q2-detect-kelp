import dask.array as da
from sklearn.base import BaseEstimator, TransformerMixin


class Divider(BaseEstimator, TransformerMixin):
    """
    This class divides the data by a number.
    :param divider: The number to divide by
    """

    def __init__(self, divider: int = 1) -> None:
        self.divider = divider

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return dask_division(X, self.divider)


def dask_division(dask_array: da.Array, divider: int = 1) -> da.Array:
    """
    This function divides the dask array by a number.
    :param dask_array: The dask array to divide
    :param divider: The number to divide by
    :return: The divided dask array
    """

    return dask_array / divider
