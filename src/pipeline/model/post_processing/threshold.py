"""Threshold the predictions of the model."""
import sys
from dataclasses import dataclass, field

import dask.array as da
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import dice
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_utils.logger import logger

if sys.version_info < (3, 11):  # Self was added in Python 3.11
    from typing_extensions import Self
else:
    from typing import Self


def threshold_func(threshold: float, y_pred: npt.NDArray[np.float_], y_true: npt.NDArray[np.bool_]) -> float:
    """Optimize the threshold using the dice dissimilarity.

    :param threshold: The threshold to use
    :param y_pred: The predictions
    :param y_true: The targets
    """
    return dice(y_true, y_pred > threshold)


@dataclass
class Threshold(TransformerMixin, BaseEstimator):
    """Threshold the predictions of the model."""

    _threshold: float = field(init=False)

    def fit(self: Self, X: npt.NDArray[np.float_] | da.Array, y: npt.NDArray[np.bool_] | da.Array) -> Self:
        """Fit the threshold.

        :param X: Output data of a model.
        :param y: Target data. Must have the same flattened shape as X.
        :return: Optimized threshold.
        """
        if isinstance(X, da.Array):
            flat_X = X.compute().flatten()
        else:
            flat_X = X.flatten()

        if isinstance(y, da.Array):
            flat_y = y.compute().flatten()
        else:
            flat_y = y.flatten()

        if flat_X.shape != flat_y.shape:
            raise ValueError("X and y must have the same flattened shape")

        # check_X_y(X, y, dtype=np.bool_, ensure_2d=False, allow_nd=True, multi_output=True, estimator=self)

        self._threshold = minimize_scalar(threshold_func, bounds=(0, 1), args=(flat_X, flat_y), method="bounded").x
        logger.debug(f"Optimized threshold: {self._threshold}")

        return self

    def transform(self: Self, X: npt.NDArray[np.float_] | da.Array) -> npt.NDArray[np.bool_]:
        """Transform the predictions.

        :param X: Input data
        :return: Transformed data
        """
        # print(X)
        return self._threshold < X


# if __name__ == "__main__":
#     X = np.random.rand(100, 100)
#     y = np.random.rand(100, 100) > 0.5
#
#     print(X)
#     print(y)
#
#     threshold = Threshold()
#     threshold.fit(X, y)
#     print(threshold._threshold)
#     print(threshold.transform(X))
