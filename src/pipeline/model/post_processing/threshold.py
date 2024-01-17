"""Threshold the predictions of the model."""
import sys
from dataclasses import dataclass
from typing import Annotated, Any

import dask.array as da
import numpy as np
import numpy.typing as npt
from annotated_types import Interval
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import dice
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from src.logging_utils.logger import logger

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class Threshold(TransformerMixin, BaseEstimator):
    """Threshold the predictions of the model.

    :param threshold: The threshold to use. If None, the threshold will be optimized.
    """

    # noinspection PyTypeHints
    threshold: Annotated[float, Interval(ge=0, le=1)] | None = None

    # noinspection PyPep8Naming
    def fit(self, X: npt.NDArray[np.float_] | da.Array, y: npt.NDArray[np.bool_] | da.Array, **kwargs: Any) -> Self:  # noqa: ANN401
        """Fit the threshold.

        :param X: Output data of a model.
        :param y: Target data. Must have the same flattened shape as X.
        :param kwargs: Unused additional arguments, for compatibility with sklearn pipelines.
        :return: Optimized threshold.
        """
        if self.threshold is not None:
            logger.info(f"Threshold manually set to {self.threshold}. Skipping optimization.")
            return self

        # Using Dask arrays here gives weird warnings about full garbage collections taking too much CPU time
        if isinstance(X, da.Array):
            X = X.compute()
        if isinstance(y, da.Array):
            y = y.compute()

        y_pred = X.ravel()
        y_true = y.ravel()

        if y_pred.shape != y_true.shape:
            raise ValueError("X and y must have the same flattened shape")

        logger.info("Optimizing threshold. This may take a while...")
        self.threshold = minimize_scalar(lambda threshold: dice(y_true, threshold < y_pred), bounds=(0, 1), method="bounded").x

        logger.info(f"Optimized threshold: {self.threshold}")
        return self

    # noinspection PyPep8Naming
    def transform(self, X: npt.NDArray[np.float_] | da.Array) -> npt.NDArray[np.bool_] | da.Array:
        """Transform the predictions.

        :param X: Input data
        :return: Transformed data
        """
        if self.threshold is None:
            raise NotFittedError("Threshold has not been set")

        return self.threshold < X
