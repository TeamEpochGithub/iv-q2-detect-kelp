"""Threshold the predictions of the model."""
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated

import dask.array as da
import numpy as np
import numpy.typing as npt
from annotated_types import Gt, Interval
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import dice
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from tqdm import tqdm

from src.logging_utils.logger import logger

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


def _calc_dice_dissimilarity(threshold: float, y_true: npt.NDArray[np.bool_], y_pred: npt.NDArray[np.float_], pbar: tqdm | None = None) -> float:
    """Calculate the dice dissimilarity for a given threshold. Used for optimizing the threshold.

    :param y_true: The true labels.
    :param y_pred: The predicted labels.
    :param threshold: The threshold to use.
    :return: The dice score.
    """
    score = dice(y_true, threshold < y_pred)

    if pbar is not None:
        pbar.update()
        pbar.set_postfix({"Threshold": threshold, "Dice Coefficient": 1 - score})

    return score


@dataclass
class Threshold(TransformerMixin, BaseEstimator):
    """Threshold the predictions of the model.

    :param threshold: The threshold to use ∈ [0, 1]. If None, the threshold will be optimized.
    :param max_iterations: The maximum number of iterations to use when optimizing the threshold. Defaults to 500. Only used for optimizing the threshold.
    """

    # noinspection PyTypeHints
    threshold: Annotated[float, Interval(ge=0, le=1)] | None = None
    max_iterations: Annotated[int, Gt(0)] = 500

    def fit(self, X: npt.NDArray[np.float_] | da.Array, y: npt.NDArray[np.bool_] | da.Array, test_indices: Iterable[int] | None = None) -> Self:  # noqa: ARG002
        """Fit the threshold.

        :param X: Output data of a model.
        :param y: Target data. Must have the same flattened shape as X.
        :param test_indices: UNUSED indices of the test data in X. Exists for compatibility with the Pipeline.
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
            raise ValueError("Shape mismatch: X and y must have the same flattened shape")

        with tqdm(total=self.max_iterations, desc="Optimizing threshold") as pbar:
            self.threshold = minimize_scalar(_calc_dice_dissimilarity, args=(y_true, y_pred, pbar), bounds=(0, 1), options={"maxiter": self.max_iterations}).x
            pbar.total = pbar.n

        logger.info(f"Optimized threshold: {self.threshold}")
        return self

    def transform(self, X: npt.NDArray[np.float_] | da.Array) -> npt.NDArray[np.bool_] | da.Array:
        """Transform the predictions.

        :param X: Input data
        :return: Transformed data
        """
        if self.threshold is None:
            raise NotFittedError("Threshold has not been set")

        logger.info(f"Thresholding at {self.threshold}")
        return self.threshold < X
