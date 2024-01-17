"""Implementation of the Dice coefficient metric for image segmentation."""
from typing import Any

import numpy as np

from src.scoring.scorer import Scorer


class DiceCoefficient(Scorer):
    """Dice coefficient metric."""

    def __init__(self, name: str = "DiceCoefficient") -> None:
        """Initialize the Dice coefficient metric.

        :param name: name of the scorer
        """
        super().__init__(name)

    def __call__(self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> float:
        """Calculate the Dice coefficient.

        :param y_true: true labels
        :param y_pred: predicted labels
        :return: Dice coefficient
        """
        return self.dice_coefficient(y_true, y_pred)

    def dice_coefficient(self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> float:
        """Calculate the Dice coefficient.

        :param y_true: true labels
        :param y_pred: predicted labels
        :return: Dice coefficient
        """
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred)

        return (2 * intersection) / union
