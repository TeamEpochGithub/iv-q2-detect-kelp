"""Implementation of the Dice coefficient metric for image segmentation."""

import numpy as np


class DiceCoefficient:
    """Dice coefficient metric."""

    def __init__(self, name: str = "DiceCoefficient"):
        super().__init__(name)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.dice_coefficient(y_true, y_pred)

    def dice_coefficient(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the Dice coefficient.

        :param y_true: true labels
        :param y_pred: predicted labels
        :return: Dice coefficient
        """
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred)

        return (2 * intersection) / union
