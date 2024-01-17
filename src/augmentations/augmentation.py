"""Base class for custom data augmentations"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from pandas._typing import npt


@dataclass
class Augmentation(ABC):
    """Base class for custom data augmentations."""

    img_to_apply: int = 1

    @abstractmethod
    def transforms(self, image: npt.NDArray[np.float_], mask: npt.NDArray[np.float_], i: int) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Apply the augmentation to the data.

        :param x: Batch of input features.
        :param y: Batch of labels.
        :param i: Index of the image to apply the augmentation to.
        :return: Augmentation applied to the image and mask at index i
        """
        pass
