"""Base class for custom data augmentations."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class Augmentation(ABC):
    """Base class for custom data augmentations."""

    p: float
    img_to_apply: int = 1

    @abstractmethod
    def transforms(self, images: npt.NDArray[np.float_], masks: npt.NDArray[np.float_], i: int) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Apply the augmentation to the data.

        :param x: Batch of input features.
        :param y: Batch of labels.
        :param i: Index of the image to apply the augmentation to.
        :return: Augmentation applied to the image and mask at index i
        """
