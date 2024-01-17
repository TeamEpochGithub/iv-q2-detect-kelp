"""Base class for data augmentation transformations."""
import asyncio
import concurrent
from dataclasses import dataclass

import albumentations
import numpy as np
from pandas._typing import npt

from src.augmentations.augmentation import Augmentation


@dataclass
class Transformations:
    """Base class for data augmentation transformations, contains a list of augmentations to apply to the data."""
    alb: albumentations.Compose
    aug: list[Augmentation]

    def transform(self, x_arr: npt.NDArray[np.float_], y_arr: npt.NDArray[np.float_]) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Applies all the augmentations to the current batch.

        :param x: Batch of input features.
        :param y: Batch of labels.
        :return: Augmented batch
        """
        # First apply the albumentations on the batch in parallel
        if self.alb is not None:
            x_arr, y_arr = self.apply_albumentations(x_arr, y_arr)
        if self.aug is not None:
            x_arr, y_arr = self.apply_augmentations(x_arr, y_arr)
        return x_arr, y_arr

    def apply_albumentations(self, x_arr: npt.NDArray[np.float_], y_arr: npt.NDArray[np.float_]) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Applies all the albumentations to the current batch.

        :param x: Batch of input features.
        :param y: Batch of labels.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            futures = [loop.run_in_executor(executor, self.apply_albumentation, x_arr[i].transpose(1, 2, 0), y_arr[i]) for i in range(len(x_arr))]
            looper = asyncio.gather(*futures)
        augmentation_results = loop.run_until_complete(looper)
        for i in range(len(x_arr)):
            x_arr[i] = augmentation_results[i][0].transpose(2, 0, 1)
            y_arr[i] = augmentation_results[i][1]
        return x_arr, y_arr

    def apply_augmentations(self, x_arr: npt.NDArray[np.float_], y_arr: npt.NDArray[np.float_]) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Apply the custom augmentations to the current batch.

        :param x: Input features.
        :param y: Labels.
        :return: augmented data
        """
        # Apply the augmentations in a paralleized way using asyncio
        with concurrent.futures.ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            futures = [loop.run_in_executor(executor, self.apply_augmentation, x_arr[i].transpose(1, 2, 0), y_arr[i]) for i in range(len(x_arr))]
            looper = asyncio.gather(*futures)
        augmentation_results = loop.run_until_complete(looper)

        # For every element in the batch, apply the augmentations list.
        for i in range(len(x_arr)):
            x_arr[i] = augmentation_results[i][0].transpose(2, 0, 1)
            y_arr[i] = augmentation_results[i][1]
        return x_arr, y_arr

    def apply_augmentation(self, x_arr: npt.NDArray[np.float_], y_arr: npt.NDArray[np.float_], i: int) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Apply the augmentation to the data.

        :param x: Input features.
        :param y: Labels.
        :return: augmented data
        """
        for augmentation in self.aug:
            x_arr, y_arr = augmentation.transforms(x_arr, y_arr, i)
        return x_arr, y_arr

    def apply_albumentation(self, image: npt.NDArray[np.float_], mask: npt.NDArray[np.float_]) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Applies the albumentation to the current image and mask.

        :param x: Input features.
        :param y: Labels.
        :return: augmented data
        """
        transformed_dict = self.alb(image=image, mask=mask)
        return transformed_dict["image"], transformed_dict["mask"]
