"""Base class for data augmentation transformations."""
import asyncio
import concurrent
from dataclasses import dataclass

import albumentations
import kornia
import numpy as np
import numpy.typing as npt
import torch

from src.augmentations.augmentation import Augmentation


@dataclass
class Transformations:
    """Base class for data augmentation transformations, contains a list of augmentations to apply to the data."""

    alb: albumentations.Compose | None = None
    aug: list[Augmentation] | None = None
    korn: kornia.augmentation.AugmentationSequential | None = None

    def __post_init__(self) -> None:
        """Initialize the Mosaic augmentation."""
        # Initialize the random number generator
        self.rng = np.random.default_rng(42)

    def transform(
        self,
        x_arr: npt.NDArray[np.float_],
        y_arr: npt.NDArray[np.float_],
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]] | tuple[torch.Tensor, torch.Tensor]:
        """Apply all the augmentations to the current batch.

        :param x: Batch of input features.
        :param y: Batch of labels.
        :return: Augmented batch
        """
        # First apply the albumentations on the batch in parallel
        if self.alb is not None:
            x_arr, y_arr = self.apply_albumentations(x_arr, y_arr)
        if self.aug is not None:
            x_arr, y_arr = self.apply_augmentations(x_arr, y_arr)
        if self.korn is not None:
            x_tensor: torch.Tensor
            y_tensor: torch.Tensor
            x_tensor, y_tensor = self.apply_kornia(x_arr, y_arr)
            return x_tensor, y_tensor
        return x_arr, y_arr

    def apply_albumentations(self, x_arr: npt.NDArray[np.float_], y_arr: npt.NDArray[np.float_]) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Apply all the albumentations to the current batch.

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
            futures = [loop.run_in_executor(executor, self.apply_augmentation, x_arr, y_arr.reshape(-1, 1, y_arr.shape[1], y_arr.shape[2]), i) for i in range(len(x_arr))]
            looper = asyncio.gather(*futures)
        augmentation_results = loop.run_until_complete(looper)

        # For every element in the batch, apply the augmentations list.
        for i in range(len(x_arr)):
            x_arr[i] = augmentation_results[i][0]
            y_arr[i] = augmentation_results[i][1]
        return x_arr, y_arr

    def apply_augmentation(self, x_arr: npt.NDArray[np.float_], y_arr: npt.NDArray[np.float_], i: int) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Apply the augmentation to the data.

        :param x: Input features.
        :param y: Labels.
        :return: augmented image
        """
        xi = np.zeros(x_arr[i].shape)
        yi = np.zeros(y_arr[i].shape)
        for augmentation in self.aug:  # type: ignore[union-attr]
            if self.rng.random() < augmentation.p:
                xi, yi = augmentation.transforms(x_arr, y_arr, i)
            else:
                xi, yi = x_arr[i], y_arr[i]
        return xi, yi

    def apply_albumentation(self, image: npt.NDArray[np.float_], mask: npt.NDArray[np.float_]) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Apply the albumentation to the current image and mask.

        :param x: Input features.
        :param y: Labels.
        :return: augmented data
        """
        transformed_dict = self.alb(image=image, mask=mask)  # type: ignore[misc]
        return transformed_dict["image"], transformed_dict["mask"]

    def apply_kornia(self, x_arr: npt.NDArray[np.float_], y_arr: npt.NDArray[np.float_]) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the torchvision transforms to both the image and the mask.

        :param x: Batch of input features.
        :param y: Batch of masks.
        """
        # concatenate the x and y to apply the same transforms to both
        x_tensor = torch.from_numpy(x_arr)
        y_tensor = torch.from_numpy(y_arr)
        merged = torch.cat((x_tensor, y_tensor.unsqueeze(1)), dim=1).cuda()
        merged = self.korn(merged)  # type: ignore[misc]
        return merged[:, :-1, :, :], merged[:, -1, :, :]
