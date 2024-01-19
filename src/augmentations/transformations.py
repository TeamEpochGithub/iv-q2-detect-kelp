"""Base class for data augmentation transformations."""
import asyncio
import concurrent
from dataclasses import dataclass

import albumentations
import numpy as np
import numpy.typing as npt

from src.augmentations.augmentation import Augmentation
import torchvision
import torch

@dataclass
class Transformations:
    """Base class for data augmentation transformations, contains a list of augmentations to apply to the data."""

    alb: albumentations.Compose = None
    aug: list[Augmentation] | None = None
    torch: torchvision.transforms.Compose | None = None

    def transform(self, x_arr: npt.NDArray[np.float_], y_arr: npt.NDArray[np.float_]) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
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
        if self.torch is not None:
            # add the to tensor conversions to the torchvision transforms since they cant be used in the config
            self.torch = torchvision.transforms.v2.Compose([torchvision.transforms.v2.ToImage(), 
                                                            torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                                                            torchvision.transforms.v2.ToPureTensor(), 
                                                            self.torch])
            x_arr, y_arr = self.apply_torchvision(x_arr, y_arr)
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
            futures = [
                loop.run_in_executor(executor, self.apply_augmentation, x_arr.copy(), y_arr.reshape(-1, 1, y_arr.shape[1], y_arr.shape[2]).copy(), i)
                for i in range(len(x_arr))
            ]
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
        :return: augmented data
        """
        for augmentation in self.aug:  # type: ignore[union-attr]
            x_arr, y_arr = augmentation.transforms(x_arr, y_arr, i)
        return x_arr, y_arr

    def apply_albumentation(self, image: npt.NDArray[np.float_], mask: npt.NDArray[np.float_]) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Apply the albumentation to the current image and mask.

        :param x: Input features.
        :param y: Labels.
        :return: augmented data
        """
        transformed_dict = self.alb(image=image, mask=mask)
        return transformed_dict["image"], transformed_dict["mask"]
    
    def apply_torchvision(self, x_arr: npt.NDArray[np.float_], y_arr: npt.NDArray[np.float_]) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Apply the torchvision transforms to both the image and the mask.

        :param x: Batch of input features.
        :param y: Batch of masks.
        """
        
        for i in range(len(x_arr)):
            x_arr[i] = self.torch(x_arr[i]).permute(1, 2, 0)
            y_arr[i] = self.torch(y_arr[i])
        return x_arr, y_arr
