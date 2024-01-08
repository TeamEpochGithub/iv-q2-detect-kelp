"""Module to convert a dask array to a torch dataset."""
from typing import Any
import time
import asyncio
import dask.array as da
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
import concurrent.futures

class Dask2TorchDataset(Dataset[Any]):
    """Class to convert a dask array to a torch dataset.

    :param X: Input features.
    :param y: Labels.
    """

    def __init__(self, X: da.Array, y: da.Array | None, *, transforms: Any = None) -> None:
        """Initialize the Dask2TorchDataset.

        :param X: Input features.
        :param y: Labels.
        :param transforms: Transforms/Augmentations to apply to the data.
        """
        self.memX: npt.NDArray[np.float_] = np.array([])
        self.daskX = X
        self.memIdx = 0
        self.memY: npt.NDArray[np.float_] | None
        self.daskY: da.Array | None
        if y is not None:
            self.memY = np.array([])
            self.daskY = y
        else:
            self.daskY = None
            self.memY = None

        self.transforms = transforms

    def __len__(self) -> int:
        """Return the length of the dataset.

        :return: Length of the dataset.
        """
        return self.daskX.shape[0] + len(self.memX)

    def __getitems__(self, idxs: list[int]) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Implement the index_to_mem method to update the memory index and compute the memory and dask arrays accordingly.

        :param idxs: list of indices to get
        :return: Item at the given index.
        """
        # Get the indices for the in mem and not in mem items
        not_in_mem_idxs = [idxs[i] - self.memIdx for i in range(len(idxs)) if idxs[i] >= len(self.memX)]
        in_mem_idxs = [idxs[i] for i in range(len(idxs)) if idxs[i] < len(self.memX)]

        # Compute the not in mem items and concat with the ones already in mem
        if len(not_in_mem_idxs) > 0:
            x_arr = np.concatenate((self.memX[in_mem_idxs], self.daskX[not_in_mem_idxs].compute()), axis=0)
        else:
            x_arr = self.memX[in_mem_idxs]

        if self.daskY is not None and self.memY is not None:
            # if y exists do the same for y
            if len(not_in_mem_idxs) > 0:
                y_arr = np.concatenate((self.memY[in_mem_idxs], self.daskY[not_in_mem_idxs].compute()), axis=0)
            else:
                y_arr = self.memY[in_mem_idxs]

            # Apply the transforms with dask.delayed to parallelize it
            if self.transforms is not None:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    loop = asyncio.get_event_loop()
                    futures = [loop.run_in_executor(executor, self.apply_augmentation, x_arr[i].transpose(1,2,0), y_arr[i]) for i in range(len(x_arr))]
                    looper = asyncio.gather(*futures)
                augmentation_results = loop.run_until_complete(looper)
                for i in range(len(x_arr)):
                    x_arr[i] = augmentation_results[i][0].transpose(2,0,1)
                    y_arr[i] = augmentation_results[i][1]
            return torch.from_numpy(x_arr), torch.from_numpy(y_arr)

        # If y does not exist, return only x
        # Apply the transforms
        if self.transforms is not None:
            for i in range(len(x_arr)):
                transformed_dict = self.transforms(image=x_arr[i].transpose(1,2,0))
                x_arr[i] = transformed_dict['image'].transpose(2,0,1)
        return torch.from_numpy(x_arr)

    def create_cache(self, size: int) -> None:
        """Convert part of the dask array to numpy and store it in memory.

        :param size: Maximum number of samples to load into memory. If -1, load all samples.
        """
        if size == -1 or size >= self.daskX.shape[0]:
            idx = self.daskX.shape[0]
        else:
            idx = size
        self.memX = self.daskX[:idx].compute()
        self.daskX = self.daskX[idx:]
        if self.daskY is not None:
            self.memY = self.daskY[:idx].compute()
            self.daskY = self.daskY[idx:]


    def apply_augmentation(self, image, mask) -> dict:
        """Apply the augmentation to the data.

        :param x: Input features.
        :param y: Labels.
        :return: augmented data
        """
        transformed_dict = self.transforms(image=image, mask=mask)
        return transformed_dict['image'], transformed_dict['mask']