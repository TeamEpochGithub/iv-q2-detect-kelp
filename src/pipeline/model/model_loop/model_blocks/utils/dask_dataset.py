"""Module to convert a dask array to a torch dataset."""
import gc
from typing import Any

import dask.array as da
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from src.augmentations.transformations import Transformations


class Dask2TorchDataset(Dataset[Any]):
    """Class to convert a dask array to a torch dataset.

    :param X: Input features.
    :param y: Labels.
    """

    def __init__(self, X: da.Array | npt.NDArray[np.float64], y: da.Array | npt.NDArray[np.float64] | None, *, transforms: Transformations | None = None) -> None:
        """Initialize the Dask2TorchDataset.

        :param X: Input features.
        :param y: Labels.
        :param transforms: Transforms/Augmentations to apply to the data.
        """
        self.memX: npt.NDArray[np.float_] = np.array([])
        self.daskX = X
        self.memIdx = 0
        self.memY: npt.NDArray[np.float_] | None
        self.daskY: da.Array | npt.NDArray[np.float64] | None
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
        if len(not_in_mem_idxs) > 0 and isinstance(self.daskX, da.Array):
            x_arr = np.concatenate((self.memX[in_mem_idxs], self.daskX[not_in_mem_idxs].compute()), axis=0)
        else:
            x_arr = self.memX[in_mem_idxs]

        if self.daskY is not None and self.memY is not None:
            # If y exists do the same for y
            if len(not_in_mem_idxs) > 0 and isinstance(self.daskY, da.Array):
                y_arr = np.concatenate((self.memY[in_mem_idxs], self.daskY[not_in_mem_idxs].compute()), axis=0)
            else:
                y_arr = self.memY[in_mem_idxs]

            # If they exist, apply the augmentations in a paralellized way using asyncio
            if self.transforms is not None:
                if y_arr.shape[1] == 1 or len(y_arr.shape) == 3:
                    x_arr, y_arr = self.transforms.transform(x_arr, y_arr)
                elif y_arr.shape[1] == 2:
                    dist_map = y_arr[:, 1, np.newaxis]

                    # Concatenate the distance map to the x_arr
                    x_arr = np.concatenate((x_arr, dist_map), axis=1)

                    x_arr, y_arr = self.transforms.transform(x_arr, y_arr[:, 0])
                    dist_map = x_arr[:, -1]
                    x_arr = x_arr[:, :-1]
                    y_arr = torch.stack((y_arr, dist_map), dim=1)

            if isinstance(x_arr, torch.Tensor) and isinstance(y_arr, torch.Tensor):
                return x_arr, y_arr
            return torch.from_numpy(x_arr), torch.from_numpy(y_arr)

        # If y doesn't exist it must be for submission and for that we don't want to augment the inference data
        # If y does not exist, return only x
        return torch.from_numpy(x_arr)

    def create_cache(self, size: int) -> None:
        """Convert part of the dask array to numpy and store it in memory.

        :param size: Maximum number of samples to load into memory. If -1, load all samples.
        """
        # If type of self.daskX is numpy array, it means that the cache is already loaded so move to self.memX
        if isinstance(self.daskX, np.ndarray) or size == -1 or size >= self.daskX.shape[0]:
            self.memX, self.daskX = self.handle_array(self.daskX, self.memX)
            if self.daskY is not None:
                self.memY, self.daskY = self.handle_array(self.daskY, self.memY)
        else:
            self.memX = self.daskX[:size].compute()
            self.daskX = self.daskX[size:]
            if self.daskY is not None:
                if isinstance(self.daskY, np.ndarray):
                    self.memY, self.daskY = self.handle_array(self.daskY, self.memY)
                else:
                    self.memY = self.daskY[:size].compute()
                    self.daskY = self.daskY[size:]

    def empty_cache(self) -> None:
        """Delete the cache."""
        self.memX = np.array([])
        self.memY = np.array([])
        gc.collect()

    def handle_array(
        self,
        dask_arr: npt.NDArray[np.float_] | da.Array,
        mem_arr: npt.NDArray[np.float_] | None,
    ) -> tuple[npt.NDArray[np.float_], da.Array | npt.NDArray[np.float64]]:
        """Handle the array to be in the correct format.

        :param dask_arr: The dask array
        :param mem_arr: The memory array
        :return: The memory and dask arrays
        """
        if isinstance(dask_arr, np.ndarray):
            mem_arr = np.array(dask_arr)
            dask_arr = da.from_array(np.array([]))
        elif isinstance(dask_arr, da.Array):
            mem_arr = dask_arr.compute()
            dask_arr = da.from_array(np.array([]))
        if mem_arr is None:
            mem_arr = np.array([])
        return mem_arr, dask_arr
