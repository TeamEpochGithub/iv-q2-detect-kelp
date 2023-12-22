from typing import Any

import dask.array as da
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset


class Dask2TorchDataset(Dataset[Any]):
    """Class to convert a dask array to a torch dataset.
    :param X: Input features.
    :param y: Labels.
    """

    def __init__(self, X: da.Array, y: da.Array | None) -> None:
        """Initialize the Dask2TorchDataset.

        :param X: Input features.
        :param y: Labels.
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
        # get the indices for the in mem and not in mem items
        not_in_mem_idxs = [idxs[i] - self.memIdx for i in range(len(idxs)) if idxs[i] >= len(self.memX)]
        in_mem_idxs = [idxs[i] for i in range(len(idxs)) if idxs[i] < len(self.memX)]

        # compute the not in mem items and concat with the ones already in mem
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
            return torch.from_numpy(x_arr), torch.from_numpy(y_arr)
        else:
            return torch.from_numpy(x_arr)

    def index_to_mem(self, idx: int) -> None:
        """Convert the dask array to numpy array and store it in memory.

        :param idx: Index of the item.
        """
        self.memIdx = idx
        self.memX = self.daskX[:idx].compute()
        self.daskX = self.daskX[idx:]
        if self.daskY is not None:
            self.memY = self.daskY[:idx].compute()
            self.daskY = self.daskY[idx:]
