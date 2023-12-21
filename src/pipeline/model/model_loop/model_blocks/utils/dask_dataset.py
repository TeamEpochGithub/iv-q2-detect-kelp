from torch.utils.data import Dataset
import dask.array as da
import torch
from typing import Any
import numpy as np


class Dask2TorchDataset(Dataset[Any]):
    """
    Class to convert a dask array to a torch dataset.
    :param X: Input features.
    :param y: Labels.
    """
    def __init__(self, X: da.Array, y: da.Array | None) -> None:
        """
        Initialize the Dask2TorchDataset.

        :param X: Input features.
        :param y: Labels.
        """
        self.memX: list[torch.Tensor] = []
        self.daskX = X
        self.memIdx = 0
        self.memY: list[torch.Tensor] | None
        self.daskY: da.Array | None
        if y is not None:
            self.memY = []
            self.daskY = y
        else:
            self.daskY = None
            self.memY = None

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        :return: Length of the dataset.
        """
        return self.daskX.shape[0] + len(self.memX)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Implement the index_to_mem method to update the memory index and compute the memory and dask arrays accordingly.

        :param idx: Index of the item.
        :return: Item at the given index.
        """
        if idx < len(self.memX):
            if self.memY is not None:
                return torch.from_numpy(self.memX[idx]), torch.from_numpy(self.memY[idx])
            else:
                return torch.from_numpy(self.memX[idx])
        else:
            x_arr = self.daskX[idx - self.memIdx].compute()
            if self.daskY is not None:
                y_arr = self.daskY[idx - self.memIdx].compute()
                return torch.from_numpy(x_arr), torch.from_numpy(y_arr)
            else:
                return torch.from_numpy(x_arr)

    def __getitems__(self, idxs):
            # when index to mem is called some indices will be in memory already
            # get all indices larger than the memory index
            not_in_mem_idxs = [idxs[i] - self.memIdx for i in range(len(idxs)) if idxs[i] >= len(self.memX)]
            in_mem_idxs = [idxs[i] for i in range(len(idxs)) if idxs[i] < len(self.memX)]

            x_arr = self.daskX[not_in_mem_idxs].compute()
            x_arr = np.concatenate((self.memX[in_mem_idxs], x_arr), axis=0)
            if self.daskY is not None:
                y_arr = self.daskY[not_in_mem_idxs].compute()
                y_arr = np.concatenate((self.memY[in_mem_idxs], y_arr), axis=0)
                return torch.from_numpy(x_arr), torch.from_numpy(y_arr)
            else:
                return torch.from_numpy(x_arr)

    def index_to_mem(self, idx: int) -> None:
        """
        Convert the dask array to numpy array and store it in memory.

        :param idx: Index of the item.
        """
        self.memIdx = idx
        self.memX = self.daskX[:idx].compute()
        self.daskX = self.daskX[idx:]
        if self.daskY is not None:
            self.memY = self.daskY[:idx].compute()
            self.daskY = self.daskY[idx:]
