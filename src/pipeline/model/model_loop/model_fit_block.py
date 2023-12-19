from sklearn.base import BaseEstimator, TransformerMixin
from torch import nn
import dask.array as da
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import numpy as np
import copy
from typing import Self
from src.logging_utils.logger import logger
from typing import Any
from typing import Iterable


class ModelFitBlock(BaseEstimator, TransformerMixin):
    """Base model for the project."""

    # TODO we dont know if we are gonna use a torch scheduler or timm or smth else
    # TODO idk what type a loss function is
    def __init__(self, model: nn.Module, optimizer: torch.optim, scheduler: Any, criterion: Any) -> None:
        """
        Initialize the ModelFitBlock.

        :param model_config: The model configuration.

        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, X: da.Array, y: da.Array, train_indices: Iterable[int], test_indices: Iterable[int],
            epochs: int, batch_size: int, patience: int, to_mem_length: int) -> Self:
        """
        Train the model.

        :param X: Input features.
        :param y: Labels.
        :param epochs: Number of epochs.
        :param batch_size: Batch size.
        :param patience: Patience for early stopping.
        :param to_mem_length: Number of samples to convert to memory.
        :return: self
        """
        # split test and train based on indices
        # TODO add scheduler to the loop if it is not none
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        # make the datsets
        train_dataset = Dask2TorchDataset(X_train, y_train)
        train_dataset.index_to_mem(to_mem_length)
        test_dataset = Dask2TorchDataset(X_test, y_test)
        test_dataset.index_to_mem(to_mem_length)
        # make a dataloaders from the datasets
        trainloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True)

        # define the loss function
        criterion = self.criterion
        # define the optimizer
        optimizer = self.optimizer
        lowest_val_loss = np.inf
        # train the model
        # print the current device of the model
        print('Starting training')
        for epoch in range(epochs):
            self.model.train()
            print(epoch)
            with tqdm(trainloader, unit="batch", disable=False) as tepoch:
                for data in tepoch:
                    X_batch, y_batch = data
                    X_batch = X_batch.to(self.device).float()
                    y_batch = y_batch.to(self.device).float()
                    # forward pass
                    y_pred = self.model(X_batch).squeeze(1)
                    loss = criterion(y_pred, y_batch)

                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # print tqdm
                    tepoch.set_description(f"Epoch {epoch}")
                    tepoch.set_postfix(loss=loss.item())

            # validation on testloader
            self.model.eval()
            with torch.no_grad():
                with tqdm(testloader, unit="batch", disable=False) as tepoch:
                    for data in tepoch:
                        X_batch, y_batch = data
                        X_batch = X_batch.to(self.device).float()
                        y_batch = y_batch.to(self.device).float()
                        # forward pass
                        y_pred = self.model(X_batch).squeeze(1)
                        val_loss = criterion(y_pred, y_batch)
                        # print tqdm
                        tepoch.set_description(f"Epoch {epoch}")
                        tepoch.set_postfix(
                            loss=val_loss.item())
            # store the best model so far based on validation loss
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_model = copy.deepcopy(self.model.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    self.model.load_state_dict(best_model)
                    # trained_epochs = (epoch - early_stopping_counter + 1)
                    break

        return self

    def predict(self, X: da.Array) -> list[torch.Tensor]:
        """
        Predict on the test data.

        :param X: Input features.
        :return: Predictions.
        """
        X_dataset = Dask2TorchDataset(X, y=None)
        X_dataloader = DataLoader(
            X_dataset, batch_size=self.model_config["batch_size"], shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            with tqdm(X_dataloader, unit="batch", disable=False) as tepoch:
                for data in tepoch:
                    X_batch, _ = data
                    X_batch = X_batch.to(self.device).float()
                    # forward pass
                    y_pred = self.model(X_batch)
                    preds.append(y_pred)
        return preds

    def score(self, X: da.Array, y: da.Array) -> float:
        # TODO implement dice score here
        raise NotImplementedError



class Dask2TorchDataset(Dataset):

    def __init__(self, X: da.Array, y: da.Array | None) -> None:
        """
        Initialize the Dask2TorchDataset.

        :param X: Input features.
        :param y: Labels.
        """
        self.memX = []
        self.daskX = X
        self.memIdx = 0
        self.daskY = None
        self.memY = None
        if y is not None:
            self.memY = []
            self.daskY = y

    def __len__(self):
        """
        Return the length of the dataset.

        :return: Length of the dataset.
        """
        return self.daskX.shape[0] + len(self.memX)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
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

    def index_to_mem(self, idx: int) -> None:
        """
        Convert the dask array to numpy array and store it in memory.

        :param idx: Index of the item.
        :return: None
        """
        self.memIdx = idx
        self.memX = self.daskX[:idx].compute()
        self.daskX = self.daskX[idx:]
        if self.daskY is not None:
            self.memY = self.daskY[:idx].compute()
            self.daskY = self.daskY[idx:]
