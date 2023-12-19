from sklearn.base import BaseEstimator
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

from collections.abc import Iterable


class BaseModel(BaseEstimator):
    """Base model for the project."""

    def __init__(self, model_config: dict[str, int | float | None]):
        """
        Initialize the BaseModel.

        
        """
        self.model_config = model_config
        # placeholder model for testing
        # overwrite in child class
        self.model = nn.Linear(self.model_config["input_size"], self.model_config["output_size"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config["learning_rate"])
        self.loss_fn = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, X: da.Array, y: da.Array, train_indices: Iterable[int], test_indices: Iterable[int]) -> Self:
        """
        Fit the model to the training data.

        :param X: Input features.
        :param y: Labels.
        :train_indices: Indices for training data.
        :test_indices: Indices for test data.

        :return: self.

        """
        # split test and train based on indices
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        # make the datsets
        train_dataset = Dask2TorchDataset(X_train, y_train)
        test_dataset = Dask2TorchDataset(X_test, y_test)
        # make a dataloaders from the datasets
        trainloader = DataLoader(train_dataset, batch_size=self.model_config["batch_size"], shuffle=False)
        testloader = DataLoader(test_dataset, batch_size=self.model_config["batch_size"], shuffle=False)

        # define the loss function
        loss_fn = self.loss_fn
        # define the optimizer
        optimizer = self.optimizer
        # params for early stopping
        # these should be read from the config
        # TODO parametrize w config
        lowest_val_loss = np.inf
        patience = 10
        # train the model
        for epoch in range(self.model_config["epochs"]):
            self.model.train()
            with tqdm(trainloader, unit="batch", disable=False) as tepoch:
                for data in tepoch:
                    X_batch, y_batch = data
                    X_batch = X_batch.to(self.device).float()
                    y_batch = y_batch.to(self.device).float()
                    # forward pass
                    y_pred = self.model(X_batch)
                    loss = loss_fn(y_pred, y_batch)

                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # print tqdm
                    tepoch.set_description(f"Epoch {epoch}")
                    tepoch.set_postfix(loss=sum(loss) / (len(loss) + 1))

            # validation on testloader
            self.model.eval()
            with torch.no_grad():
                with tqdm(testloader, unit="batch", disable=False) as tepoch:
                    for data in tepoch:
                        X_batch, y_batch = data
                        X_batch = X_batch.float()
                        y_batch = y_batch.float()
                        # forward pass
                        y_pred = self.model(X_batch)
                        val_loss = loss_fn(y_pred, y_batch)
                        # print tqdm
                        tepoch.set_description(f"Epoch {epoch}")
                        tepoch.set_postfix(loss=sum(val_loss) / (len(val_loss) + 1))
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
        X_dataloader = DataLoader(X_dataset, batch_size=self.model_config["batch_size"], shuffle=False)
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
        # will calculate the dice score by running the predict function and computing the dice score
        raise NotImplementedError

    def load(self, path: str) -> Self:
        raise NotImplementedError


class Dask2TorchDataset(Dataset[Any]):
    def __init__(self, X: da.Array, y: da.Array | None):
        """
        Initialize the dataset.
        :param X: The input features.
        :param y: The target variable.
        """
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        :return: The length of the dataset.
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Get the item at the index.
        :param idx: The index of the item.
        :return: The item at the index.
        """
        X_array = self.X[idx].compute()
        if self.y is not None:
            y_array = self.y[idx].compute()
            return torch.from_numpy(X_array), torch.from_numpy(y_array)
        else:
            return torch.from_numpy(X_array)
