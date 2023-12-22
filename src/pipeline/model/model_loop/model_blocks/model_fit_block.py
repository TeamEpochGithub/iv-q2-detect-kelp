"""ModelBlock class."""
import copy
from typing import Any, Self

import dask.array as da
import numpy as np
import torch
from joblib import hash
from sklearn.base import BaseEstimator, TransformerMixin
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.logging_utils.logger import logger
from src.pipeline.model.model_loop.model_blocks.utils.dask_dataset import Dask2TorchDataset


class ModelBlock(BaseEstimator, TransformerMixin):
    """Base model for the project.

    :param model: Model to train.
    :param optimizer: Optimizer.
    :param scheduler: Scheduler.
    :param criterion: Loss function.
    :param epochs: Number of epochs.
    :param batch_size: Batch size.
    :param patience: Patience for early stopping.
    """

    # TODO(Epoch): We dont know if we are gonna use a torch scheduler or timm or smth else
    # TODO(@Jeffrey): Idk what type a loss function or optimizer is
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        criterion: nn.Module,
        epochs: int = 10,
        batch_size: int = 32,
        patience: int = 5,
    ) -> None:
        """Initialize the ModelBlock.

        :param model: Model to train.
        :param optimizer: Optimizer.
        :param scheduler: Scheduler.
        :param criterion: Loss function.
        :param epochs: Number of epochs.
        :param batch_size: Batch size.
        :param patience: Patience for early stopping.

        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        # Save model related parameters (Done here so hash changes based on num epochs)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        # Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Setting model to device: {self.device}")
        self.model.to(self.device)

    def fit(self, X: da.Array, y: da.Array, train_indices: list[int], test_indices: list[int], to_mem_length: int = 3000) -> Self:
        """Train the model.

        :param X: Input features.
        :param y: Labels.
        :param train_indices: Indices of the training data.
        :param test_indices: Indices of the test data.
        :param to_mem_length: Number of samples to load into memory.
        :return: Fitted model.
        """
        # TODO(Epoch): Add scheduler to the loop if it is not none

        train_indices.sort()
        test_indices.sort()

        # Rechunk the data
        logger.info("Rechunking the data")
        X = X.rechunk((1, -1, -1, -1))
        y = y.rechunk((1, -1, -1))

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        # Make datasets from the train and test sets
        logger.info(f"Making datasets with {to_mem_length} samples in memory")
        train_dataset = Dask2TorchDataset(X_train, y_train)
        train_dataset.index_to_mem(to_mem_length)
        test_dataset = Dask2TorchDataset(X_test, y_test)
        test_dataset.index_to_mem(to_mem_length)

        # Create dataloaders from the datasets
        trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda batch: (batch[0], batch[1]))
        testloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda batch: (batch[0], batch[1]))

        # Define the loss function
        criterion = self.criterion

        # Define the optimizer
        optimizer = self.optimizer
        lowest_val_loss = np.inf

        # Train model
        logger.info("Training the model")
        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            val_losses = []

            # Train using trainloader
            with tqdm(trainloader, unit="batch", disable=False) as tepoch:
                for data in tepoch:
                    X_batch, y_batch = data
                    X_batch = X_batch.to(self.device).float()
                    y_batch = y_batch.to(self.device).float()

                    # Forward pass
                    y_pred = self.model(X_batch).squeeze(1)
                    loss = criterion(y_pred, y_batch)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Print tqdm
                    train_losses.append(loss.item())
                    tepoch.set_description(f"Epoch {epoch}")
                    tepoch.set_postfix(loss=sum(train_losses) / len(train_losses))

            # Validate using testloader
            self.model.eval()
            with torch.no_grad(), tqdm(testloader, unit="batch", disable=False) as tepoch:
                for data in tepoch:
                    X_batch, y_batch = data
                    X_batch = X_batch.to(self.device).float()
                    y_batch = y_batch.to(self.device).float()

                    # Forward pass
                    y_pred = self.model(X_batch).squeeze(1)
                    val_loss = criterion(y_pred, y_batch)

                    # Print tqdm
                    val_losses.append(val_loss.item())
                    tepoch.set_description(f"Epoch {epoch}")
                    tepoch.set_postfix(loss=sum(val_losses) / len(val_losses))

            # Store the best model so far based on validation loss
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_model = copy.deepcopy(self.model.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    self.model.load_state_dict(best_model)
                    # trained_epochs = (epoch - early_stopping_counter + 1)
                    break

        # Save the model in the tm folder
        # TODO(Epoch): This is placeholder for now but this is deterministic
        self.save_model()

        return self

    def save_model(self) -> None:
        """Save the model in the tm folder."""
        block_hash = (
            str(hash(str(self.model)))[:6]
            + str(hash(str(self.optimizer)))[:6]
            + str(hash(str(self.criterion)))[:6]
            + str(hash(str(self.scheduler)))[:6]
            + "_"
            + str(hash(self.epochs))[:6]
            + str(hash(self.batch_size))[:6]
            + str(hash(self.patience))[:6]
        )
        logger.info(f"Saving model to tm/{block_hash}.pt")
        torch.save(self.model.state_dict(), f"tm/{block_hash}.pt")
        logger.info(f"Model saved to tm/{block_hash}.pt")

    def predict(self, X: da.Array, to_mem_length: int = 3000) -> list[torch.Tensor]:
        """Predict on the test data.

        :param X: Input features.
        :param to_mem_length: Number of samples to load into memory.
        :return: Predictions.
        """
        logger.info(f"Predicting on the test data with {to_mem_length} samples in memory")
        X_dataset = Dask2TorchDataset(X, y=None)
        X_dataset.index_to_mem(to_mem_length)
        X_dataloader = DataLoader(X_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda batch: (batch[0]))
        self.model.eval()
        preds = []
        with torch.no_grad(), tqdm(X_dataloader, unit="batch", disable=False) as tepoch:
            for data in tepoch:
                X_batch = data
                X_batch = X_batch.to(self.device).float()
                # forward pass
                y_pred = self.model(X_batch)
                preds.append(y_pred)
        logger.info("Done predicting")
        return preds

    def transform(self, X: da.Array, y: da.Array | None = None) -> list[torch.Tensor]:
        """Transform method for sklearn pipeline.

        :param X: Input features.
        :param y: Labels.
        :return: Predictions and labels.
        """
        return self.predict(X)

    def __str__(self) -> str:
        """Return the string representation of the model.

        :return: String representation of the model.
        """
        return "ModelBlock"
