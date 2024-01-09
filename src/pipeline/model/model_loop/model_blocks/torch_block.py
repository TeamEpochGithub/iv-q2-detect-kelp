"""TorchBlock class."""
import copy
from collections.abc import Callable
from typing import Any, Self

import dask.array as da
import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator
from src.pipeline.model.model_loop.model_blocks.utils.dask_dataset import Dask2TorchDataset


class TorchBlock(BaseEstimator, TransformerMixin):
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
        optimizer: Callable[..., Optimizer],
        scheduler: LRScheduler | None,
        criterion: nn.Module,
        epochs: int = 10,
        batch_size: int = 32,
        patience: int = 5,
    ) -> None:
        """Initialize the TorchBlock.

        :param model: Model to train.
        :param optimizer: Optimizer. As partial function call so that model.parameters() can still be added.
        :param scheduler: Scheduler.
        :param criterion: Loss function.
        :param epochs: Number of epochs.
        :param batch_size: Batch size.
        :param patience: Patience for early stopping.

        """
        self.model = model
        self.optimizer = optimizer(model.parameters())
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

    def fit(self, X: da.Array, y: da.Array, train_indices: list[int], test_indices: list[int], cache_size: int = -1) -> Self:
        """Train the model.

        :param X: Input features.
        :param y: Labels.
        :param train_indices: Indices of the training data.
        :param test_indices: Indices of the test data.
        :param cache_size: Number of samples to load into memory.
        :return: Fitted model.
        """
        # TODO(Epoch): Add scheduler to the loop if it is not none
        # Train the model with self.model named model, print model name to print_section_separator
        # Print the model name to print_section_separator
        print_section_separator(f"Training model: {self.model.__class__.__name__}")
        logger.debug(f"Training model: {self.model.__class__.__name__}")

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
        # Get the ratio of train to all data
        train_ratio = len(X_train) / (len(X_test) + len(X_train))
        # Make datasets from the train and test sets
        logger.info(f"Making datasets with {'all' if cache_size == -1 else cache_size} samples in memory")
        # Setting cache size to -1 will load all samples into memory
        # If it is not -1 then it will load cache_size * train_ratio samples into memory for training
        # and cache_size * (1 - train_ratio) samples into memory for testing
        # np.round is there to make sure we dont miss a sample due to int to float conversion
        train_dataset = Dask2TorchDataset(X_train, y_train)
        train_dataset.create_cache(cache_size if cache_size == -1 else int(np.round(cache_size * train_ratio)))
        test_dataset = Dask2TorchDataset(X_test, y_test)
        test_dataset.create_cache(cache_size if cache_size == -1 else int(np.round(cache_size * (1 - train_ratio))))

        # Create dataloaders from the datasets
        trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda batch: (batch[0], batch[1]))
        testloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda batch: (batch[0], batch[1]))

        # Train model
        logger.info("Training the model")
        # TODO(#38): Add early stopping for train full
        self.lowest_val_loss = np.inf
        if len(testloader) == 0:
            logger.warning(f"Doing train full, early stopping is not yet implemented for this case so the model will be trained for {self.epochs} epochs")
        for epoch in range(self.epochs):
            # Train using trainloader
            train_loss = self._train_one_epoch(trainloader, desc=f"Epoch {epoch} Train")
            logger.debug(f"Epoch {epoch} Train Loss: {train_loss}")

            # Validate using testloader if we have validation data
            if len(testloader) > 0:
                self.val_loss = self._val_one_epoch(testloader, desc=f"Epoch {epoch} Valid")
                logger.debug(f"Epoch {epoch} Valid Loss: {self.val_loss}")

            if len(testloader) > 0:
                # not train full
                if self.early_stopping():
                    break
            else:
                # train full TODO(#38)
                pass

        return self

    def _train_one_epoch(self, dataloader: DataLoader[tuple[Tensor, Tensor]], desc: str) -> float:
        """Train the model for one epoch.

        :param dataloader: Dataloader for the training data.
        :param desc: Description for the tqdm progress bar.
        :return: Average loss for the epoch.
        """
        losses = []
        self.model.train()
        pbar = tqdm(dataloader, unit="batch")
        for batch in pbar:
            X_batch, y_batch = batch
            X_batch = X_batch.to(self.device).float()
            y_batch = y_batch.to(self.device).float()

            # Forward pass
            y_pred = self.model(X_batch).squeeze(1)
            loss = self.criterion(y_pred, y_batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print tqdm
            losses.append(loss.item())
            pbar.set_description(desc=desc)
            pbar.set_postfix(loss=sum(losses) / len(losses))
        return sum(losses) / len(losses)

    def _val_one_epoch(self, dataloader: DataLoader[tuple[Tensor, Tensor]], desc: str) -> float:
        """Compute validation loss of the model for one epoch.

        :param dataloader: Dataloader for the testing data.
        :param desc: Description for the tqdm progress bar.
        :return: Average loss for the epoch.
        """
        losses = []
        self.model.eval()
        pbar = tqdm(dataloader, unit="batch")
        with torch.no_grad():
            for batch in pbar:
                X_batch, y_batch = batch
                X_batch = X_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).float()

                # Forward pass
                y_pred = self.model(X_batch).squeeze(1)
                loss = self.criterion(y_pred, y_batch)

                # Print losses
                losses.append(loss.item())
                pbar.set_description(desc=desc)
                pbar.set_postfix(loss=sum(losses) / len(losses))
        return sum(losses) / len(losses)

    def save_model(self, model_hash: str) -> None:
        """Save the model in the tm folder.

        :param block_hash: Hash of the model pipeline
        """
        logger.info(f"Saving model to tm/{model_hash}.pt")
        torch.save(self.model.state_dict(), f"tm/{model_hash}.pt")
        logger.info(f"Model saved to tm/{model_hash}.pt")

    def load_model(self, model_hash: str) -> None:
        """Load the model from the tm folder.

        :param block_hash: Hash of the model pipeline
        """
        logger.info(f"Loading model from tm/{model_hash}.pt")
        self.model.load_state_dict(torch.load(f"tm/{model_hash}.pt"))
        logger.info(f"Model loaded from tm/{model_hash}.pt")

    def predict(self, X: da.Array, cache_size: int = -1) -> np.ndarray[Any, Any]:
        """Predict on the test data.

        :param X: Input features.
        :param cache_size: Number of samples to load into memory.
        :return: Predictions.
        """
        print_section_separator(f"Predicting of model: {self.model.__class__.__name__}")
        logger.debug(f"Training model: {self.model.__class__.__name__}")
        logger.info(f"Predicting on the test data with {'all' if cache_size == -1 else cache_size} samples in memory")
        X_dataset = Dask2TorchDataset(X, y=None)
        logger.info("Loading test images into RAM...")
        X_dataset.create_cache(cache_size)
        logger.info("Done loading test images into RAM - Starting predictions")
        X_dataloader = DataLoader(X_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda batch: batch)
        self.model.eval()
        preds = []
        with torch.no_grad(), tqdm(X_dataloader, unit="batch", disable=False) as tepoch:
            for data in tepoch:
                X_batch = data
                X_batch = X_batch.to(self.device).float()
                # forward pass
                y_pred = self.model(X_batch).cpu().numpy()
                preds.extend(y_pred)
        logger.info("Done predicting")

        return np.array(preds)

    def transform(self, X: da.Array, y: da.Array | None = None) -> np.ndarray[Any, Any]:
        """Transform method for sklearn pipeline.

        :param X: Input features.
        :param y: Labels.
        :return: Predictions and labels.
        """
        return self.predict(X)

    def early_stopping(self) -> bool:
        """Check if early stopping should be done.

        :return: Whether early stopping should be done.
        """
        # Store the best model so far based on validation loss
        if self.val_loss < self.lowest_val_loss:
            self.lowest_val_loss = self.val_loss
            self.best_model = copy.deepcopy(self.model.state_dict())
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.patience:
                logger.info("Loading best model")
                self.model.load_state_dict(self.best_model)
                return True
        return False

    def __str__(self) -> str:
        """Return the string representation of the model.

        :return: String representation of the model.
        """
        return "TorchBlock"
