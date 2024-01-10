"""TorchBlock class."""
import copy
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Annotated, Any, Self

import dask.array as da
import numpy as np
import torch
import wandb
from annotated_types import Gt
from sklearn.base import BaseEstimator, TransformerMixin
from torch import Tensor, nn
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.logging_utils.logger import logger
from src.pipeline.model.model_loop.model_blocks.utils.dask_dataset import Dask2TorchDataset


@dataclass
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

    model: nn.Module
    optimizer: Callable[[Iterator[Parameter]], Optimizer]
    scheduler: LRScheduler | None
    criterion: nn.Module
    epochs: Annotated[int, Gt(0)] = 10
    batch_size: Annotated[int, Gt(0)] = 32
    patience: Annotated[int, Gt(0)] = 5

    def __post_init__(self) -> None:
        """Post init function."""
        # Set the optimizer
        self.initialized_optimizer = self.optimizer(self.model.parameters())

        # Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Setting model to device: {self.device}")
        self.model.to(self.device)

        # Early stopping
        self.last_val_loss = np.inf
        self.lowest_val_loss = np.inf

    def fit(self, X: da.Array, y: da.Array, train_indices: list[int], test_indices: list[int], cache_size: int = -1) -> Self:
        """Train the model & log the train and validation losses to Weights & Biases.

        :param X: Input features.
        :param y: Labels.
        :param train_indices: Indices of the training data.
        :param test_indices: Indices of the test data.
        :param cache_size: Number of samples to load into memory.
        :return: Fitted model.
        """
        # TODO(Jasper): Add scheduler to the loop if it is not none

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
        # np.round is there to make sure we don't miss a sample due to int to float conversion
        train_dataset = Dask2TorchDataset(X_train, y_train)
        train_dataset.create_cache(cache_size if cache_size == -1 else int(np.round(cache_size * train_ratio)))
        test_dataset = Dask2TorchDataset(X_test, y_test)
        test_dataset.create_cache(cache_size if cache_size == -1 else int(np.round(cache_size * (1 - train_ratio))))

        # Create dataloaders from the datasets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda batch: (batch[0], batch[1]))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda batch: (batch[0], batch[1]))

        # Train model
        logger.info("Training the model")

        train_losses: list[float] = []
        val_losses: list[float] = []

        if wandb.run:
            wandb.define_metric("Training/Train Loss", summary="min")
            wandb.define_metric("Training/Validation Loss", summary="min")

        # TODO(Tolga): Add early stopping for train full
        # https://gitlab.ewi.tudelft.nl/dreamteam-epoch/epoch-iv/q2-detect-kelp/-/issues/38
        self.lowest_val_loss = np.inf
        if len(test_loader) == 0:
            logger.warning(f"Doing train full, early stopping is not yet implemented for this case so the model will be trained for {self.epochs} epochs")

        for epoch in range(self.epochs):
            # Train using train_loader
            train_losses.append(self._train_one_epoch(train_loader, desc=f"Epoch {epoch} Train"))

            if wandb.run:
                # Log only the train loss in the "Training" section
                wandb.log({"Training/Train Loss": train_losses[-1]}, step=epoch + 1)

            # Validate using test_loader if we have validation data
            if len(test_loader) > 0:
                self.last_val_loss = self._val_one_epoch(test_loader, desc=f"Epoch {epoch} Valid")
                val_losses.append(self.last_val_loss)

                if wandb.run:
                    # Also log the validation loss in the "Training" section
                    wandb.log({"Training/Validation Loss": val_losses[-1]}, step=epoch + 1)

                    # Log both the train and validation loss in a line plot in the "Training" section
                    wandb.log(
                        {
                            "Training/Loss": wandb.plot.line_series(
                                xs=range(epoch + 1), ys=[train_losses, val_losses], keys=["Train", "Validation"], title="Training/Loss", xname="Epoch"
                            )
                        }
                    )

                if self.early_stopping():
                    break
            else:  # Train full TODO(#38)
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
            self.initialized_optimizer.zero_grad()
            loss.backward()
            self.initialized_optimizer.step()

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
        if self.last_val_loss < self.lowest_val_loss:
            self.lowest_val_loss = self.last_val_loss
            self.best_model = copy.deepcopy(self.model.state_dict())
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.patience:
                logger.info(f"Loading best model with validation loss {self.lowest_val_loss}")
                self.model.load_state_dict(self.best_model)
                return True
        return False
