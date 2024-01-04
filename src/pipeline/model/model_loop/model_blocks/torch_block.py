"""TorchBlock class."""
import copy
from collections.abc import Callable, Iterator
from typing import Annotated, Self

import dask.array as da
import numpy as np
import torch
from annotated_types import Gt
from joblib import hash
from sklearn.base import BaseEstimator, TransformerMixin
from torch import Tensor, nn
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.logging_utils.logger import logger
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

    # TODO(Jasper): We dont know if we are gonna use a torch scheduler or timm or smth else
    def __init__(
        self,
        model: nn.Module,
        optimizer: Callable[[Iterator[Parameter]], Optimizer],
        scheduler: LRScheduler | None,
        criterion: nn.Module,
        epochs: Annotated[int, Gt(0)] = 10,
        batch_size: Annotated[int, Gt(0)] = 32,
        patience: Annotated[int, Gt(0)] = 5,
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

        # Make datasets from the train and test sets
        logger.info(f"Making datasets with {'all' if cache_size == -1 else cache_size} samples in memory")
        train_dataset = Dask2TorchDataset(X_train, y_train)
        train_dataset.create_cache(cache_size)
        test_dataset = Dask2TorchDataset(X_test, y_test)
        test_dataset.create_cache(cache_size)

        # Create dataloaders from the datasets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda batch: (batch[0], batch[1]))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda batch: (batch[0], batch[1]))

        # Train model
        logger.info("Training the model")
        lowest_val_loss = np.inf

        train_losses: list[float] = []
        val_losses: list[float] = []
        for epoch in range(self.epochs):
            # Train using trainloader
            train_losses.append(self._train_one_epoch(train_loader, desc=f"Epoch {epoch} Train"))

            # Validate using testloader
            val_losses.append(self._val_one_epoch(test_loader, desc=f"Epoch {epoch} Valid"))

            if wandb.run:
                # Log the metrics in a line plot in the "Training" section
                wandb.log(
                    {"Training/Loss": wandb.plot.line_series(xs=range(epoch + 1), ys=[train_losses, val_losses], keys=["Train", "Validation"], title="Loss", xname="Epoch")}
                )

                # Log the metrics in separate charts in the "Training" section
                wandb.log({"Training/Train Loss": train_losses[-1], "Training/Validation Loss": val_losses[-1]}, step=epoch + 1)

            # Store the best model so far based on validation loss
            if val_losses[-1] < lowest_val_loss:
                lowest_val_loss = val_losses[-1]
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
        # TODO(Jasper): This is placeholder for now but this is deterministic
        self.save_model()

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

    def predict(self, X: da.Array, cache_size: int = -1) -> list[torch.Tensor]:
        """Predict on the test data.

        :param X: Input features.
        :param cache_size: Number of samples to load into memory.
        :return: Predictions.
        """
        logger.info(f"Predicting on the test data with {'all' if cache_size == -1 else cache_size} samples in memory")
        X_dataset = Dask2TorchDataset(X, y=None)
        X_dataset.create_cache(cache_size)
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
        return "TorchBlock"
