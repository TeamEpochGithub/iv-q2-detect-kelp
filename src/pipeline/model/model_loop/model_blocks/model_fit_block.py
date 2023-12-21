from sklearn.base import BaseEstimator, TransformerMixin
from torch import nn
import dask.array as da
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import copy
from typing import Self
from typing import Any
from collections.abc import Iterable
from src.pipeline.model.model_loop.model_blocks.utils.dask_dataset import Dask2TorchDataset
from joblib import hash
from src.logging_utils.logger import logger

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

    # TODO we dont know if we are gonna use a torch scheduler or timm or smth else
    # TODO idk what type a loss function or optimizer is
    def __init__(self, model: nn.Module, optimizer: Any, scheduler: Any, criterion: Any,
                 epochs: int = 10, batch_size: int = 32, patience: int = 5,) -> None:
        """
        Initialize the ModelBlock.

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

        # save model related parameters
        # we do this here so that the hash chnages based on num epochs
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Setting model to device: {self.device}")
        self.model.to(self.device)

    def fit(self, X: da.Array, y: da.Array, train_indices: Iterable[int], test_indices: Iterable[int],
            to_mem_length: int = 3000) -> Self:
        """
        Train the model.

        :param X: Input features.
        :param y: Labels.
        :param train_indices: Indices of the training data.
        :param test_indices: Indices of the test data.
        :param to_mem_length: Number of samples to load into memory.
        :return: Fitted model.
        """
        # split test and train based on indices
        # TODO add scheduler to the loop if it is not none
        logger.info("Splitting data into train and test sets")
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        # make the datsets
        logger.info(f"Making datasets with {to_mem_length} samples in memory")
        train_dataset = Dask2TorchDataset(X_train, y_train)
        train_dataset.index_to_mem(to_mem_length)
        test_dataset = Dask2TorchDataset(X_test, y_test)
        test_dataset.index_to_mem(to_mem_length)
        # make a dataloaders from the datasets
        def collate_fn(batch):
            X = batch[0]
            y = batch[1]
            return X, y
        trainloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        testloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

        # define the loss function
        criterion = self.criterion
        # define the optimizer
        optimizer = self.optimizer
        lowest_val_loss = np.inf
        # train the model
        # print the current device of the model
        
        logger.info("Training the model")
        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            val_losses = []
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
                    train_losses.append(loss.item())
                    tepoch.set_description(f"Epoch {epoch}")
                    tepoch.set_postfix(loss=sum(train_losses) / len(train_losses))

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
                        val_losses.append(val_loss.item())
                        tepoch.set_description(f"Epoch {epoch}")
                        tepoch.set_postfix(loss=sum(val_losses) / len(val_losses))
            # store the best model so far based on validation loss
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
        # save the model in the tm folder
        # TODO this is placeholder for now but this is deterministic
        block_hash = str(hash(str(self.model)))[:6] + str(hash(str(self.optimizer)))[:6] + \
            str(hash(str(self.criterion)))[:6] + str(hash(str(self.scheduler)))[:6] + '_' + \
            str(hash(self.epochs))[:6] + str(hash(self.batch_size))[:6] + \
            str(hash(self.patience))[:6]
        logger.info(f"Saving model to tm/{block_hash}.pt")
        torch.save(self.model.state_dict(), f'tm/{block_hash}.pt')
        logger.info(f"Model saved to tm/{block_hash}.pt")
        return self

    def predict(self, X: da.Array, to_mem_length: int = 3000) -> list[torch.Tensor]:
        """
        Predict on the test data.

        :param X: Input features.
        :param to_mem_length: Number of samples to load into memory.
        :return: Predictions.
        """
        logger.info("Predicting on the test data")
        X_dataset = Dask2TorchDataset(X, y=None)
        X_dataset.index_to_mem(to_mem_length)
        X_dataloader = DataLoader(
            X_dataset, batch_size=self.batch_size, shuffle=True)
        self.model.eval()
        preds = []
        with torch.no_grad():
            with tqdm(X_dataloader, unit="batch", disable=False) as tepoch:
                for data in tepoch:
                    X_batch = data
                    X_batch = X_batch.to(self.device).float()
                    # forward pass
                    y_pred = self.model(X_batch)
                    preds.append(y_pred)
        return preds

    def __str__(self) -> str:
        """
        Return the string representation of the model.

        :return: String representation of the model.
        """
        return str(self.model)

    def transform(self, X: da.Array, y: da.Array) -> list[torch.Tensor]:
        """
        transform method for sklearn pipeline
        :param X: Input features.
        :param y: Labels.
        :return: Predictions and labels.
        """
        return self.predict(X)
