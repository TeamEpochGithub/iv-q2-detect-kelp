"""Pipeline step for a PyTorch Model."""
import copy
import functools
import gc
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import dask.array as da
import numpy as np
import numpy.typing as npt
import torch
from annotated_types import Gt, Interval
from joblib import hash
from scipy.ndimage import distance_transform_edt
from sklearn.base import BaseEstimator, TransformerMixin
from torch import Tensor, nn
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.augmentations.transformations import Transformations
from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator
from src.modules.models.custom_data_parallel import CustomDataParallel
from src.pipeline.model.model_loop.model_blocks.utils.collate_fn import collate_fn
from src.pipeline.model.model_loop.model_blocks.utils.dask_dataset import Dask2TorchDataset
from src.pipeline.model.model_loop.model_blocks.utils.reverse_transform import reverse_transform
from src.pipeline.model.model_loop.model_blocks.utils.torch_layerwise_lr import torch_layerwise_lr_groups
from src.pipeline.model.model_loop.model_blocks.utils.transform_batch import transform_batch

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


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
    :param test_size: The relative size of the test set ∈ [0, 1].
    :param transformations: Transformations to apply to the data.
    :param layerwise_lr_decay: Layerwise learning rate decay.
    """

    model: nn.Module
    optimizer: functools.partial[Optimizer]
    scheduler: Callable[[Optimizer], LRScheduler] | None
    criterion: nn.Module
    epochs: Annotated[int, Gt(0)] = 10
    batch_size: Annotated[int, Gt(0)] = 32
    patience: Annotated[int, Gt(0)] = 5
    # noinspection PyTypeHints
    test_size: Annotated[float, Interval(ge=0, le=1)] = 0.2  # Hashing purposes
    best_model_state_dict: Mapping[str, Any] = field(default_factory=dict, init=False, repr=False)
    transformations: Transformations | None = None
    layerwise_lr_decay: float | None = None
    self_ensemble: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Post init function."""
        self.set_hash("")

        self.save_model_to_disk = True

        if self.layerwise_lr_decay is None:
            # Apply the optimizer to all parameters at once
            self.initialized_optimizer = self.optimizer(self.model.parameters())
        else:
            # Make a dummy optimizer to extract the base learning rate
            dummy_optimizer = self.optimizer([Parameter(torch.zeros(1))])
            base_lr = dummy_optimizer.defaults["lr"]

            # Apply the optimizer to each layer with a different learning rate
            param_groups = torch_layerwise_lr_groups(self.model, base_lr, self.layerwise_lr_decay)
            self.initialized_optimizer = self.optimizer(param_groups)

        # Set the scheduler
        self.initialized_scheduler: LRScheduler | None
        if self.scheduler is not None:
            self.initialized_scheduler = self.scheduler(self.initialized_optimizer)
        else:
            self.initialized_scheduler = None

        # Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Setting model to device: {self.device}")

        # if multiple GPUs are available, distribute the batch size over the GPUs
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = CustomDataParallel(self.model)

        self.model.to(self.device)

        # Early stopping
        self.last_val_loss = np.inf
        self.lowest_val_loss = np.inf
        # created_fig = False
        # if not created_fig:
        #     writer = SummaryWriter('runs/model_visualization')
        #     dummy_input = torch.randn(1, 13, 256, 256).cuda()
        #     writer.add_graph(self.model, dummy_input)
        #     writer.close()
        #     created_fig = True

    def fit(self, X: da.Array, y: da.Array, train_indices: list[int], test_indices: list[int], cache_size: int = -1, *, save_model: bool = True) -> Self:
        """Train the model & log the train and validation losses to Weights & Biases.

        :param X: Input features.
        :param y: Labels.
        :param train_indices: Indices of the training data.
        :param test_indices: Indices of the test data.
        :param cache_size: Number of samples to load into memory.
        :param save_model: Whether to save the model to disk.
        :return: Fitted model.
        """
        # Check if the model exists
        self.save_model_to_disk = save_model
        if Path(f"tm/{self.prev_hash}.pt").exists() and save_model:
            logger.info(f"Model exists at tm/{self.prev_hash}.pt, skipping training")
            return self

        # Train the model with self.model named model, print model name to print_section_separator
        # Print the model name to print_section_separator
        print_section_separator(f"Training model: {self.model.__class__.__name__}")
        logger.debug(f"Training model: {self.model.__class__.__name__}")

        train_indices.sort()
        test_indices.sort()

        # Add distance maps to y
        dist_map = distance_transform_edt(~(y.compute().astype(np.int8))) / 700
        # dist_map = da.from_array(dist_map, chunks=dist_map.shape)

        y = np.concatenate((y.compute()[:, None], dist_map[:, None]), axis=1)

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
        start_time = time.time()
        train_dataset = Dask2TorchDataset(X_train, y_train, transforms=self.transformations)
        logger.info(f"Created train dataset in {time.time() - start_time} seconds")
        start_time = time.time()
        train_dataset.create_cache(cache_size if cache_size == -1 else int(np.round(cache_size * train_ratio)))
        logger.info(f"Created train cache in {time.time() - start_time} seconds")
        start_time = time.time()
        test_dataset = Dask2TorchDataset(X_test, y_test)
        logger.info(f"Created test dataset in {time.time() - start_time} seconds")
        start_time = time.time()
        test_dataset.create_cache(cache_size if cache_size == -1 else int(np.round(cache_size * (1 - train_ratio))))
        logger.info(f"Created test cache in {time.time() - start_time} seconds")

        # Create dataloaders from the datasets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)  # type: ignore[arg-type]
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)  # type: ignore[arg-type]

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

        # Train the model
        self._training_loop(train_loader, test_loader, train_losses, val_losses)
        logger.info("Done training the model")

        if self.best_model_state_dict:
            logger.info(f"Reverting to model with best validation loss {self.lowest_val_loss}")
            self.model.load_state_dict(self.best_model_state_dict)

        if save_model:
            self.save_model()

        # Empty data from memory
        train_dataset.empty_cache()
        test_dataset.empty_cache()
        del train_dataset
        del test_dataset

        return self

    def _training_loop(
        self,
        train_loader: DataLoader[tuple[Tensor, Tensor]],
        test_loader: DataLoader[tuple[Tensor, Tensor]],
        train_losses: list[float],
        val_losses: list[float],
    ) -> None:
        """Training loop for the model.

        :param train_loader: Dataloader for the testing data.
        :param test_loader: Dataloader for the training data. (can be empty)
        :param train_losses: List of train losses.
        :param val_losses: List of validation losses.
        """
        for epoch in range(self.epochs):
            # Train using train_loader
            train_loss = self._train_one_epoch(train_loader, epoch)
            logger.debug(f"Epoch {epoch} Train Loss: {train_loss}")
            train_losses.append(train_loss)

            # Log train loss
            if wandb.run:
                wandb.log({"Training/Train Loss": train_losses[-1]}, step=epoch + 1)

            # Compute validation loss
            if len(test_loader) > 0:
                self.last_val_loss = self._val_one_epoch(test_loader, desc=f"Epoch {epoch} Valid")
                logger.debug(f"Epoch {epoch} Valid Loss: {self.last_val_loss}")
                val_losses.append(self.last_val_loss)

                # Log validation loss and plot train/val loss against each other
                if wandb.run:
                    wandb.log({"Training/Validation Loss": val_losses[-1]}, step=epoch + 1)
                    wandb.log(
                        {
                            "Training/Loss": wandb.plot.line_series(
                                xs=range(epoch + 1),
                                ys=[train_losses, val_losses],
                                keys=["Train", "Validation"],
                                title="Training/Loss",
                                xname="Epoch",
                            ),
                        },
                    )

                # TODO(#38): Train full early stopping
                if self.early_stopping():
                    # Log the trained epochs - patience to wandb
                    if wandb.run:
                        wandb.log({"Epochs": (epoch + 1) - self.patience})
                    break
            elif wandb.run:
                # Log the trained epochs to wandb if we finished training
                wandb.log({"Epochs": epoch + 1})

    def _train_one_epoch(self, dataloader: DataLoader[tuple[Tensor, Tensor]], epoch: int) -> float:
        """Train the model for one epoch.

        :param dataloader: Dataloader for the training data.
        :param epoch: Epoch number.
        :return: Average loss for the epoch.
        """
        losses = []
        self.model.train()
        pbar = tqdm(dataloader, unit="batch", desc=f"Epoch {epoch} Train")
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
            pbar.set_postfix(loss=sum(losses) / len(losses))

        # Step the scheduler
        if self.initialized_scheduler is not None:
            self.initialized_scheduler.step(epoch=epoch)

        # Remove the cuda cache
        torch.cuda.empty_cache()
        gc.collect()

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
        logger.info(f"Saving model to tm/{self.prev_hash}.pt")
        torch.save(self.model, f"tm/{self.prev_hash}.pt")
        logger.info(f"Model saved to tm/{self.prev_hash}.pt")
        self.model_is_saved = True

    def load_model(self) -> None:
        """Load the model from the tm folder."""
        # Load the model if it exists
        if not Path(f"tm/{self.prev_hash}.pt").exists():
            raise FileNotFoundError(f"Model does not exist at tm/{self.prev_hash}.pt, train the model first")

        logger.info(f"Loading model from tm/{self.prev_hash}.pt")
        checkpoint = torch.load(f"tm/{self.prev_hash}.pt")

        # Load the weights from the checkpoint
        if isinstance(checkpoint, nn.DataParallel):
            model = checkpoint.module
        else:
            model = checkpoint

        # Set the current model to the loaded model
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model.state_dict())
        else:
            self.model.load_state_dict(model.state_dict())

        logger.info(f"Model loaded from tm/{self.prev_hash}.pt")

    def predict(self, X: da.Array | npt.NDArray[np.float64], cache_size: int = -1, *, feature_map: bool = False) -> np.ndarray[Any, Any]:  # noqa: C901
        """Predict on the test data.

        :param X: Input features.
        :param cache_size: Number of samples to load into memory.
        :return: Predictions.
        """
        # Load the model if it exists
        if self.save_model_to_disk:
            self.load_model()

        print_section_separator(f"Predicting of model: {self.model.__class__.__name__}")
        logger.debug(f"Training model: {self.model.__class__.__name__}")
        logger.info(f"Predicting on the test data with {'all' if cache_size == -1 else cache_size} samples in memory")
        X_dataset = Dask2TorchDataset(X, y=None)
        start_time = time.time()
        logger.info("Loading test images into RAM...")
        X_dataset.create_cache(cache_size)
        logger.info(f"Created test cache in {time.time() - start_time} seconds")
        X_dataloader = DataLoader(X_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda batch: batch)
        self.model.eval()
        preds = []
        with torch.no_grad(), tqdm(X_dataloader, unit="batch", disable=False) as tepoch:
            for data in tepoch:
                X_batch = data
                X_batch = X_batch.to(self.device).float()

                if feature_map:
                    # forward pass
                    if hasattr(self.model, "model") and hasattr(self.model.model, "segmentation_head"):
                        self.model.model.segmentation_head = nn.Identity()
                    y_pred = self.model(X_batch).cpu().numpy()
                    preds.extend(y_pred)
                    continue

                # forward pass
                if self.self_ensemble:
                    predictions = []
                    for flip in [False, True]:
                        for rotation in range(4):
                            # Transform the batch
                            X_batch_transformed = transform_batch(X_batch.clone(), rotation, flip=flip)

                            # Get prediction
                            y_pred_transformed = self.model(X_batch_transformed)

                            # Reverse the transformation on prediction
                            y_pred_reversed = reverse_transform(y_pred_transformed, rotation, flip=flip)

                            # Collect the predictions
                            predictions.append(y_pred_reversed)

                    # Average the predictions
                    y_pred = torch.mean(torch.stack(predictions), dim=0).cpu().numpy()
                else:
                    y_pred = self.model(X_batch).cpu().numpy()

                if y_pred.shape[1] == 1:
                    preds.extend(y_pred)
                elif y_pred.shape[1] == 2:
                    y_pred = np.argmax(y_pred, axis=1)
                    preds.extend(y_pred)
                elif y_pred.shape[1] == 3:
                    regression_preds = y_pred[:, 0] > 0.5
                    classification_preds = y_pred[:, 1:].argmax(axis=1)

                    stacked_preds = np.stack([regression_preds, classification_preds], axis=1)

                    # Perform the logical OR operation on the two channels
                    union_preds = np.logical_or(stacked_preds[:, 0], stacked_preds[:, 1])

                    # Convert the boolean array to an integer array
                    union_preds = union_preds.astype(np.uint8)

                    # preds.extend(union_preds)
                    preds.extend(union_preds)
                else:
                    raise ValueError(f"Invalid number of channels in the output of the model: {y_pred.shape[1]}")

        logger.info("Done predicting")

        return np.array(preds)

    def transform(self, X: da.Array | npt.NDArray[np.float64]) -> np.ndarray[Any, Any]:
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
            self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.patience:
                logger.info("Ran out of patience, stopping early")
                return True
        return False

    def set_hash(self, prev_hash: str) -> str:
        """Set the hash.

        :param prev_hash: Previous hash
        :return: Hash
        """
        torch_block_hash = hash(str(self) + prev_hash)

        self.prev_hash = torch_block_hash

        return torch_block_hash
