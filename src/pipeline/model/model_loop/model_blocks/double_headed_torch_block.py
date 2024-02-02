"""Double headed torch block."""
import gc
import time
from dataclasses import dataclass
from typing import Any

import dask.array as da
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator
from src.pipeline.model.model_loop.model_blocks.torch_block import TorchBlock
from src.pipeline.model.model_loop.model_blocks.utils.dask_dataset import Dask2TorchDataset


@dataclass
class DoubleHeadedTorchBlock(TorchBlock):
    """Double headed torch block."""

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
            y_pred_seg, y_pred_reg = self.model(X_batch)  # (B, 2, H, W), (B, 1, H, W)
            y_pred = torch.cat((y_pred_reg, y_pred_seg), dim=1)  # (B, 3, H, W)

            loss = self.criterion(y_pred, y_batch)  # Loss is a combination of regression and classification loss (see AuxiliaryLossDouble)

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
                y_pred_seg, y_pred_reg = self.model(X_batch)  # (B, 2, H, W), (B, 1, H, W)
                y_pred = torch.cat((y_pred_reg, y_pred_seg), dim=1)  # (B, 3, H, W)
                loss = self.criterion(y_pred, y_batch)  # Loss is a combination of regression and classification loss (see AuxiliaryLossDouble)

                # Print losses
                losses.append(loss.item())
                pbar.set_description(desc=desc)
                pbar.set_postfix(loss=sum(losses) / len(losses))
        return sum(losses) / len(losses)

    def predict(self, X: da.Array | npt.NDArray[np.float64], cache_size: int = -1, *, feature_map: bool = False) -> np.ndarray[Any, Any]:
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
                    self.model.remove_heads()
                    y_pred = self.model(X_batch).cpu().numpy()
                    preds.extend(y_pred)
                    continue

                # forward pass
                y_pred_seg, y_pred_reg = self.model(X_batch)
                y_pred_seg = y_pred_seg.cpu().numpy()
                y_pred_reg = y_pred_reg.cpu().numpy()

                regression_preds = y_pred_reg > 0.5
                classification_preds = np.expand_dims(y_pred_seg.argmax(axis=1), axis=1)

                stacked_preds = np.stack([regression_preds, classification_preds], axis=1)

                # Perform the logical OR operation on the two channels
                union_preds = np.logical_or(stacked_preds[:, 0], stacked_preds[:, 1])

                # Convert the boolean array to an integer array
                union_preds = union_preds.astype(np.uint8)

                preds.extend(union_preds)

        logger.info("Done predicting")

        return np.array(preds)
