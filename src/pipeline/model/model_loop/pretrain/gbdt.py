"""Feature column consisting of per-pixel predictions of a GBDT model."""
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import catboost
import dask.array as da
import numpy as np
from numpy import typing as npt
from sklearn.model_selection import train_test_split

from src.logging_utils.logger import logger
from src.pipeline.model.model_loop.pretrain.pretrain_block import PretrainBlock

if sys.version_info < (3, 11):  # Self was added in Python 3.11
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class GBDT(PretrainBlock):
    """Add feature column consisting of per-pixel predictions of a GBDT model.

    :param max_images: The maximum number of images to use for training. If None, all training images will be used.
    :param test_split: The test split to use for early stopping. Split will be made amongst training images.
    """

    max_images: int | None = None
    early_stopping_split: float = 0.2

    def __post_init__(self) -> None:
        """Initialize the GBDT model."""
        self.trained_model = None

    def fit(self, X: da.Array, y: da.Array, train_indices: list[int], *, save_pretrain: bool = True, save_pretrain_with_split: bool = False) -> Self:
        """Fit the model.

        :param X: The data to fit
        :param y: The target variable
        :return: The fitted transformer
        """
        if save_pretrain_with_split:
            self.train_split_hash(train_indices=train_indices)
        if Path(f"tm/{self.prev_hash}.gbdt").exists() and save_pretrain:
            logger.info(f"GBDT already exists at {f'tm/{self.prev_hash}.gbdt'}")
            return self

        logger.info("Fitting GBDT...")
        start_time = time.time()

        # Select images to train on
        train_indices.sort()
        if self.max_images and len(train_indices) > self.max_images:
            train_indices = train_indices[: self.max_images]
        logger.info(f"Using {len(train_indices)} images")

        X = X[train_indices]
        y = y[train_indices]

        # Reshape X from (N, C, H, W) to (N*H*W, C), ensuring that channels is the last dimension
        X = X.transpose((0, 2, 3, 1)).reshape((-1, X.shape[1]))

        # Reshape y from (N, H, W) to (N*H*W,)
        y = y.reshape((-1,))

        # Convert to numpy (will trigger all computations)
        X = X.compute()
        y = y.compute()
        logger.info(f"Computed X and y, shape: {X.shape}, {y.shape} in {time.time() - start_time} seconds")

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.early_stopping_split)

        # Fit the catboost model
        # Check if labels are continuous or binary
        cbm = catboost.CatBoostRegressor(iterations=100, verbose=True, early_stopping_rounds=10)
        cbm.fit(X_train, y_train, eval_set=(X_test, y_test))

        logger.info(f"Fitted GBDT in {time.time() - start_time} seconds total")

        # Save the model
        self.trained_model = cbm
        if save_pretrain:
            cbm.save_model(f"tm/{self.prev_hash}.gbdt")
            logger.info(f"Saved GBDT to {f'tm/{self.prev_hash}.gbdt'}")

        return self

    def transform(self, X: da.Array) -> da.Array:
        """Transform the data. This will load the model from disk and add a column for each pixel.

        :param X: The data to transform
        :param y: The target variable
        :return: The transformed data
        """
        logger.info("Transforming with GBDT...")

        # Load the model
        cbm = catboost.CatBoostRegressor()
        if self.trained_model is None:
            # Verify that the model exists
            if not Path(f"tm/{self.prev_hash}.gbdt").exists():
                raise ValueError(f"GBDT does not exist, cannot find {f'tm/{self.prev_hash}.gbdt'}")

            cbm.load_model(f"tm/{self.prev_hash}.gbdt")
            logger.info(f"Loaded GBDT from {f'tm/{self.prev_hash}.gbdt'}")
        else:
            cbm = self.trained_model

        X = X.rechunk({0: "auto", 1: -1, 2: -1, 3: -1})

        # Predict in parallel with dask map blocks
        def predict(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            # x has shape (B, C, H, W), transpose and reshape for catboost to (N, C)
            x_ = x.transpose((0, 2, 3, 1)).reshape((-1, x.shape[1]))

            # Predict and reshape back to (N, 1, H, W)
            pred = cbm.predict(x_).reshape((x.shape[0], 1, x.shape[2], x.shape[3]))
            return np.concatenate([x, pred], axis=1)

        return X.map_blocks(predict, dtype=np.float32, chunks=(X.chunks[0], (X.chunks[1][0] + 1,), X.chunks[2], X.chunks[3]), meta=np.array((), dtype=np.float32))
