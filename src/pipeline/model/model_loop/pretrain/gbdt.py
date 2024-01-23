"""Feature column consisting of per-pixel predictions of a GBDT model."""
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import catboost
import dask.array as da
import numpy as np
from lightgbm import LGBMClassifier  # type: ignore[import-not-found]
from numpy import typing as npt
from xgboost import XGBClassifier  # type: ignore[import-not-found]

from src.logging_utils.logger import logger
from src.pipeline.model.model_loop.pretrain.pretrain_block import PretrainBlock
from src.scoring.dice_coefficient import DiceCoefficient

if sys.version_info < (3, 11):  # Self was added in Python 3.11
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class GBDT(PretrainBlock):
    """Add feature column consisting of per-pixel predictions of a GBDT model.

    :param max_images: The maximum number of images to use for training. If None, all training images will be used.
    :param test_split: The test split to use for early stopping. Split will be made amongst training images.
    :param model_type: The type of GBDT model to use. Currently only catboost is supported.
    """

    max_images: int | None = None
    early_stopping_split: float = 0.2
    model_type: str = "XGBoost"

    def __post_init__(self) -> None:
        """Initialize the GBDT model."""
        self.trained_model = None
        # Check if type is valid
        if self.model_type not in ["Catboost", "XGBoost", "LightGBM"]:
            raise ValueError(f"Invalid model type {self.model_type}. Please choose from ['Catboost', 'XGBoost', 'LightGBM']")

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
        y = y.compute() > 0.5  # ensure binary labels

        logger.info(f"Computed X and y, shape: {X.shape}, {y.shape} in {time.time() - start_time} seconds")

        # Split into train and test without shuffling
        train_size = int((1 - self.early_stopping_split) * len(X))
        X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

        # Fit the catboost model
        # Check if labels are continuous or binary
        if self.model_type == "Catboost":
            logger.info("Fitting catboost model...")
            model = catboost.CatBoostClassifier(iterations=100, verbose=True, early_stopping_rounds=10)
            model.fit(X_train, y_train, eval_set=(X_test, y_test))
        elif self.model_type == "XGBoost":
            logger.info("Fitting XGBoost model...")
            model = XGBClassifier(n_estimators=100, n_jobs=-1, early_stopping_rounds=10)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        elif self.model_type == "LightGBM":
            logger.info("Fitting LightGBM model...")
            model = LGBMClassifier(n_estimators=100, n_jobs=-1, early_stopping_rounds=10, verbose=1, num_iterations=50)
            model.fit(X_train, y_train, eval_set=(X_test, y_test))

        score = DiceCoefficient()(y_test, model.predict_proba(X_test)[:, 1])
        logger.info(f"Score of fitted model: {score}")

        logger.info(f"Fitted GBDT in {time.time() - start_time} seconds total")

        # Save the model
        self.trained_model = model
        if save_pretrain:
            with open(f"tm/{self.prev_hash}.gbdt", "wb") as f:
                pickle.dump(model, f)
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
        if self.trained_model is None:
            # Verify that the model exists
            if not Path(f"tm/{self.prev_hash}.gbdt").exists():
                raise ValueError(f"GBDT does not exist, cannot find {f'tm/{self.prev_hash}.gbdt'}")

            with open(f"tm/{self.prev_hash}.gbdt", "rb") as f:
                model = pickle.load(f)  # noqa: S301
            logger.info(f"Loaded GBDT from {f'tm/{self.prev_hash}.gbdt'}")
        else:
            model = self.trained_model

        X = X.rechunk({0: "auto", 1: -1, 2: -1, 3: -1})

        # Predict in parallel with dask map blocks
        def predict(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            # x has shape (B, C, H, W), transpose and reshape for catboost to (N, C)
            x_ = x.transpose((0, 2, 3, 1)).reshape((-1, x.shape[1]))

            # Predict and reshape back to (N, 1, H, W)
            pred = model.predict_proba(x_)[:, 1].reshape((x.shape[0], 1, x.shape[2], x.shape[3]))
            return np.concatenate([x, pred], axis=1)

        return X.map_blocks(predict, dtype=np.float32, chunks=(X.chunks[0], (X.chunks[1][0] + 1,), X.chunks[2], X.chunks[3]), meta=np.array((), dtype=np.float32))
