"""Scaler block to fit and transform the data."""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import dask
import dask.array as da
import joblib
from sklearn.base import BaseEstimator

from src.logging_utils.logger import logger
from src.pipeline.model.model_loop.pretrain.pretrain_block import PretrainBlock

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class ScalerBlock(PretrainBlock):
    """Scaler block to fit and transform the data.

    :param scaler: Scaler.
    """

    scaler: BaseEstimator = field(default_factory=BaseEstimator)

    def __post_init__(self) -> None:
        """Post init hook."""
        super().__post_init__()
        self.cache_pretrain: bool = True

    def fit(self, X: da.Array, y: da.Array, train_indices: list[int], *, save_pretrain: bool = True, save_pretrain_with_split: bool = False) -> Self:  # noqa: ARG002
        """Fit the scaler.

        :param X: Data to fit. Shape should be (N, C, H, W).
        :param y: Target data. Shape should be (N, H, W).
        :param train_indices: Indices of the training data in X
        :param save_pretrain: Whether to save the pretrain.
        :param save_pretrain_with_split: Whether to save this block with the split.
        :return: The fitted transformer
        """
        # Check if the scaler exists
        if save_pretrain_with_split:
            self.train_split_hash(train_indices=train_indices)
        self.cache_pretrain = save_pretrain

        if Path(f"tm/{self.prev_hash}.scaler").exists() and save_pretrain:
            logger.info("Scaler already exists, loading it")
            return self

        logger.info("Fitting scaler...")
        start_time = time.time()
        # Save the original shape
        # Reshape X so that it is 2D
        train_indices.sort()
        # Flatten and rechunk the train data so all pixels per channels are a single row
        # The shape is (C, N*H*W) after reshaping
        X_reshaped = X[train_indices].transpose([0, 2, 3, 1]).reshape([-1, X.shape[1]])
        X_reshaped = X_reshaped.rechunk({1: X_reshaped.shape[1]})
        # Fit the scaler on the data
        self.scaler.fit(X_reshaped)

        if save_pretrain:
            self.save_scaler()

        logger.info("Fitted scaler in %s seconds", time.time() - start_time)
        return self

    def transform(self, X: da.Array) -> da.Array:
        """Transform the data.

        :param X: Data to transform. Shape should be (N, C, H, W)
        :return: Transformed data
        """
        # Load the scaler if it exists
        if not hasattr(self.scaler, "scale_"):
            self.load_scaler()

        # ignore warning about large chunks when reshaping, as we are doing it on purpose for the scalar
        # ignores type error because this is literally the example from the dask docs
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):  # type: ignore[arg-type]
            # Reshape the data to 2D
            # Flatten and rechunk all the data so all pixels per channels are a single row
            # The shape is (C, N*H*W) after reshaping
            X_reshaped = X.transpose([0, 2, 3, 1]).reshape([-1, X.shape[1]])
            X_reshaped = X_reshaped.rechunk({0: "auto", 1: -1})
            # Apply the scaler
            X_reshaped = self.scaler.transform(X_reshaped)
            X = X_reshaped.reshape(X.shape[0], X.shape[2], X.shape[3], X.shape[1]).transpose([0, 3, 1, 2])
            X = X.rechunk({0: "auto", 1: -1, 2: -1, 3: -1})
        logger.info("Lazily transformed the data using the scaler")
        logger.info(f"Shape of the data after transforming: {X.shape}")
        if self.cache_pretrain:
            return self.save_pretrain(X)

        return X

    def save_scaler(self) -> None:
        """Save the scaler using joblib."""
        logger.info(f"Saving scaler to tm/{self.prev_hash}.scaler")
        joblib.dump(self.scaler, f"tm/{self.prev_hash}.scaler")
        logger.info(f"Saved scaler to tm/{self.prev_hash}.scaler")

    def load_scaler(self) -> None:
        """Load the scaler using joblib."""
        # Check if the scaler exists
        if not Path(f"tm/{self.prev_hash}.scaler").exists():
            raise FileNotFoundError(f"Scaler at tm/{self.prev_hash}.scaler does not exist, train the scaler first")

        logger.info(f"Loading scaler from tm/{self.prev_hash}.scaler")
        self.scaler = joblib.load(f"tm/{self.prev_hash}.scaler")
        logger.info(f"Loaded scaler from tm/{self.prev_hash}.scaler")
