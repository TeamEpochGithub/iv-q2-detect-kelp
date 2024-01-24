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
        self.train_indices: None | list[int] = None

    def fit(self, X: da.Array, y: da.Array, train_indices: list[int], *, save_pretrain: bool = True, save_pretrain_with_split: bool = False) -> Self:  # noqa: ARG002
        """Fit the scaler.

        :param X: Data to fit. Shape should be (N, C, H, W).
        :param y: Target data. Shape should be (N, H, W).
        :param train_indices: Indices of the training data in X.
        :param save_pretrain_with_split: Whether to save this block with the split.
        :return: The fitted transformer
        """
        # Check if the scaler exists
        if save_pretrain_with_split:
            self.train_split_hash(train_indices=train_indices)
            self.save_pretrain_with_split = True
        else:
            self.save_pretrain_with_split = False
        self.train_indices = train_indices
        if Path(f"tm/{self.prev_hash}.scaler").exists() and save_pretrain:
            logger.info("Scaler already exists, loading it")
            return self

        train_indices.sort()
        logger.info("Fitting scaler...")
        start_time = time.time()
        # Fit the scaler on the data
        self.scaler.fit(X[train_indices])

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
            # Apply the scaler
            X = self.scaler.transform(X)

        logger.info("Lazily transformed the data using the scaler")
        logger.info(f"Shape of the data after transforming: {X.shape}")
        if self.train_indices is not None and self.save_pretrain_with_split:
            return self.save_pretrain(X, self.train_indices)
        return X

    def save_scaler(self) -> None:
        """Save the scaler using joblib.

        :param scaler_hash: Hash of the scaler.
        """
        logger.info(f"Saving scaler to tm/{self.prev_hash}.scaler")
        joblib.dump(self.scaler, f"tm/{self.prev_hash}.scaler")
        logger.info(f"Saved scaler to tm/{self.prev_hash}.scaler")

    def load_scaler(self) -> None:
        """Load the scaler using joblib.

        :param scaler_hash: Hash of the scaler.
        """
        # Check if the scaler exists
        if not Path(f"tm/{self.prev_hash}.scaler").exists():
            logger.error(f"Scaler at tm/{self.prev_hash}.scaler does not exist, train the scaler first")
            sys.exit(1)

        logger.info(f"Loading scaler from tm/{self.prev_hash}.scaler")
        self.scaler = joblib.load(f"tm/{self.prev_hash}.scaler")
        logger.info(f"Loaded scaler from tm/{self.prev_hash}.scaler")
