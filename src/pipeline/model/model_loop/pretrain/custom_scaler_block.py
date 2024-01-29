"""Scaler block to fit and transform the data."""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import dask.array as da
from sklearn.base import BaseEstimator

from src.logging_utils.logger import logger
from src.pipeline.model.model_loop.pretrain.scaler_block import ScalerBlock

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class CustomScalerBlock(ScalerBlock):
    """Scaler block to fit and transform the data.

    :param scaler: Scaler.
    """

    scaler: BaseEstimator = field(default_factory=BaseEstimator)

    def __post_init__(self) -> None:
        """Post init hook."""
        super().__post_init__()
        self.cache_pretrain: bool = False

    def fit(self, X: da.Array, y: da.Array, train_indices: list[int], *, save_pretrain: bool = True, save_pretrain_with_split: bool = False) -> Self:  # noqa: ARG002
        """Fit the scaler.

        :param X: Data to fit. Shape should be (N, C, H, W).
        :param y: Target data. Shape should be (N, H, W).
        :param train_indices: Indices of the training data in X.
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

        # Apply the scaler
        X = self.scaler.transform(X)

        logger.info("Lazily transformed the data using the scaler")
        logger.info(f"Shape of the data after transforming: {X.shape}")
        if self.cache_pretrain:
            return self.save_pretrain(X)
        return X
