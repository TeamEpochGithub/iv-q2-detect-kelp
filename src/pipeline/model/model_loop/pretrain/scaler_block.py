"""Scaler block to fit and transform the data."""

from pathlib import Path
import sys
from dataclasses import dataclass

import dask
import dask.array as da
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_utils.logger import logger
from joblib import hash

if sys.version_info < (3, 11):  # Self was added in Python 3.11
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class ScalerBlock(BaseEstimator, TransformerMixin):
    """Scaler block to fit and transform the data.

    :param scaler: Scaler.
    """

    scaler: BaseEstimator

    def __post_init__(self) -> None:
        """Post init function."""
        self.set_hash("")
        super().__init__()

    def fit(self, X: da.Array, y: da.Array, train_indices: list[int], save_scaler: bool = True) -> Self:
        """Fit the scaler.

        :param X: Data to fit. Shape should be (N, C, H, W)
        :param y: Target data. Shape should be (N, H, W)
        :return: Fitted scaler
        """

        # Check if the scaler exists
        if Path(f"tm/{self.prev_hash}.scaler").exists():
            logger.info("Scaler already exists, loading it")
            return self

        logger.info("Fitting scaler")
        # Save the original shape
        # Reshape X so that it is 2D
        train_indices.sort()
        # Flatten and rechunk the train data so all pixels per channels are a single row
        # The shape is (C, N*H*W) after reshaping
        X_reshaped = X[train_indices].transpose([0, 2, 3, 1]).reshape([-1, X.shape[1]])
        X_reshaped = X_reshaped.rechunk({1: X_reshaped.shape[1]})
        # Fit the scaler on the data
        self.scaler.fit(X_reshaped)
        logger.info("Fitted scaler")

        if save_scaler:
            self.save_scaler()

        return self

    def transform(self, X: da.Array) -> da.Array:
        """Transform the data.

        :param X: Data to transform. Shape should be (N, C, H, W)
        :return: Transformed data
        """

        # Load the scaler if it exists
        if not hasattr(self.scaler, 'scale_'):
            self.load_scaler()

        logger.info("Transforming the data using the scaler")

        # ignore warning about large chunks when reshaping, as we are doing it on purpose for the scalar
        # ignores type error because this is literally the example from the dask docs
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):  # type: ignore[arg-type]
            # Reshape the data to 2D
            # Flatten and rechunk all the data so all pixels per channels are a single row
            # The shape is (C, N*H*W) after reshaping
            X_reshaped = X.transpose([0, 2, 3, 1]).reshape([-1, X.shape[1]])
            X_reshaped = X_reshaped.rechunk({1: X_reshaped.shape[1]})
            # Apply the scaler
            X_reshaped = self.scaler.transform(X_reshaped)
            X = X_reshaped.reshape(X.shape[0], X.shape[2], X.shape[3], X.shape[1]).transpose([0, 3, 1, 2])
            X = X.rechunk()
        logger.info("Transformed the data using the scaler")
        return X

    def set_hash(self, prev_hash: str) -> str:
        """set_hash function sets the hash for the pipeline.

        :param prev_hash: previous hash
        :return: hash
        """
        scaler_hash = hash(str(self.scaler) + prev_hash)

        self.prev_hash = scaler_hash

        return scaler_hash

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
            logger.debug(f"Scaler from tm/{self.prev_hash}.scaler does not exist, error if saving scaler was set to true")
            return

        logger.info(f"Loading scaler from tm/{self.prev_hash}.scaler")
        self.scaler = joblib.load(f"tm/{self.prev_hash}.scaler")
        logger.info(f"Loaded scaler from tm/{self.prev_hash}.scaler")
