"""Scaler block to fit and transform the data."""

from typing import Self

import dask.array as da
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_utils.logger import logger


class ScalerBlock(BaseEstimator, TransformerMixin):
    """Scaler block to fit and transform the data.

    :param scaler: Scaler.
    """

    def __init__(self, scaler: BaseEstimator) -> None:
        """Initialize the ScalerBlock.

        :param scaler: Scaler.
        """
        self.scaler = scaler

    def fit(self, X: da.Array, y: da.Array, train_indices: list[int]) -> Self:
        """Fit the scaler.

        :param X: Data to fit. Shape should be (N, C, H, W)
        :param y: Target data. Shape should be (N, H, W)
        :return: Fitted scaler
        """
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
        return self

    def save_scaler(self, scaler_hash: str) -> None:
        """Save the scaler using joblib.

        :param scaler_hash: Hash of the scaler.
        """
        logger.info(f"Saving scaler from tm/{scaler_hash}.scaler")
        joblib.dump(self.scaler, f"tm/{scaler_hash}.scaler")
        logger.info(f"Saved scaler from tm/{scaler_hash}.scaler")

    def load_scaler(self, scaler_hash: str) -> None:
        """Load the scaler using joblib.

        :param scaler_hash: Hash of the scaler.
        """
        logger.info(f"Loading scaler from tm/{scaler_hash}.scaler")
        self.scaler = joblib.load(f"tm/{scaler_hash}.scaler")
        logger.info(f"Loaded scaler from tm/{scaler_hash}.scaler")

    def transform(self, X: da.Array) -> da.Array:
        """Transform the data.

        :param X: Data to transform. Shape should be (N, C, H, W)
        :return: Transformed data
        """
        logger.info("Transforming the data using the scaler")
        # Reshape the data to 2D
        # Flatten and rechunk all the data so all pixels per channels are a single row
        # The shape is (C, N*H*W) after reshaping
        X_reshaped = X.transpose([0, 2, 3, 1]).reshape([-1, X.shape[1]])
        X_reshaped = X_reshaped.rechunk({1: X_reshaped.shape[1]})
        # Apply the scaler
        X_reshaped = self.scaler.transform(X_reshaped)
        X = X_reshaped.reshape(X.shape[0], X.shape[2], X.shape[3], X.shape[1]).transpose([0, 3, 1, 2])
        logger.info("Transformed the data using the scaler")
        return X
