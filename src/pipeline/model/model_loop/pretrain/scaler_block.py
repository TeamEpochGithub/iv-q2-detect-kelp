from sklearn.base import BaseEstimator, TransformerMixin
from src.logging_utils.logger import logger

import numpy as np
import dask.array as da
import time

class ScalerBlock(BaseEstimator, TransformerMixin):

    def __init__(self, scaler: BaseEstimator) -> None:
        """Initialize the ScalerBlock.

        :param scaler: Scaler.
        """
        self.scaler = scaler

    def fit(self, X: da.Array, y: da.Array, train_indices: list[int]):
        """Fit the scaler.

        :param X: Data to fit
        :param y: Target data
        :return: Fitted scaler
        """
        logger.info("Fitting scaler")
        # save the original shape
        start_time = time.time()
        # reshape X so that it is 2D
        train_indices.sort()
        X_reshaped = X[train_indices].transpose([0, 2, 3, 1]).reshape([-1, X.shape[1]])
        X_reshaped = X_reshaped.rechunk({1: X_reshaped.shape[1]})
        # fit the scaler on the data
        self.scaler.fit(X_reshaped)
        logger.info(f"Fitted scaler in {time.time() - start_time} seconds")
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data.
        
        :param X: Data to transform
        :return: Transformed data
        """

        logger.info("Transforming scaler")
        start_time = time.time()
        # reshape the data to 2D
        X_reshaped = X.transpose([0, 2, 3, 1]).reshape([-1, X.shape[1]])
        X_reshaped = X_reshaped.rechunk({1: X_reshaped.shape[1]})
        # apply the scaler
        X_reshaped = self.scaler.transform(X_reshaped)
        X = X_reshaped.reshape(X.shape[0], X.shape[2], X.shape[3], X.shape[1]).transpose([0, 3, 1, 2])
        logger.info(f"Transformed scaler in {time.time() - start_time} seconds")
        return X