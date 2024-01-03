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

    def fit(self, X: da.Array, y: da.Array = None):
        """Fit the scaler.

        :param X: Data to fit
        :param y: Target data
        :return: Fitted scaler
        """
        logger.info("Fitting scaler")
        # save the original shape
        start_time = time.time()
        # find the mean of the array for each channel as if it were flattened
        # so I should get 1 mean for each channel
        # Reshape the array to combine all dimensions except the channel dimension
        X_reshaped = X.transpose([0, 2, 3, 1]).reshape([-1, X.shape[1]])

        # Compute the mean and standard deviation for each channel
        print(X_reshaped.shape)
        self.mean_per_channel = X_reshaped.mean(axis=0)
        print(self.mean_per_channel)
        self.std_per_channel = X_reshaped.std(axis=0)
        print(self.std_per_channel)

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
        X_reshaped = X.transpose([0, 2, 3, 1]).reshape([-1, X.shape[1]])
        X_reshaped = (X_reshaped - self.mean_per_channel) / self.std_per_channel
        X = X_reshaped.reshape(X.shape[0], X.shape[2], X.shape[3], X.shape[1]).transpose([0, 3, 1, 2])
        logger.info(f"Transformed scaler in {time.time() - start_time} seconds")
        return X