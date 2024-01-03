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
        self.original_shape = X.shape
        start_time = time.time()
        # here flatten the data so that you have a 2d array with channels and flattened images (all images in one row) and fit the scaler on this and reshape it back
        X_flat = []
        # y has 1 channel so its 3d
        y_flat = y.flatten().reshape(1, -1)
        for channel in range(X.shape[1]):
            X_flat.append(X[:, channel, :, :].flatten())
        X = da.array(X_flat)
        y = y_flat
        X = X.rechunk({1: X.shape[1]})
        y = y.rechunk({1: y.shape[1]})
        print(X.shape)
        print(y.shape)
        self.scaler.fit(X, y)
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
        X = X.reshape([X.shape[1], -1])
        X = self.scaler.transform(X)
        X = X.reshape(self.original_shape)
        logger.info(f"Transformed scaler in {time.time() - start_time} seconds")
        return X