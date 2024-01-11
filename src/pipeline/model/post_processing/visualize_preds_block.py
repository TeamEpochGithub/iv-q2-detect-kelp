import sys
from dataclasses import dataclass
import time

import dask
import dask.array as da
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from src.utils.setup import setup_test_data

from src.logging_utils.logger import logger

if sys.version_info < (3, 11):  # Self was added in Python 3.11
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class VisualizationBlock(BaseEstimator, TransformerMixin):

    def fit(self, X: np.ndarray, y: da.Array, test_indices: list[int], raw_data_path: str) -> Self:
        # Since this step comes after te model loop, X is the predictions
        # In this case X will be the predictions of the model
        # y will be the actual values
        self.preds = X
        self.targets = y[test_indices].compute()
        _, filenames = setup_test_data(raw_data_path)
        f = open("results.csv", "w")
        for pred, filename in tqdm(zip(self.preds, filenames, strict=False)):
            files = filename.split("_")
            image_key = files[0]
            # calculate the sum of preds
            pred_sum = np.sum(pred)
            # calculate the sum of targets
            target_sum = np.sum(self.targets[image_key])
            # calculate the intersection (product)
            intersection = np.sum(pred * self.targets[image_key])
            # now write these reuslts to a csv file
            f.write(f"{image_key},{pred_sum},{target_sum},{intersection}\n")
        f.close()
        
        return self
    
    def transform(self, X: da.Array) -> da.Array:
        return X