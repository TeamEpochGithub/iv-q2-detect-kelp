import sys
from dataclasses import dataclass
import time

import dask
import dask.array as da
import joblib
import numpy as np
import hydra
import tifffile
import os

from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from src.utils.setup import setup_test_data
from pathlib import Path
from src.logging_utils.logger import logger

if sys.version_info < (3, 11):  # Self was added in Python 3.11
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class VisualizationBlock(BaseEstimator, TransformerMixin):

    raw_data_path: str

    def fit(self, X: np.ndarray, y: da.Array, *, test_indices: list[int] = []) -> Self:
        # Since this step comes after te model loop, X is the predictions
        # In this case X will be the predictions of the model
        # y will be the actual values
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.preds = X[test_indices]
        self.targets = y[test_indices].compute()
        _, filenames = setup_test_data(self.raw_data_path)
        f = open(f"{output_dir}/results.csv", "w")
        f.write("image_key,pred_sum,target_sum,intersection\n")
        preds_loc = f"{output_dir}/preds"
        if not os.path.exists(preds_loc):
            os.makedirs(preds_loc)

        for i, (pred, filename) in enumerate(tqdm(zip(self.preds, filenames, strict=False))):
            files = filename.split("_")
            image_key = files[0]
            # calculate the sum of preds
            pred_sum = np.sum(pred)
            # calculate the sum of targets
            target_sum = np.sum(self.targets[i])
            # calculate the intersection (product)
            intersection = np.sum(pred * self.targets[i])
            dice = (2 * intersection) / (pred_sum + target_sum)
            # now write these reuslts to a csv file
            f.write(f"{image_key},{pred_sum},{target_sum},{intersection},{dice}\n")

            # also write the predicted mask as tifffiles without thresholding
            tifffile.imwrite(f"{preds_loc}/{image_key}_pred.tif", pred)


        f.close()
        
        return self
    
    def transform(self, X: da.Array) -> da.Array:
        return X