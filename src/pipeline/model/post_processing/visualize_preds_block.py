"""VisualizationBlock is the class used to create the visualization block."""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import hydra
import numpy as np
import numpy.typing as npt
import tifffile
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_utils.logger import logger
from src.scoring.dice_coefficient import DiceCoefficient
from src.utils.setup import setup_test_data

if sys.version_info < (3, 11):  # Self was added in Python 3.11
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class VisualizationBlock(BaseEstimator, TransformerMixin):
    """VisualizationBlock is the class used to create the visualization block.

    :param raw_data_path: The path to the raw data
    """

    raw_data_path: str
    save_train: bool = False

    def fit(self, X: npt.NDArray[np.float64 | np.float32 | np.int32], y: da.Array, *, test_indices: list[int] | None = None) -> Self:
        """Store the predicted images and their corresponding scores to the output folder.

        :param X: The predicted images
        :param y: The actual images
        :param test_indices: The indices of the test images

        :return: self
        """
        # Since this step comes after te model loop, X is the predictions
        # In this case X will be the predictions of the model
        # y will be the label values
        if test_indices is None:
            test_indices = []
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        test_indices.sort()
        _, filenames = setup_test_data(self.raw_data_path)

        if self.save_train:
            self.preds = X
            self.targets = y.compute()
            filenames = list(np.array(filenames))
        else:
            self.preds = X[test_indices]
            self.targets = y[test_indices].compute()
            filenames = list(np.array(filenames)[test_indices])

        preds_loc = f"{output_dir}/preds"
        if not os.path.exists(preds_loc):
            os.makedirs(preds_loc)

        dice = DiceCoefficient()
        dice_coefs, intersections, sum_targets, sum_preds = [], [], [], []

        logger.info("Calculating dice coefficients...")
        for i, (pred, filename) in enumerate(zip(self.preds, filenames, strict=False)):
            files = filename.split("_")
            image_key = files[0]
            # Calculate the intemrediates for later use
            intersection = np.sum(self.targets[i] * pred)
            sum_target = np.sum(self.targets[i])
            sum_pred = np.sum(pred)
            # Calculate the dice coefficient
            dice_coef = dice(self.targets[i], pred)
            # Store the results in to lists
            dice_coefs.append(dice_coef)
            intersections.append(intersection)
            sum_targets.append(sum_target)
            sum_preds.append(sum_pred)
            # Also write the predicted mask as tifffiles without thresholding
            tifffile.imwrite(f"{preds_loc}/{image_key}_pred.tif", pred)
        logger.info("Done calculating dice coefficients. Storing the results")
        # Sort the lists by dice_coef
        idxs = np.argsort(dice_coefs)
        dice_coefs = list(np.array(dice_coefs)[idxs])
        intersections = list(np.array(intersections)[idxs])
        sum_targets = list(np.array(sum_targets)[idxs])
        sum_preds = list(np.array(sum_preds)[idxs])
        filenames = list(np.array(filenames)[idxs])
        if self.save_train:
            # Create a column called in_val to indicate whether the image is in the validation set (1) or not (0)
            in_val = np.zeros(len(filenames))
            in_val[test_indices] = 1
        else:
            in_val = np.ones(len(filenames))

        # Write the results to a csv file
        with open(f"{output_dir}/results.csv", "w") as f:
            f.write("image_key,in_val,sum_targets,sum_preds,intersections,dice_coef,\n")
            for i, (filename, dice_coef) in enumerate(zip(filenames, dice_coefs, strict=False)):
                files = filename.split("_")
                image_key = files[0]
                f.write(f"{image_key}, {in_val[i]}, {sum_targets[i]}, {sum_preds[i]}, {intersections[i]}, {dice_coef}\n")

            f.close()
        logger.info("Done storing the results")
        return self

    def transform(self, X: da.Array) -> da.Array:
        """Return the predictions.

        :param X: The predictions
        :return: The predictions
        """
        return X
