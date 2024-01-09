"""Create a submission file of the predictions (store single-band TIF files with predictions of each image in test_predictions)."""
import os
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from tqdm import tqdm

from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator


def make_submission(path: str, predictions: np.ndarray[Any, Any], filenames: list[str], threshold: float = 0.5) -> None:
    """Create a submission file of the predictions (store single-band TIF files with predictions of each image in test_predictions).

    :param path: Path to the submission folder
    :param predictions: Predictions of the model
    :param filenames: Filenames of the test data
    :param threshold: Threshold to use for the predictions
    """
    print_section_separator("Making submission")
    logger.info("Creating submission.zip")
    # Create a submission file of the predictions (store single-band TIF files with predictions of each image in test_predictions)
    loc = path + "/test_predictions/"
    # Create the dir
    if not os.path.exists(loc):
        os.makedirs(loc)
    # Remove all files from zip
    for file in os.listdir(loc):
        os.remove(os.path.join(loc, file))

    logger.info("Setting thresholds...")
    for pred, filename in tqdm(zip(predictions, filenames, strict=False)):
        files = filename.split("_")
        final_name = files[0] + "_kelp.tif"
        # Set replace values above the threshold to be 1 and below threshold to be 0 of pred
        pred[pred > threshold] = 1
        pred[pred <= threshold] = 0
        # Save the prediction as tiff file
        # Threshold the
        tifffile.imwrite(os.path.join(loc, final_name), pred)

    # Create a zip file of all files in loc
    import zipfile

    logger.info(f"All predictions saved in {loc}.")
    logger.info("Creating submission.zip")
    with zipfile.ZipFile("submission.zip", "w") as zipped:
        files_zip = Path(loc).rglob("*.tif")  # get all files.
        for file_zip in files_zip:
            zipped.write(file_zip, file_zip.name)

    logger.info("Submission.zip created! Now let's get that new top score! :)")
