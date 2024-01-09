"""Common functions used at the start of the main scripts train.py, cv.py, and submit.py."""
import os
import sys

import dask.array
from dask_image.imread import imread
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr

from src.logging_utils.logger import logger


def setup_config(cfg: DictConfig) -> None:
    """Verify that config has no missing values and log it to yaml.

    :param cfg: The config object. Created with Hydra or OmegaConf.
    """
    # Check for missing keys in the config file
    missing = OmegaConf.missing_keys(cfg)
    if missing:
        raise ValueError(f"Missing keys in config file\n{missing}")


def setup_pipeline(pipeline_cfg: DictConfig, log_dir: str, is_train: bool | None) -> Pipeline:
    """Instantiate the pipeline and log it to HTML.

    :param pipeline_cfg: The model pipeline config. Root node should be a ModelPipeline
    :param log_dir: The directory to save the pipeline to.
    :param is_train: Whether the pipeline is for training or not.
    """
    logger.info("Instantiating the pipeline")
    if not is_train:
        pipeline_cfg["feature_pipeline"]["processed_path"] = "data/test"
    model_pipeline = instantiate(pipeline_cfg)

    logger.debug(f"Pipeline: \n{model_pipeline}")

    logger.info("Saving pipeline to HTML")
    set_config(display="diagram")
    pipeline_html = estimator_html_repr(model_pipeline)
    with open(f"{log_dir}/pipeline.html", "w", encoding="utf-8") as f:
        f.write(pipeline_html)

    return model_pipeline


def setup_train_data(data_path: str, target_path: str, feature_pipeline: Pipeline) -> tuple[dask.array.Array, dask.array.Array, dask.array.Array]:
    """Lazily read the raw data with dask, and find the shape after processing.

    :param data_path: Path to the raw data.
    :param target_path: Path to the raw target.
    :param feature_pipeline: The feature pipeline.

    :return: X, y, x_processed
    """
    logger.info("Lazily reading the raw data")
    X = imread(f"{data_path}/*.tif").transpose(0, 3, 1, 2)
    y = imread(f"{target_path}/*.tif")
    logger.info(f"Raw data shape: {X.shape}")
    logger.info(f"Raw target shape: {y.shape}")

    # Lazily process the features to know the shape in advance
    # Suppress logger and print messages while getting the indices to avoid clutter in the log file
    logger.info("Finding shape of processed data")
    logger.setLevel("ERROR")
    with open(os.devnull, "w") as null_file:
        sys.stdout = null_file

    x_processed = feature_pipeline.fit_transform(X)

    sys.stdout = sys.__stdout__
    logger.setLevel("INFO")
    logger.info(f"Processed data shape: {x_processed.shape}")
    return X, y, x_processed


def setup_test_data(data_path: str, feature_pipeline: Pipeline) -> tuple[dask.array.Array, dask.array.Array, list[str]]:
    """Lazily read the raw data with dask, and find the shape after processing the test data.

    :param data_path: Path to the raw data.
    :param feature_pipeline: The feature pipeline.

    :return: X, x_processed, filenames
    """
    logger.info("Lazily reading the raw test data")
    X = imread(f"{data_path}/*.tif").transpose(0, 3, 1, 2)
    filenames = [file for file in os.listdir(data_path) if file.endswith(".tif")]
    logger.info(f"Raw test data shape: {X.shape}")

    # Lazily process the features to know the shape in advance
    # Suppress logger messages while getting the indices to avoid clutter in the log file
    logger.info("Finding shape of processed data")
    logger.setLevel("ERROR")
    with open(os.devnull, "w") as null_file:
        sys.stdout = null_file
    x_processed = feature_pipeline.transform(X)
    sys.stdout = sys.__stdout__
    logger.setLevel("INFO")
    logger.info(f"Processed data shape: {x_processed.shape}")

    return X, x_processed, filenames
