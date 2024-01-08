"""Common functions used at the start of the main scripts train.py, cv.py, and submit.py."""
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import cast

import dask.array
from dask_image.imread import imread
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr

import wandb
from src.logging_utils.logger import logger


def setup_config(cfg: DictConfig) -> None:
    """Verify that config has no missing values and log it to yaml.

    :param cfg: The config object. Created with Hydra or OmegaConf.
    """
    # Check for missing keys in the config file
    missing = OmegaConf.missing_keys(cfg)
    if missing:
        raise ValueError(f"Missing keys in config file\n{missing}")


def setup_pipeline(pipeline_cfg: DictConfig, output_dir: Path, is_train: bool | None) -> Pipeline:
    """Instantiate the pipeline and log it to HTML.

    :param pipeline_cfg: The model pipeline config. Root node should be a ModelPipeline
    :param output_dir: The directory to save the pipeline to.
    :param is_train: Whether the pipeline is for training or not.
    """
    logger.info("Instantiating the pipeline")
    pipeline_cfg["feature_pipeline"]["is_train"] = is_train
    model_pipeline = instantiate(pipeline_cfg)

    logger.debug(f"Pipeline: \n{model_pipeline}")

    logger.info("Saving pipeline to HTML")
    set_config(display="diagram")
    pipeline_html = estimator_html_repr(model_pipeline)
    with open(output_dir / "pipeline.html", "w", encoding="utf-8") as f:
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
    # Suppress logger messages while getting the indices to avoid clutter in the log file
    logger.info("Finding shape of processed data")
    logger.setLevel("ERROR")
    x_processed = feature_pipeline.fit_transform(X)
    logger.setLevel("INFO")
    logger.info(f"Processed data shape: {x_processed.shape}")

    return X, y, x_processed


def setup_wandb(cfg: DictConfig, script_name: str, output_dir: Path) -> None:
    """Initialize Weights & Biases and log the config and code.

    :param cfg: The config object. Created with Hydra or OmegaConf.
    :param script_name: The name of the script, e.g. train, cv, etc.
    :param output_dir: The directory to the Hydra outputs.
    """
    logger.info("Initializing Weights & Biases")
    wandb.init(
        project="detect-kelp",
        group=script_name,
        settings=wandb.Settings(start_method="thread", code_dir="."),
        dir=output_dir,
    )
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    if cfg.wandb.log_config:
        logger.info("Uploading config files to Weights & Biases")
        # Store the config as an artefact of W&B
        artifact = wandb.Artifact(script_name + "_config", type="config")
        config_path = output_dir / ".hydra/config.yaml"
        artifact.add_file(str(config_path), script_name + ".yaml")
        wandb.log_artifact(artifact)

    # Log code to W&B
    if cfg.wandb.save_code.enabled:
        logger.info("Uploading code files to Weights & Biases")

        if wandb.run is None:  # Can't be True after wandb.init, but this casts wandb.run to be non-None, which is necessary for MyPy
            return

        wandb.run.log_code(
            root=".",
            exclude_fn=cast(Callable[[str, str], bool], lambda abs_path, root: re.match(cfg.wandb.save_code.exclude, Path(abs_path).relative_to(root).as_posix()) is not None),
        )

    logger.info("Done initializing Weights & Biases")


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
    x_processed = feature_pipeline.transform(X)
    logger.setLevel("INFO")
    logger.info(f"Processed data shape: {x_processed.shape}")

    return X, x_processed, filenames
