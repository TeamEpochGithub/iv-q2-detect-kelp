"""Common functions used at the start of the main scripts train.py, cv.py, and submit.py."""
import os
import re
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any, cast

import dask.array
from dask_image.imread import imread
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn import set_config
from sklearn.utils import estimator_html_repr

import wandb
from src.logging_utils.logger import logger
from src.pipeline.ensemble.ensemble import EnsemblePipeline
from src.pipeline.model.model import ModelPipeline
from wandb.sdk.lib import RunDisabled


def setup_config(cfg: DictConfig) -> None:
    """Verify that config has no missing values and log it to yaml.

    :param cfg: The config object. Created with Hydra or OmegaConf.
    """
    # Check for missing keys in the config file
    missing = OmegaConf.missing_keys(cfg)

    # If both model and ensemble are specified, raise an error
    if cfg.get("model") and cfg.get("ensemble"):
        raise ValueError("Both model and ensemble specified in config.")

    # If neither model nor ensemble are specified, raise an error
    if not cfg.get("model") and not cfg.get("ensemble"):
        raise ValueError("Neither model nor ensemble specified in config.")

    # If model and ensemble are in missing raise an error
    if "model" in missing and "ensemble" in missing:
        raise ValueError("Both model and ensemble are missing from config.")

    # If any other keys except model and ensemble are missing, raise an error
    if len(missing) > 1:
        raise ValueError(f"Missing keys in config: {missing}")


def setup_pipeline(pipeline_cfg: DictConfig, output_dir: Path, is_train: bool | None) -> ModelPipeline | EnsemblePipeline:
    """Instantiate the pipeline and log it to HTML.

    :param pipeline_cfg: The model pipeline config. Root node should be a ModelPipeline
    :param output_dir: The directory to save the pipeline to.
    :param is_train: Whether the pipeline is for training or not.
    """
    logger.info("Instantiating the pipeline")

    test_size = pipeline_cfg.get("test_size", -1)

    if "model" in pipeline_cfg:
        model_cfg = pipeline_cfg.model

        # Add test size to the config
        model_cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)
        model_cfg_dict = update_model_cfg_test_size(model_cfg_dict, test_size, is_train=is_train)

        cfg = OmegaConf.create(model_cfg_dict)

    elif "ensemble" in pipeline_cfg:
        ensemble_cfg = pipeline_cfg.ensemble

        ensemble_cfg_dict = OmegaConf.to_container(ensemble_cfg, resolve=True)
        if isinstance(ensemble_cfg_dict, dict):
            for model in ensemble_cfg_dict.get("models", []):
                ensemble_cfg_dict[model] = update_model_cfg_test_size(ensemble_cfg_dict[model], test_size, is_train=is_train)

        cfg = OmegaConf.create(ensemble_cfg_dict)

    model_pipeline = instantiate(cfg)

    logger.debug(f"Pipeline: \n{model_pipeline}")

    logger.info("Saving pipeline to HTML")
    set_config(display="diagram")
    pipeline_html = estimator_html_repr(model_pipeline)
    with open(output_dir / "pipeline.html", "w", encoding="utf-8") as f:
        f.write(pipeline_html)

    return model_pipeline


def update_model_cfg_test_size(
    model_cfg_dict: dict[str | bytes | int | Enum | float | bool, Any] | list[Any] | str | None, test_size: int = -1, *, is_train: bool | None
) -> dict[str | bytes | int | Enum | float | bool, Any] | list[Any] | str | None:
    """Update the test size in the model config.

    :param cfg: The model config.
    :param test_size: The test size.

    :return: The updated model config.
    """
    if isinstance(model_cfg_dict, dict):
        for model_block in model_cfg_dict.get("model_loop_pipeline", {}).get("model_blocks_pipeline", {}).get("model_blocks", []):
            model_block["test_size"] = test_size
        for pretrain_block in model_cfg_dict.get("model_loop_pipeline", {}).get("pretrain_pipeline", {}).get("pretrain_steps", []):
            pretrain_block["test_size"] = test_size

        if not is_train:
            model_cfg_dict.get("feature_pipeline", {})["processed_path"] = "data/test"
    return model_cfg_dict


def setup_train_data(data_path: str, target_path: str) -> tuple[dask.array.Array, dask.array.Array]:
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

    return X, y


def setup_wandb(
    cfg: DictConfig, job_type: str, output_dir: Path, name: str | None = None, group: str | None = None
) -> tuple[wandb.sdk.wandb_run.Run | RunDisabled | None, DictConfig]:
    """Initialize Weights & Biases and log the config and code.

    :param cfg: The config object. Created with Hydra or OmegaConf.
    :param job_type: The type of job, e.g. Training, CV, etc.
    :param output_dir: The directory to the Hydra outputs.
    :param name: The name of the run.
    :param group: The namer of the group of the run.
    """
    config = OmegaConf.to_container(cfg, resolve=True)

    # Check if config is a dict
    if not isinstance(config, dict):
        raise TypeError("Config is not a dict")

    logger.debug("Initializing Weights & Biases")
    run = wandb.init(
        project="detect-kelp",
        name=name,
        group=group,
        job_type=job_type,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
        settings=wandb.Settings(start_method="thread", code_dir="."),
        dir=output_dir,
        reinit=True,
        resume=False,
    )

    if isinstance(run, wandb.sdk.lib.RunDisabled) or run is None:  # Can't be True after wandb.init, but this casts wandb.run to be non-None, which is necessary for MyPy
        raise RuntimeError("Failed to initialize Weights & Biases")

    if cfg.wandb.log_config:
        logger.debug("Uploading config files to Weights & Biases")
        curr_config = "conf/" + job_type + ".yaml"

        # Get the model file name
        model_name = OmegaConf.load(curr_config)["defaults"][2]["model"]  # type: ignore[index]

        # Store the config as an artefact of W&B
        artifact = wandb.Artifact(job_type + "_config", type="config")
        config_path = output_dir / ".hydra/config.yaml"
        artifact.add_file(str(config_path), "config.yaml")
        artifact.add_file(curr_config)
        artifact.add_file(f"conf/model/{model_name}.yaml")
        wandb.log_artifact(artifact)

    if cfg.wandb.log_code.enabled:
        logger.debug("Uploading code files to Weights & Biases")

        run.log_code(
            root=".",
            exclude_fn=cast(Callable[[str, str], bool], lambda abs_path, root: re.match(cfg.wandb.log_code.exclude, Path(abs_path).relative_to(root).as_posix()) is not None),
        )

    logger.info("Done initializing Weights & Biases")
    return run, cfg


def setup_test_data(data_path: str) -> tuple[dask.array.Array, list[str]]:
    """Lazily read the raw data with dask, and find the shape after processing the test data.

    :param data_path: Path to the raw data.

    :return: X, filenames
    """
    logger.info("Lazily reading the raw test data")
    X = imread(f"{data_path}/*.tif").transpose(0, 3, 1, 2)
    filenames = [file for file in os.listdir(data_path) if file.endswith(".tif")]
    logger.info(f"Raw test data shape: {X.shape}")

    return X, filenames
