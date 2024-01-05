"""Train.py is the main script for training the model and will take in the raw data and output a trained model."""
import re
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import omegaconf
from dask_image.imread import imread
from distributed import Client
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn import set_config
from sklearn.base import estimator_html_repr
from sklearn.model_selection import train_test_split

import wandb
from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator
from src.utils.flatten_dict import flatten_dict

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class WandBLogCodeConfig:
    """Schema for the code logging to Weights & Biases.

    :param enabled: Whether to log the code to Weights & Biases.
    :param exclude: Regex of files to exclude from logging.
    """

    enabled: bool
    exclude: str


@dataclass
class WandBConfig:
    """Schema for the Weights & Biases config yaml file.

    :param enabled: Whether to log to Weights & Biases.
    :param log_code: Whether to log the code to Weights & Biases.
    """

    enabled: bool
    log_config: bool
    log_code: WandBLogCodeConfig


@dataclass
class TrainConfig:
    """Schema for the train config yaml file.

    :param model: Model pipeline.
    :param test_size: Test size.
    :param raw_data_path: Path to the raw data.
    :param raw_target_path: Path to the raw target.
    :param wandb: Whether to log to Weights & Biases and other settings.
    """

    model: Any
    test_size: float
    raw_data_path: str
    raw_target_path: str
    wandb: WandBConfig


# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_train", node=TrainConfig)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def run_train(cfg: TrainConfig) -> None:
    """Train a model pipeline with a train-test split."""
    # Check for missing keys in the config file
    missing = OmegaConf.missing_keys(cfg)
    if missing:
        raise ValueError(f"Missing keys in config file\n{missing}")

    import coloredlogs

    coloredlogs.install()

    outputs_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    if cfg.wandb.enabled:
        # Initialize W&B
        wandb.init(
            project="detect-kelp",
            group="train",
            settings=wandb.Settings(start_method="thread", code_dir="."),
            dir=outputs_dir,
        )
        wandb.config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        if wandb.run is None:  # Can't happen after wandb.init, but this casts wandb.run to be non-None, which is necessary for MyPy
            return

        if cfg.wandb.log_config:
            # Store the config as an artefact of W&B
            artifact = wandb.Artifact("train_config", type="config")
            config_path = outputs_dir / ".hydra/config.yaml"
            artifact.add_file(str(config_path), "train.yaml")
            wandb.log_artifact(artifact)

        # Log code to W&B
        if cfg.wandb.log_code.enabled:
            logger.info("Uploading code files to Weights & Biases")

            wandb.run.log_code(
                root=".",
                exclude_fn=cast(
                    Callable[[str, str], bool], lambda abs_path, root: re.match(cfg.wandb.log_code.exclude, Path(abs_path).relative_to(root).as_posix()) is not None
                ),
            )

    # Print section separator
    print_section_separator("Q2 Detect Kelp States -- Training")

    # Set up the pipeline
    logger.info("Setting up the pipeline")
    orig_time = time.time()
    model_pipeline = instantiate(cfg.model.pipeline).get_pipeline()
    logger.info(f"Pipeline setup time: {time.time() - orig_time} seconds")
    logger.debug(f"Pipeline: {model_pipeline}")

    # Save the pipeline to an HTML file, next to the log file in the hydra output
    set_config(display="diagram")
    pipeline_html = estimator_html_repr(model_pipeline)

    with open(outputs_dir / "pipeline.html", "w", encoding="utf-8") as f:
        f.write(pipeline_html)

    # Read in the raw data
    logger.info("Reading in the raw feature and target data")
    X = imread(f"{cfg.raw_data_path}/*.tif").transpose(0, 3, 1, 2)
    y = imread(f"{cfg.raw_target_path}/*.tif")
    logger.info(f"Raw data shape: {X.shape}")
    logger.info(f"Raw target shape: {y.shape}")

    # Lazily process the features to know the shape in advance
    # Suppress logger messages while getting the indices to avoid clutter in the log file
    logger.info("Finding shape of processed data")
    logger.setLevel("ERROR")
    feature_pipeline = model_pipeline.named_steps.feature_pipeline
    x_processed = feature_pipeline.fit_transform(X)
    logger.setLevel("INFO")
    logger.info(f"Processed data shape: {x_processed.shape}")
    indices = np.arange(x_processed.shape[0])

    # Split indices into train and test
    train_indices, test_indices = train_test_split(indices, test_size=cfg.test_size)
    logger.info("Splitting the data into train and test sets")
    logger.debug(f"Train indices: {train_indices}")
    logger.debug(f"Test indices: {test_indices}")

    # Set train and test indices for each model block
    # Due to how SKLearn pipelines work, we have to set the model fit parameters using a deeply nested dictionary
    # Then we convert it to a flat dictionary with __ as the separator between each level
    fit_params = {
        "model_loop_pipeline": {
            "model_blocks_pipeline": {
                name: {"train_indices": train_indices, "test_indices": test_indices, "cache_size": -1}
                for name, _ in model_pipeline.named_steps.model_loop_pipeline.named_steps.model_blocks_pipeline.steps
            }
        }
    }
    fit_params_flat = flatten_dict(fit_params)

    # Fit the pipeline
    model_pipeline.fit(X, y, **fit_params_flat)

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    # Run with dask client, which will automatically close if there is an error
    with Client() as client:
        logger.info(f"Client: {client}")
        run_train()
