"""Train.py is the main script for training the model and will take in the raw data and output a trained model."""
import copy
import os
import warnings
from contextlib import nullcontext
from pathlib import Path

import hydra
import numpy as np
from distributed import Client
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

import wandb
from src.config.train_config import TrainConfig
from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator
from src.utils.script.generate_params import generate_train_params
from src.utils.script.lock import Lock
from src.utils.seed_torch import set_torch_seed
from src.utils.setup import setup_config, setup_pipeline, setup_train_data, setup_wandb

warnings.filterwarnings("ignore", category=UserWarning)
# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"
# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_train", node=TrainConfig)
count = 0


@hydra.main(version_base=None, config_path="conf", config_name="train")
def run_train(cfg: DictConfig) -> None:
    """Train a model pipeline with a train-test split. Entry point for Hydra which loads the config file."""
    # Run the train config with a dask client, and optionally a lock
    optional_lock = Lock if not cfg.allow_multiple_instances else nullcontext
    with optional_lock(), Client() as client:
        logger.info(f"Client: {client}")
        run_train_cfg(cfg)


def run_train_cfg(cfg: DictConfig) -> None:  # TODO(Jeffrey): Use TrainConfig instead of DictConfig
    """Train a model pipeline with a train-test split."""
    print_section_separator("Q2 Detect Kelp States - Training")
    set_torch_seed()

    import coloredlogs

    coloredlogs.install()

    # Check for missing keys in the config file
    setup_config(cfg)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    if cfg.wandb.enabled:
        setup_wandb(cfg, "train", output_dir)

    # Preload the pipeline and save it to HTML
    print_section_separator("Setup pipeline")
    model_pipeline = setup_pipeline(cfg, output_dir, is_train=True)

    # Lazily read the raw data with dask, and find the shape after processing
    X, y = setup_train_data(cfg.raw_data_path, cfg.raw_target_path)
    indices = np.arange(X.shape[0])

    # Split indices into train and test
    if cfg.test_size == 0:
        train_indices, test_indices = list(indices), []
    else:
        train_indices, test_indices = train_test_split(indices, test_size=cfg.test_size, random_state=42)
    logger.info(f"Train/Test size: {len(train_indices)}/{len(test_indices)}")

    # Generate the parameters for training
    fit_params = generate_train_params(cfg, model_pipeline, train_indices=train_indices, test_indices=test_indices)

    # Fit the pipeline
    original_y = copy.deepcopy(y)
    if "model" in cfg:
        target_pipeline = model_pipeline.get_target_pipeline()

        if target_pipeline is not None:
            logger.info("Now fitting the target pipeline...")
            y = target_pipeline.fit_transform(y)

    print_section_separator("Fit_transform model pipeline")
    predictions = model_pipeline.fit_transform(X, y, **fit_params)

    if len(test_indices) > 0:
        print_section_separator("Scoring")
        scorer = instantiate(cfg.scorer)
        score = scorer(original_y[test_indices].compute(), predictions[test_indices])
        logger.info(f"Score: {score}")

        if wandb.run:
            wandb.log({"Score": score})

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    run_train()
