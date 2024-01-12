"""Train.py is the main script for training the model and will take in the raw data and output a trained model."""
import os
import warnings
from pathlib import Path

import hydra
import numpy as np
import wandb
from distributed import Client
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.config.train_config import TrainConfig
from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator
from src.utils.seed_torch import set_torch_seed
from src.utils.setup import setup_config, setup_pipeline, setup_train_data, setup_wandb

warnings.filterwarnings("ignore", category=UserWarning)
# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"

# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_train", node=TrainConfig)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def run_train(cfg: DictConfig) -> None:  # TODO(Jeffrey): Use TrainConfig instead of DictConfig
    """Train a model pipeline with a train-test split."""
    print_section_separator("Q2 Detect Kelp States - Training")
    set_torch_seed()

    import coloredlogs

    coloredlogs.install()

    # Check for missing keys in the config file
    setup_config(cfg)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    if cfg.wandb.enabled:
        setup_wandb(cfg, "Training", output_dir)

    # Preload the pipeline and save it to HTML
    print_section_separator("Setup pipeline")
    model_pipeline = setup_pipeline(cfg, output_dir, is_train=True)

    # Lazily read the raw data with dask, and find the shape after processing
    X, y = setup_train_data(cfg.raw_data_path, cfg.raw_target_path)
    indices = np.arange(X.shape[0])

    # Split indices into train and test
    if cfg.test_size == 0:
        train_indices, test_indices = indices, []
    else:
        train_indices, test_indices = train_test_split(indices, test_size=cfg.test_size)
    logger.info(f"Train/Test size: {len(train_indices)}/{len(test_indices)}")
    logger.info("Now fitting the pipeline...")
    # Set train and test indices for each model block
    # Due to how SKLearn pipelines work, we have to set the model fit parameters using a deeply nested dictionary
    # Then we convert it to a flat dictionary with __ as the separator between each level

    fit_params = {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "cache_size": cfg.cache_size,
    }

    # Fit the pipeline
    model_pipeline.fit(X, y, **fit_params)

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    # Run with dask client, which will automatically close if there is an error
    with Client() as client:
        logger.info(f"Client: {client}")
        run_train()
