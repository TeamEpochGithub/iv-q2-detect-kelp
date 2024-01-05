"""Train.py is the main script for training the model and will take in the raw data and output a trained model."""
import glob
import logging
import warnings
from dataclasses import dataclass
from typing import Any

import hydra
import numpy as np
from distributed import Client
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator
from src.utils.flatten_dict import flatten_dict
from src.utils.hashing import hash_model, hash_scaler
from src.utils.setup import setup_config, setup_pipeline, setup_train_data

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class TrainConfig:
    """Schema for the train config yaml file."""

    model: Any
    test_size: float
    raw_data_path: str = "data/raw/train_satellite"
    raw_target_path: str = "data/raw/train_kelp"


# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_train", node=TrainConfig)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def run_train(cfg: DictConfig) -> None:
    """Train a model pipeline with a train-test split."""
    print_section_separator("Q2 Detect Kelp States -- Training")
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Coloured logs
    import coloredlogs

    coloredlogs.install()

    # Check for missing keys in the config file
    setup_config(cfg)

    # Hash representation of model pipeline only based on model and test size
    model_hash = hash_model(cfg)

    # Hash representation of scaler based on pretrain, feature_pipeline and test_size
    scaler_hash = hash_scaler(cfg)

    # Check if model is cached already, if so do not call fit.
    # This is done to avoid retraining the model if it is already cached.
    if glob.glob(f"tm/{model_hash}.pt"):
        logger.info(f"Trained model already cached at tm/{model_hash}.pt, skipping training")
        return
    if scaler_hash is None:
        logging.warning("No scaler found in config, training without scaler")

    # Preload the pipeline and save it to HTML
    model_pipeline = setup_pipeline(cfg.model.pipeline, log_dir, is_train=True)

    # Lazily read the raw data with dask, and find the shape after processing
    feature_pipeline = model_pipeline.named_steps.feature_pipeline_step
    X, y, x_processed = setup_train_data(cfg.raw_data_path, cfg.raw_target_path, feature_pipeline)
    indices = np.arange(x_processed.shape[0])

    # Split indices into train and test
    if cfg.test_size == 0:
        train_indices, test_indices = indices, []
    else:
        train_indices, test_indices = train_test_split(indices, test_size=cfg.test_size)
    logger.info(f"Train/Test size: {len(train_indices)}/{len(test_indices)}")

    # Set train and test indices for each model block
    # Due to how SKLearn pipelines work, we have to set the model fit parameters using a deeply nested dictionary
    # Then we convert it to a flat dictionary with __ as the separator between each level
    fit_params = {
        "model_loop_pipeline_step": {
            "model_blocks_pipeline_step": {
                name: {"train_indices": train_indices, "test_indices": test_indices, "cache_size": -1}
                for name, _ in model_pipeline.named_steps.model_loop_pipeline_step.named_steps.model_blocks_pipeline_step.steps
            },
        }
    }

    # Add pretrain indices if it exists. Stupid mypy doesn't understand hasattr
    if hasattr(model_pipeline.named_steps.model_loop_pipeline_step.named_steps, "pretrain_pipeline_step"):
        fit_params["model_loop_pipeline_step"]["pretrain_pipeline_step"] = {"train_indices": train_indices}  # type: ignore[dict-item]

    fit_params_flat = flatten_dict(fit_params)

    # Fit the pipeline
    model_pipeline.fit(X, y, **fit_params_flat)
    # Get the model and scaler
    model = next(iter(model_pipeline.named_steps.model_loop_pipeline_step.named_steps.model_blocks_pipeline_step.named_steps.values()))
    # Save the model
    model.save_model(model_hash)

    # Get the ScalerBlock if it exists
    if scaler_hash is not None:
        scaler = model_pipeline.named_steps.model_loop_pipeline_step.named_steps.pretrain_pipeline_step
        scaler.save_scaler(scaler_hash)


if __name__ == "__main__":
    # Run with dask client, which will automatically close if there is an error
    with Client() as client:
        logger.info(f"Client: {client}")
        run_train()
