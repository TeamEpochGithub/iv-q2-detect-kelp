"""Submit.py is the main script for running inference on the test set and creating a submission."""
import glob
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
from distributed import Client
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator
from src.utils.hashing import hash_model, hash_scaler
from src.utils.make_submission import make_submission
from src.utils.setup import setup_config, setup_pipeline, setup_test_data

warnings.filterwarnings("ignore", category=UserWarning)
# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"


@dataclass
class SubmitConfig:
    """Schema for the train config yaml file."""

    model: Any
    test_size: float
    raw_data_path: str = "data/raw/train_satellite"
    raw_target_path: str = "data/raw/train_kelp"


# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_submit", node=SubmitConfig)


@hydra.main(version_base=None, config_path="conf", config_name="submit")
def run_submit(cfg: DictConfig) -> None:  # TODO(Jeffrey): Use SubmitConfig instead of DictConfig
    """Run the main script for submitting the predictions."""
    # Print section separator
    print_section_separator("Q2 Detect Kelp States -- Submit")
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    import coloredlogs

    coloredlogs.install()

    # Check for missing keys in the config file
    setup_config(cfg)

    # Hash representation of model pipeline only based on model and test size
    model_hash = hash_model(cfg)

    # Hash representation of scaler based on pretrain, feature_pipeline and test_size
    scaler_hash = hash_scaler(cfg)

    # Check if model is cached already, if not give an error
    if not glob.glob(f"tm/{model_hash}.pt"):
        logger.error(f"Model {model_hash} not found. Please train the model first and ensure the test_size is also the same.")
        raise FileNotFoundError(f"Model {model_hash} not found. Please train the model first.")

    if scaler_hash is not None and not glob.glob(f"tm/{scaler_hash}.scaler"):
        # Check if scaler is cached already, if not give an error
        logger.error(f"Scaler {scaler_hash} not found. Please train the model first.")
        raise FileNotFoundError(f"Scaler {scaler_hash} not found. Please train the model first.")

    # Preload the pipeline and save it to HTML
    model_pipeline = setup_pipeline(cfg.model.pipeline, output_dir, is_train=False)

    # Load the test data
    feature_pipeline = model_pipeline.named_steps.feature_pipeline_step
    X, _, filenames = setup_test_data(cfg.raw_data_path, feature_pipeline)

    # Load the model from the model hash
    next(iter(model_pipeline.named_steps.model_loop_pipeline_step.named_steps.model_blocks_pipeline_step.named_steps.values())).load_model(model_hash)

    # Load the scaler from the scaler hash
    if scaler_hash is not None:
        model_pipeline.named_steps.model_loop_pipeline_step.named_steps.pretrain_pipeline_step.load_scaler(scaler_hash)

    # Predict on the test data
    predictions = model_pipeline.transform(X)

    # Make submission
    make_submission(output_dir, predictions, filenames, threshold=0.5)


if __name__ == "__main__":
    # Run with dask client, which will automatically close if there is an error
    with Client() as client:
        logger.info(f"Client: {client}")
        run_submit()
