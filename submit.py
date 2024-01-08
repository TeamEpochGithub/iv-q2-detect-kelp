"""Submit.py is the main script for running inference on the test set and creating a submission."""
import glob
import warnings
from dataclasses import dataclass
from typing import Any

import hydra
from distributed import Client
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator
from src.utils.hashing import hash_models, hash_scalers
from src.utils.make_submission import make_submission
from src.utils.setup import setup_config, setup_ensemble, setup_pipeline, setup_test_data

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class SubmitConfig:
    """Schema for the train config yaml file."""

    model: Any
    ensemble: Any
    test_size: float
    raw_data_path: str = "data/raw/test_satellite"
    raw_target_path: str = "data/raw/train_kelp"


# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_submit", node=SubmitConfig)


@hydra.main(version_base=None, config_path="conf", config_name="submit")
def run_submit(cfg: DictConfig) -> None:
    """Run the main script for submitting the predictions."""
    # Print section separator
    print_section_separator("Q2 Detect Kelp States -- Submit")
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Coloured logs
    import coloredlogs

    coloredlogs.install()

    # Check for missing keys in the config file
    setup_config(cfg)

    # Hash representation of model pipeline only based on model and test size
    model_hashes = hash_models(cfg)

    # Hash representation of scaler based on pretrain, feature_pipeline and test_size
    scaler_hashes = hash_scalers(cfg)

    # Check if models are cached already, if not give an error
    for model_hash in model_hashes:
        if not glob.glob(f"tm/{model_hash}.pt"):
            logger.error(f"Model {model_hash} not found. Please train the model first and ensure the test_size is also the same.")
            raise FileNotFoundError(f"Model {model_hash} not found. Please train the model first.")

    # Check if scalers are cached already, if not give an error
    for scaler_hash in scaler_hashes:
        if scaler_hash is not None and not glob.glob(f"tm/{scaler_hash}.scaler"):
            logger.error(f"Scaler {scaler_hash} not found. Please train the model first.")
            raise FileNotFoundError(f"Scaler {scaler_hash} not found. Please train the model first.")

    # Preload the pipeline and save it to HTML
    if "model" in cfg:
        model_pipeline = setup_pipeline(cfg.model, log_dir, is_train=False)
    elif "ensemble" in cfg:
        model_pipeline = setup_ensemble(cfg.ensemble, log_dir, is_train=False)
    else:
        raise ValueError("Either model or ensemble must be specified in the config file.")

    # Load the test data
    if "model" in cfg:
        feature_pipeline = model_pipeline.named_steps.feature_pipeline_step
    elif "ensemble" in cfg:
        # Take first feature pipeline from ensemble TODO
        feature_pipeline = model_pipeline.models["0"].named_steps.feature_pipeline_step
    X, _, filenames = setup_test_data(cfg.raw_data_path, feature_pipeline)

    # Load the model from the model hashes
    for i, model_hash in enumerate(model_hashes):
        next(iter(model_pipeline.models[str(i)].named_steps.model_loop_pipeline_step.named_steps.model_blocks_pipeline_step.named_steps.values())).load_model(model_hash)

    # Load the scalers from the scaler hashes
    for i, scaler_hash in enumerate(scaler_hashes):
        if scaler_hash is not None:
            model_pipeline.models[str(i)].named_steps.model_loop_pipeline_step.named_steps.pretrain_pipeline_step.load_scaler(scaler_hash)

    # Predict on the test data
    predictions = model_pipeline.transform(X)

    # Make submission
    make_submission(log_dir, predictions, filenames, threshold=0.5)


if __name__ == "__main__":
    # Run with dask client, which will automatically close if there is an error
    with Client() as client:
        logger.info(f"Client: {client}")
        run_submit()
