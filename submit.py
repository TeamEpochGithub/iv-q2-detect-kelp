"""Submit.py is the main script for running inference on the test set and creating a submission."""
import os
import warnings
from pathlib import Path

import hydra
from distributed import Client
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from src.config.submit_config import SubmitConfig
from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator
from src.utils.make_submission import make_submission
from src.utils.setup import setup_config, setup_pipeline, setup_test_data

warnings.filterwarnings("ignore", category=UserWarning)

# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"

# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_submit", node=SubmitConfig)


@hydra.main(version_base=None, config_path="conf", config_name="submit")
# TODO(Jeffrey): Use SubmitConfig instead of DictConfig
def run_submit(cfg: DictConfig) -> None:
    """Run the main script for submitting the predictions."""
    # Print section separator
    print_section_separator("Q2 Detect Kelp States - Submit")
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Set up logging
    import coloredlogs

    coloredlogs.install()

    # Check for missing keys in the config file
    setup_config(cfg)

    # Preload the pipeline and save it to HTML
    print_section_separator("Setup pipeline")
    model_pipeline = setup_pipeline(cfg, output_dir, is_train=False)

    # Load the test data
    X, filenames = setup_test_data(cfg.raw_data_path)

    # Predict on the test data
    logger.info("Now transforming the pipeline...")
    predictions = model_pipeline.transform(X)

    # Make submission
    if predictions is not None:
        make_submission(output_dir, predictions, filenames)
    else:
        raise ValueError("Predictions are None")


if __name__ == "__main__":
    # Run with dask client, which will automatically close if there is an error
    with Client() as client:
        logger.info(f"Client: {client}")
        run_submit()
