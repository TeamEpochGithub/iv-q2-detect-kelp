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
from src.utils.script.hash_check import check_hash
from src.utils.setup import setup_config, setup_ensemble, setup_pipeline, setup_test_data

warnings.filterwarnings("ignore", category=UserWarning)

# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"

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

    # Check if the model and scaler hashes are cached already, if not give an error
    model_hashes, scaler_hashes = check_hash(cfg)

    # Preload the pipeline and save it to HTML
    if "model" in cfg:
        model_pipeline = setup_pipeline(cfg.model, output_dir, is_train=False)
    elif "ensemble" in cfg:
        model_pipeline = setup_ensemble(cfg.ensemble, output_dir, is_train=False)

    # Load the test data
    if "model" in cfg:
        feature_pipeline = model_pipeline.named_steps.feature_pipeline_step
    elif "ensemble" in cfg:
        # Take first feature pipeline from ensemble TODO
        model1 = next(iter(model_pipeline.models.values()))
        feature_pipeline = model1.named_steps.feature_pipeline_step
    X, _, filenames = setup_test_data(cfg.raw_data_path, feature_pipeline)

    # Load the model from the model hashes

    model_keys = list(model_pipeline.models.keys())
    for i, model_hash in enumerate(model_hashes):
        next(iter(model_pipeline.models[model_keys[i]].named_steps.model_loop_pipeline_step.named_steps.model_blocks_pipeline_step.named_steps.values())).load_model(
            model_hash
        )

    # Load the scalers from the scaler hashes
    for i, scaler_hash in enumerate(scaler_hashes):
        if scaler_hash is not None:
            model_pipeline.models[model_keys[i]].named_steps.model_loop_pipeline_step.named_steps.pretrain_pipeline_step.load_scaler(scaler_hash)

    # Predict on the test data
    predictions = model_pipeline.transform(X)

    # Make submission
    make_submission(output_dir, predictions, filenames, threshold=0.25)


if __name__ == "__main__":
    # Run with dask client, which will automatically close if there is an error
    with Client() as client:
        logger.info(f"Client: {client}")
        run_submit()
