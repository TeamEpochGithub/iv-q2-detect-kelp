"""cv.py is the main script for doing cv and will take in the raw data, do cv and log the cv results."""
import warnings
from dataclasses import dataclass
from typing import Any

import hydra
from distributed import Client
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold

from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator
from src.utils.flatten_dict import flatten_dict
from src.utils.setup import setup_config, setup_pipeline, setup_train_data

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class CVConfig:
    """Schema for the cv config yaml file."""

    model: Any
    n_splits: int
    raw_data_path: str = "data/raw/train_satellite"
    raw_target_path: str = "data/raw/train_kelp"


# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_cv", node=CVConfig)


@hydra.main(version_base=None, config_path="conf", config_name="cv")
def run_cv(cfg: DictConfig) -> None:
    """Do cv on a model pipeline with K fold split."""
    print_section_separator("Q2 Detect Kelp States -- CV")
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Coloured logs
    import coloredlogs

    coloredlogs.install()

    # Check for missing keys in the config file
    setup_config(cfg)

    # Preload the pipeline and save it to HTML
    model_pipeline = setup_pipeline(cfg.model.pipeline, log_dir, is_train=True)

    # Lazily read the raw data with dask, and find the shape after processing
    feature_pipeline = model_pipeline.named_steps.feature_pipeline_step
    X, y, x_processed = setup_train_data(cfg.raw_data_path, cfg.raw_target_path, feature_pipeline)

    # Perform stratified k-fold cross validation, where the group of each image is determined by having kelp or not.
    kf = StratifiedKFold(n_splits=cfg.n_splits)
    stratification_key = y.compute().reshape(y.shape[0], -1).max(axis=1)
    for i, (train_indices, test_indices) in enumerate(kf.split(x_processed, stratification_key)):
        print_section_separator(f"CV - Fold {i}")
        logger.info(f"Train/Test size: {len(train_indices)}/{len(test_indices)}")

        logger.info("Creating clean pipeline for this fold")
        model_pipeline = instantiate(cfg.model.pipeline)

        # Set train and test indices for each model block
        # Due to how SKLearn pipelines work, we have to set the model fit parameters using a deeply nested dictionary
        # Then we convert it to a flat dictionary with __ as the separator between each level
        fit_params = {
            "model_loop_pipeline_step": {
                "model_blocks_pipeline_step": {
                    name: {"train_indices": train_indices, "test_indices": test_indices}
                    for name, _ in model_pipeline.named_steps.model_loop_pipeline_step.named_steps.model_blocks_pipeline_step.steps
                },
                "pretrain_pipeline_step": {
                    "train_indices": train_indices,
                },
            }
        }
        fit_params_flat = flatten_dict(fit_params)

        # Fit the pipeline
        model_pipeline.fit(X, y, **fit_params_flat)


if __name__ == "__main__":
    # Run with dask client, which will automatically close if there is an error
    with Client() as client:
        logger.info(f"Client: {client}")
        run_cv()
