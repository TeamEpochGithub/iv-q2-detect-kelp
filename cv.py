"""Train.py is the main script for training the model and will take in the raw data and output a trained model."""
import time
import warnings
from dataclasses import dataclass
from typing import Any

import hydra
from dask_image.imread import imread
from distributed import Client
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn import set_config
from sklearn.base import estimator_html_repr
from sklearn.model_selection import StratifiedKFold

from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator
from src.utils.flatten_dict import flatten_dict

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class CVConfig:
    """Schema for the train config yaml file."""

    model: Any
    n_splits: int
    raw_data_path: str = "data/raw/train_satellite"
    raw_target_path: str = "data/raw/train_kelp"


# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_cv", node=CVConfig)


@hydra.main(version_base=None, config_path="conf", config_name="cv")
def run_train(cfg: CVConfig) -> None:
    """Train a model pipeline with a train-test split."""
    # Check for missing keys in the config file
    missing = OmegaConf.missing_keys(cfg)
    if missing:
        raise ValueError(f"Missing keys in config file\n{missing}")

    # Coloured logs
    import coloredlogs

    coloredlogs.install()

    # Print section separator
    print_section_separator("Q2 Detect Kelp States -- CV")

    # Set up the pipeline
    logger.info("Setting up the pipeline")
    orig_time = time.time()
    model_pipeline = instantiate(cfg.model.pipeline).get_pipeline()
    logger.info(f"Pipeline setup time: {time.time() - orig_time} seconds")
    logger.debug(f"Pipeline: {model_pipeline}")

    # Save the pipeline to an HTML file, next to the log file in the hydra output
    set_config(display="diagram")
    pipeline_html = estimator_html_repr(model_pipeline)
    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    with open(f"{out_dir}/pipeline.html", "w", encoding="utf-8") as f:
        f.write(pipeline_html)

    # Read in the raw data
    logger.info("Lazily reading the raw data")
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

    # Perform stratified k-fold cross validation, where the group of each image is determined by having kelp or not.
    kf = StratifiedKFold(n_splits=cfg.n_splits)
    stratification_key = y.compute().reshape(y.shape[0], -1).max(axis=1)
    for i, (train_indices, test_indices) in enumerate(kf.split(x_processed, stratification_key)):
        print_section_separator(f"CV - Fold {i}")

        logger.info("Creating clean pipeline for this fold")
        model_pipeline = instantiate(cfg.model.pipeline).get_pipeline()

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


if __name__ == "__main__":
    # Run with dask client, which will automatically close if there is an error
    with Client() as client:
        logger.info(f"Client: {client}")
        run_train()
