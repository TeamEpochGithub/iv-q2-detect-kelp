"""cv.py is the main script for doing cv and will take in the raw data, do cv and log the cv results."""
import os
import warnings
from pathlib import Path

import hydra
import randomname
import wandb
from distributed import Client
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold

from src.config.cross_validation_config import CVConfig
from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator
from src.utils.script.generate_params import generate_cv_params
from src.utils.script.reset_wandb_env import reset_wandb_env
from src.utils.setup import setup_config, setup_pipeline, setup_train_data, setup_wandb

warnings.filterwarnings("ignore", category=UserWarning)
# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"

# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_cv", node=CVConfig)


@hydra.main(version_base=None, config_path="conf", config_name="cv")
def run_cv(cfg: DictConfig) -> None:  # TODO(Jeffrey): Use CVConfig instead of DictConfig
    """Do cv on a model pipeline with K fold split."""
    print_section_separator("Q2 Detect Kelp States -- CV")

    import coloredlogs

    coloredlogs.install()

    # Check for missing keys in the config file
    setup_config(cfg)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Lazily read the raw data with dask, and find the shape after processing
    X, y = setup_train_data(cfg.raw_data_path, cfg.raw_target_path)

    # Perform stratified k-fold cross validation, where the group of each image is determined by having kelp or not.
    kf = StratifiedKFold(n_splits=cfg.n_splits)
    stratification_key = y.compute().reshape(y.shape[0], -1).max(axis=1)

    # Set up Weights & Biases group name
    wandb_group_name = randomname.get_name()

    for i, (train_indices, test_indices) in enumerate(kf.split(X, stratification_key)):
        # https://github.com/wandb/wandb/issues/5119
        # This is a workaround for the issue where sweeps override the run id annoyingly
        reset_wandb_env()

        # Print section separator
        print_section_separator(f"CV - Fold {i}")
        logger.info(f"Train/Test size: {len(train_indices)}/{len(test_indices)}")

        if cfg.wandb.enabled:
            setup_wandb(cfg, "cv", output_dir, name=f"{wandb_group_name}_{i}", group=wandb_group_name)

        logger.info("Creating clean pipeline for this fold")
        model_pipeline = setup_pipeline(cfg, output_dir, is_train=True)

        # Generate the parameters for training
        fit_params = generate_cv_params(cfg, model_pipeline, train_indices, test_indices)

        # Fit the pipeline and get predictions
        predictions = model_pipeline.fit_transform(X, y, **fit_params)
        scorer = instantiate(cfg.scorer)
        score = scorer(y[test_indices].compute(), predictions[test_indices])
        logger.info(f"Score: {score}")
        wandb.log({"Score": score})

        logger.info("Finishing wandb run")
        wandb.finish()


if __name__ == "__main__":
    # Run with dask client, which will automatically close if there is an error
    with Client() as client:
        logger.info(f"Client: {client}")
        run_cv()
