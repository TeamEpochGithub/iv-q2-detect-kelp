"""The main script for Cross Validation. Takes in the raw data, does CV and logs the results."""
import copy
import os
import warnings
from contextlib import nullcontext
from pathlib import Path

import hydra
import randomname
from distributed import Client
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig

import wandb
from src.config.cross_validation_config import CVConfig
from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator
from src.utils.script.generate_params import generate_cv_params
from src.utils.script.lock import Lock
from src.utils.script.reset_wandb_env import reset_wandb_env
from src.utils.seed_torch import set_torch_seed
from src.utils.setup import setup_config, setup_pipeline, setup_train_data, setup_wandb

warnings.filterwarnings("ignore", category=UserWarning)
# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"

# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_cv", node=CVConfig)


@hydra.main(version_base=None, config_path="conf", config_name="cv")
def run_cv(cfg: DictConfig) -> None:  # TODO(Jeffrey): Use CVConfig instead of DictConfig
    """Do cv on a model pipeline with K fold split. Entry point for Hydra which loads the config file."""
    # Run the cv config with a dask client, and optionally a lock
    optional_lock = Lock if not cfg.allow_multiple_instances else nullcontext
    with optional_lock(), Client() as client:
        logger.info(f"Client: {client}")
        run_cv_cfg(cfg)


def run_cv_cfg(cfg: DictConfig) -> None:
    """Do cv on a model pipeline with K fold split."""
    print_section_separator("Q2 Detect Kelp States -- CV")

    import coloredlogs

    coloredlogs.install()

    # Set seed
    set_torch_seed()

    # Check for missing keys in the config file
    setup_config(cfg)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Lazily read the raw data with dask, and find the shape after processing
    X, y = setup_train_data(cfg.raw_data_path, cfg.raw_target_path)

    # Set up Weights & Biases group name
    wandb_group_name = randomname.get_name()

    for i, (train_indices, test_indices) in enumerate(instantiate(cfg.splitter).split(X, y)):
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

        # Fit the pipeline
        target_pipeline = model_pipeline.get_target_pipeline()
        original_y = copy.deepcopy(y)

        if target_pipeline is not None:
            print_section_separator("Target pipeline")
            y = target_pipeline.fit_transform(y)

        # Fit the pipeline and get predictions
        predictions = model_pipeline.fit_transform(X, y, **fit_params)
        scorer = instantiate(cfg.scorer)
        score = scorer(original_y[test_indices].compute(), predictions[test_indices])
        logger.info(f"Score: {score}")
        wandb.log({"Score": score})

        logger.info("Finishing wandb run")
        wandb.finish()


if __name__ == "__main__":
    run_cv()
