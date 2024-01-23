"""cv.py is the main script for doing cv and will take in the raw data, do cv and log the cv results."""
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
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold

import wandb
from src.config.cross_validation_config import CVConfig
from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator
from src.utils.script.generate_params import generate_cv_params
from src.utils.script.lock import Lock
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

    # Check for missing keys in the config file
    setup_config(cfg)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Lazily read the raw data with dask, and find the shape after processing
    X, y = setup_train_data(cfg.raw_data_path, cfg.raw_target_path)
    X = X[:200]
    y = y[:200]

    # Perform stratified k-fold cross validation, where the group of each image is determined by having kelp or not.
    kf = StratifiedKFold(n_splits=cfg.n_splits)
    stratification_key = y.compute().reshape(y.shape[0], -1).max(axis=1)

    # Set up Weights & Biases group name
    wandb_group_name = randomname.get_name()

    # If wandb is enabled, set up the sweep
    sweep_run = None
    scores = []
    if cfg.wandb.enabled and wandb.sweep:
        print_section_separator("Sweep")
        # Set up the sweep config
        sweep_run = wandb.init(group=wandb_group_name, reinit=True)
        sweep_run.save()


    for i, (train_indices, test_indices) in enumerate(kf.split(X, stratification_key)):
        # https://github.com/wandb/wandb/issues/5119
        # This is a workaround for the issue where sweeps override the run id annoyingly
        reset_wandb_env()

        # Print section separator
        print_section_separator(f"CV - Fold {i}")
        logger.info(f"Train/Test size: {len(train_indices)}/{len(test_indices)}")

        if cfg.wandb.enabled:
            fold_run = setup_wandb(cfg, "cv", output_dir, name=f"{wandb_group_name}_{i}", group=wandb_group_name)

        for key, value in os.environ.items():
            if key.startswith("WANDB_"):
                logger.info(f"{key}: {value}")

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
        if fold_run is not None:
            fold_run.log({"Score": score})
        scores.append(score)

        logger.info("Finishing wandb run")
        wandb.join()

    print(sweep_run)
    
    if sweep_run is not None:
        # Log the average score
        if len(scores) > 0:
            sweep_run.log(dict(sweep_score=sum(scores) / len(scores)))
    else:
        print_section_separator("CV Results")

if __name__ == "__main__":
    run_cv()
