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
from src.utils.flatten_dict import flatten_dict
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

    # Preload the pipeline and save it to HTML
    print_section_separator("Setup pipeline")
    model_pipeline = setup_pipeline(cfg.model.pipeline, output_dir, is_train=True)

    # Lazily read the raw data with dask, and find the shape after processing
    feature_pipeline = model_pipeline.named_steps.feature_pipeline_step
    X, y, x_processed = setup_train_data(cfg.raw_data_path, cfg.raw_target_path, feature_pipeline)

    # Perform stratified k-fold cross validation, where the group of each image is determined by having kelp or not.
    kf = StratifiedKFold(n_splits=cfg.n_splits)
    stratification_key = y.compute().reshape(y.shape[0], -1).max(axis=1)

    # Set up Weights & Biases group name
    wandb_group_name = randomname.get_name()

    for i, (train_indices, test_indices) in enumerate(kf.split(x_processed, stratification_key)):
        print_section_separator(f"CV - Fold {i}")
        logger.info(f"Train/Test size: {len(train_indices)}/{len(test_indices)}")

        if cfg.wandb.enabled:
            setup_wandb(cfg, "CV", output_dir, name=f"Fold {i}", group=wandb_group_name)

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
            }
        }
        # Add pretrain indices if it exists for the scalerblock
        if hasattr(model_pipeline.named_steps.model_loop_pipeline_step.named_steps, "pretrain_pipeline_step") and hasattr(
            model_pipeline.named_steps.model_loop_pipeline_step.named_steps.pretrain_pipeline_step.named_steps, "ScalerBlock"
        ):
            fit_params["model_loop_pipeline_step"]["pretrain_pipeline_step"] = {}
            fit_params["model_loop_pipeline_step"]["pretrain_pipeline_step"]["ScalerBlock"] = {"train_indices": train_indices}  # type: ignore[index]

        fit_params_flat = flatten_dict(fit_params)

        # Fit the pipeline
        print_section_separator("Preprocessing - Transformations")
        model_pipeline.fit(X, y, **fit_params_flat)

        # Only get the predictions for the test indices
        predictions = model_pipeline.predict(X[test_indices])
        scorer = instantiate(cfg.scorer)
        score = scorer(y[test_indices].compute(), predictions)
        logger.info(f"Score: {score}")
        wandb.log({"Score": score})

        if wandb.run is not None:
            wandb.run.finish()


if __name__ == "__main__":
    # Run with dask client, which will automatically close if there is an error
    with Client() as client:
        logger.info(f"Client: {client}")
        run_cv()
