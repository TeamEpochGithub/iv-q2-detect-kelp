"""Hashing.py contains functions for hashing objects."""

from joblib import hash
from omegaconf import DictConfig

from src.logging_utils.logger import logger


def hash_model(cfg: DictConfig) -> str:
    """Hash the model pipeline.

    :param cfg: The config object. Created with Hydra or OmegaConf.
    :return: The hash of the model pipeline.
    """
    model_hash = str(hash(str(cfg["model"]) + str(cfg["test_size"])))
    logger.info(f"Model hash: {model_hash}")
    return model_hash


def hash_scaler(cfg: DictConfig) -> str | None:
    """Hash the scaler pipeline.

    :param cfg: The config object. Created with Hydra or OmegaConf.
    :return: The hash of the scaler pipeline or None if scaler does not exist in config.
    """
    # Check if scaler is in the config, if not we return None
    pretrain = cfg.get("model", {}).get("pipeline", {}).get("model_loop_pipeline", {}).get("pretrain_pipeline")
    has_pretrain = pretrain is not None
    has_scaler = False if not has_pretrain else pretrain.get("scaler") is not None
    if has_scaler:
        scaler_hash = hash(
            str(cfg["model"]["pipeline"]["model_loop_pipeline"]["pretrain_pipeline"]) + str(cfg["model"]["pipeline"]["feature_pipeline"]) + str(cfg["test_size"])
        )
        logger.info(f"Scaler hash: {scaler_hash}")
        return scaler_hash
    return None
