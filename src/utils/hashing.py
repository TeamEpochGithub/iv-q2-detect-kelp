"""Hashing.py contains functions for hashing objects."""

from joblib import hash
from omegaconf import DictConfig

from src.logging_utils.logger import logger


def hash_models(cfg: DictConfig) -> list[str]:
    """Hash the model pipeline.

    :param cfg: The config object. Created with Hydra or OmegaConf.
    :return: The hash of the model pipeline.
    """
    model_hashes = []
    if "model" in cfg:
        model_hash = str(hash(str(cfg["model"]) + str(cfg["test_size"])))
        model_hashes.append(model_hash)
    elif "ensemble" in cfg:
        for model in cfg.ensemble.models.values():
            model_hash = str(hash(str(model) + str(cfg["test_size"])))
            model_hashes.append(model_hash)
    else:
        raise ValueError("No model or ensemble specified in config.")

    logger.info(f"Model hashes: {model_hashes}")
    return model_hashes


def hash_model(cfg: DictConfig) -> str:
    """Hash the model pipeline.

    :param cfg: The config object. Created with Hydra or OmegaConf.
    :return: The hash of the model pipeline.
    """
    model_hashes = hash_models(cfg)
    return model_hashes[0]


def hash_scalers(cfg: DictConfig) -> list[str]:
    """Hash the scaler pipeline.

    :param cfg: The config object. Created with Hydra or OmegaConf.
    :return: The hash of the scaler pipeline or None if scaler does not exist in config.
    """
    scaler_hashes = []
    if "model" in cfg:
        # Check if scaler is in the config, if not we return None
        pretrain = cfg.get("model", {}).get("model_loop_pipeline", {}).get("pretrain_pipeline")
        has_pretrain = pretrain is not None
        has_scaler = False if not has_pretrain else pretrain.get("scaler") is not None
        if has_scaler:
            scaler_hash = hash(str(cfg["model"]["model_loop_pipeline"]["pretrain_pipeline"]) + str(cfg["model"]["feature_pipeline"]) + str(cfg["test_size"]))
            scaler_hashes.append(scaler_hash)
        else:
            scaler_hashes.append(None)
    elif "ensemble" in cfg:
        for model in cfg.ensemble.models.values():
            pretrain = model.get("model_loop_pipeline", {}).get("pretrain_pipeline")
            has_pretrain = pretrain is not None
            has_scaler = False if not has_pretrain else pretrain.get("scaler") is not None
            if has_scaler:
                scaler_hash = hash(str(model["model_loop_pipeline"]["pretrain_pipeline"]) + str(model["feature_pipeline"]) + str(cfg["test_size"]))
                scaler_hashes.append(scaler_hash)
            else:
                scaler_hashes.append(None)

    logger.info(f"Scaler hashes: {scaler_hashes}")

    return scaler_hashes


def hash_scaler(cfg: DictConfig) -> str | None:
    """Hash the scaler pipeline.

    :param cfg: The config object. Created with Hydra or OmegaConf.
    :return: The hash of the scaler pipeline or None if scaler does not exist in config.
    """
    scaler_hashes = hash_scalers(cfg)
    return scaler_hashes[0]
