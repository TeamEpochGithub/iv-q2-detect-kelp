"""Check if the model and scaler hashes are cached already, if not give an error."""
import glob

from omegaconf import DictConfig

from src.logging_utils.logger import logger
from src.utils.hashing import hash_models, hash_scalers


def check_hash_submit(cfg: DictConfig) -> tuple[list[str], list[str]]:
    """Check if the model and scaler hashes are cached already, if not give an error."""
    # Hash representation of model pipeline only based on model and test size
    model_hashes = hash_models(cfg)

    # Hash representation of scaler based on pretrain, feature_pipeline and test_size
    scaler_hashes = hash_scalers(cfg)

    # Check if models are cached already, if not give an error
    for model_hash in model_hashes:
        if not glob.glob(f"tm/{model_hash}.pt"):
            raise FileNotFoundError(f"Model {model_hash} not found. Please train the model first.")

    # Check if scalers are cached already, if not give an error
    for scaler_hash in scaler_hashes:
        if scaler_hash is not None and not glob.glob(f"tm/{scaler_hash}.scaler"):
            raise FileNotFoundError(f"Scaler {scaler_hash} not found. Please train the model first.")

    return model_hashes, scaler_hashes


def check_hash_train(cfg: DictConfig) -> tuple[list[str], list[str]]:
    """Check if the model and scaler hashes are cached already, if not give an error."""
    # Hash representation of model pipeline only based on model and test size
    model_hashes = hash_models(cfg)

    # Hash representation of scaler based on pretrain, feature_pipeline and test_size
    scaler_hashes = hash_scalers(cfg)

    # Check if models are cached already, if not give an error
    for i, model_hash in enumerate(model_hashes):
        if glob.glob(f"tm/{model_hash}.pt"):
            model_hashes[i] = ""

    # Check if scalers are cached already, if not give an error
    for i, scaler_hash in enumerate(scaler_hashes):
        if scaler_hash is None:
            logger.warning(f"No scaler found for model {i}. Training without scaler.")
        elif glob.glob(f"tm/{scaler_hash}.scaler"):
            scaler_hashes[i] = ""

    return model_hashes, scaler_hashes
