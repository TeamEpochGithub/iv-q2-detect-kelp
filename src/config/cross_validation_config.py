"""Schema for the cross validation configuration."""
from dataclasses import dataclass
from typing import Any

from src.config.wandb_config import WandBConfig


@dataclass
class CVConfig:
    """Schema for the cross validation configuration.

    :param model: Model pipeline.
    :param n_splits: Number of splits for cross validation.
    :param raw_data_path: Path to the raw data.
    :param raw_target_path: Path to the raw target.
    :param scorer: Scorer object to be instantiated.
    :param wandb: Whether to log to Weights & Biases and other settings.
    """

    model: Any
    n_splits: int
    raw_data_path: str
    raw_target_path: str
    scorer: str
    wandb: WandBConfig
