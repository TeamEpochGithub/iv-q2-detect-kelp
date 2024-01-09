"""Schema for the train configuration."""
from dataclasses import dataclass
from typing import Any

from src.config.wandb_config import WandBConfig


@dataclass
class TrainConfig:
    """Schema for the train configuration.

    :param model: The model pipeline.
    :param test_size: The size of the test set ∈ [0, 1].
    :param raw_data_path: Path to the raw data.
    :param raw_target_path: Path to the raw target.
    :param wandb: Whether to log to Weights & Biases and other settings.
    """

    model: Any
    test_size: float
    raw_data_path: str
    raw_target_path: str
    wandb: WandBConfig
