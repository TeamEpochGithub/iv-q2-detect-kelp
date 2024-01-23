"""Schema for the cross validation configuration."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.wandb_config import WandBConfig


@dataclass
class CVConfig:
    """Schema for the cross validation configuration.

    :param model: Model pipeline.
    :param ensemble: Ensemble pipeline.
    :param raw_data_path: Path to the raw data.
    :param raw_target_path: Path to the raw target.
    :param scorer: Scorer object to be instantiated.
    :param cache_size: Cache size for the pipeline.
    :param wandb: Whether to log to Weights & Biases and other settings.
    :param splitter: Cross validation splitter.
    :param allow_multiple_instances: Whether to allow multiple instances of training at the same time.
    """

    model: Any
    ensemble: Any
    raw_data_path: Path
    raw_target_path: Path
    metadata_path: Path
    scorer: Any
    cache_size: int
    wandb: WandBConfig
    splitter: Any
    allow_multiple_instances: bool = False
