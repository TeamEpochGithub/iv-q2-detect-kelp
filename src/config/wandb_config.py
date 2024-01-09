"""Schemas for the Weights & Biases configuration."""
from dataclasses import dataclass


@dataclass
class WandBLogCodeConfig:
    """Schema for the code logging to Weights & Biases.

    :param enabled: Whether to log the code to Weights & Biases.
    :param exclude: Regex of files to exclude from logging.
    """

    enabled: bool
    exclude: str = ""


@dataclass
class WandBConfig:
    """Schema for the Weights & Biases configuration.

    :param enabled: Whether to log to Weights & Biases.
    :param log_config: Whether to log the config to Weights & Biases.
    :param log_code: Whether to log the code to Weights & Biases.
    :param tags: Optional list of tags for the run.
    :param notes: Optional notes for the run.
    """

    enabled: bool
    log_config: bool
    log_code: WandBLogCodeConfig
    tags: list[str] | None = None
    notes: str | None = None
