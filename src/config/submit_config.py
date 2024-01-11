"""Schema for the submit configuration."""
from dataclasses import dataclass
from typing import Any


@dataclass
class SubmitConfig:
    """Schema for the submit configuration.

    :param model: Model pipeline.
    :param test_size: The size of the test set ∈ [0, 1].
    :param raw_data_path: Path to the raw data.
    :param raw_target_path: Path to the raw target.
    """

    model: Any
    ensemble: Any
    test_size: float
    raw_data_path: str
    raw_target_path: str
