"""Abstract scorer class from which other scorers inherit from."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Scorer(ABC):
    """Abstract scorer class from which other scorers inherit from."""

    def __init__(self, name: str) -> None:
        """Initialize the scorer with a name."""
        self.name = name

    @abstractmethod
    def __call__(self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> float:
        """Calculate the score."""

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return self.name
