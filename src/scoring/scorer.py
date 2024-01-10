"""Abstract scorer class from which other scorers inherit from."""

from abc import ABC, abstractmethod
from typing import List


class Scorer(ABC):
    """Abstract scorer class from which other scorers inherit from."""

    def __init__(self, name: str):
        """Initialize the scorer with a name."""
        self.name = name

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Calculate the score."""
        pass

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return self.name