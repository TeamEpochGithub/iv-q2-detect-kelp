"""Custom DataParallel class."""
from torch import nn


class CustomDataParallel(nn.DataParallel):  # type: ignore[type-arg]
    """Custom DataParallel class."""

    def __repr__(self) -> str:
        """Return the representation of the module. This is to get the same hash for the model with and without DataParallel."""
        return self.module.__repr__()
