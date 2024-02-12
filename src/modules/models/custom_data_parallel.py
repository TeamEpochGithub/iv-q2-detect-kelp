from torch import nn


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel class."""
    def __repr__(self):
        """Return the representation of the module. This is to get the same hash for the model with and without DataParallel."""
        return self.module.__repr__()