"""Custom standard scaler implementation."""
import sys
from dataclasses import dataclass

import dask.array as da
from sklearn.base import BaseEstimator

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class CustomStandardScaler(BaseEstimator):
    """Custom standard scaler implementation."""

    def fit(self, X: da.Array, y: da.Array | None = None) -> Self:  # noqa: ARG002
        """Fit the scaler."""
        self.mean_per_channel = X.mean(axis=(0, 2, 3)).compute()
        self.std_per_channel = X.std(axis=(0, 2, 3)).compute()
        return self

    def transform(self, X: da.Array) -> da.Array:
        """Transform the data."""
        return (X - self.mean_per_channel.reshape(1, -1, 1, 1)) / self.std_per_channel.reshape(1, -1, 1, 1)
