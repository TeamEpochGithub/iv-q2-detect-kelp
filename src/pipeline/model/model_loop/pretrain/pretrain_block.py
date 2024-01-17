"""Pretrain block class."""
import sys
from abc import abstractmethod
from dataclasses import dataclass

import dask.array as da
from joblib import hash
from sklearn.base import BaseEstimator, TransformerMixin

if sys.version_info < (3, 11):  # Self was added in Python 3.11
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class PretrainBlock(BaseEstimator, TransformerMixin):
    """Pretrain block class.

    :param test_size: Test size
    """

    test_size: float = 0.2

    def __post_init__(self) -> None:
        """Post init function."""
        self.set_hash("")

    @abstractmethod
    def fit(self, X: da.Array, y: da.Array, train_indices: list[int], *, save_pretrain: bool = True) -> Self:
        """Return self, no fitting necessary.

        :param X: Data to fit
        :param y: Target data
        :param train_indices: Train indices
        :param save_pretrain: Whether to save the pretrain
        :return: self
        """

    @abstractmethod
    def transform(self, X: da.Array) -> da.Array:
        """Transform the data.

        :param X: Data to transform
        :return: Transformed data
        """

    def set_hash(self, prev_hash: str) -> str:
        """set_hash function sets the hash for the pipeline.

        :param prev_hash: previous hash
        :return: hash
        """
        curr_hash = hash(str(self) + prev_hash)

        self.prev_hash = curr_hash

        return curr_hash
