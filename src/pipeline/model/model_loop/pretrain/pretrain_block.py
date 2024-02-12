"""Pretrain block class."""
import sys
import time
from abc import abstractmethod
from dataclasses import dataclass

import dask.array as da
from joblib import hash
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_utils.logger import logger
from src.pipeline.caching.util.store_raw import store_raw

if sys.version_info < (3, 11):
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
        self.pretrain_path = "data/training"

    @abstractmethod
    def fit(self, X: da.Array, y: da.Array, train_indices: list[int], *, save_pretrain: bool = True, save_pretrain_with_split: bool = False) -> Self:
        """Return self, no fitting necessary.

        :param X: Data to fit
        :param y: Target data
        :param train_indices: Train indices
        :param save_pretrain: Whether to save the pretrain
        :param save_pretrain_with_split: Whether to save the pretrain with the split
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

    def save_pretrain(self, X: da.Array) -> da.Array:
        """Save the pretrain data.

        :param X: Data to save
        :param train_indices: Train indices
        :return: Saved data
        """
        start_time = time.time()
        logger.info("Saving pretrain data...")
        # Save the pretrain data
        result = store_raw(self.pretrain_path + "/" + hash(self.prev_hash), X)
        logger.info("Saved pretrain data in %s seconds", time.time() - start_time)
        return result

    def set_pretrain_path(self, path: str) -> None:
        """Set the path for saving pretrain.

        :param path: path to save data
        """
        self.pretrain_path = path

    def train_split_hash(self, train_indices: list[int]) -> str:
        """Split the hash on train split.

        :param train_indices: Train indices
        :return: Split hash
        """
        self.prev_hash = hash(self.prev_hash + str(train_indices))
        return self.prev_hash
