"""Scaler block to fit and transform the data."""
import sys
import time
from dataclasses import dataclass, field

import dask.array as da

from src.logging_utils.logger import logger
from src.pipeline.model.model_loop.pretrain.pretrain_block import PretrainBlock

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class ColumnSelection(PretrainBlock):
    """Block that selects a subset of columns from the data. Should probably be at the start of the pretrain pipeline.

    :param columns: Columns to select.
    """

    columns: list[int] = field(default_factory=list)

    def fit(self, X: da.Array, y: da.Array, train_indices: list[int], *, save_pretrain: bool = True, save_pretrain_with_split: bool = False) -> Self:  # noqa: ARG002
        """Return self, no fitting necessary. Exists for Pipeline compatibility.

        :param X: UNUSED data to fit.
        :param y: UNUSED target data.
        :param train_indices: UNUSED indices of the training data in X.
        :param save_pretrain: UNUSED whether to save this block.
        :param save_pretrain_with_split: UNUSED whether to save this block with the split.
        :return: self
        """
        return self

    def transform(self, X: da.Array) -> da.Array:
        """Transform the data.

        :param X: Data to transform
        :return: Transformed data
        """
        logger.info("Selecting columns...")
        start_time = time.time()
        X = X[:, self.columns]
        logger.info("Selected columns in %s seconds", time.time() - start_time)
        return X
