"""Scaler block to fit and transform the data."""
import sys
import time
from dataclasses import dataclass, field

import dask.array as da

from src.logging_utils.logger import logger
from src.pipeline.model.model_loop.pretrain.pretrain_block import PretrainBlock

if sys.version_info < (3, 11):  # Self was added in Python 3.11
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class ColumnSelection(PretrainBlock):
    """Block that selects a subset of columns from the data. Should probably be at the start of the pretrain pipeline.

    :param columns: Columns to select.
    """

    columns: list[int] = field(default_factory=list)

    def fit(self, X: da.Array, y: da.Array, train_indices: list[int], *, save_pretrain: bool = True) -> Self:
        """Return self, no fitting necessary.

        :param X: Data to fit
        :param y: Target data
        """
        return self

    def transform(self, X: da.Array) -> da.Array:
        """Transform the data.

        :param X: Data to transform
        """
        logger.info("Selecting columns...")
        start_time = time.time()
        X = X[:, self.columns]
        logger.info("Selected columns in %s seconds", time.time() - start_time)
        return X
