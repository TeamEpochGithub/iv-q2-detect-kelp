"""The caching tif pipeline is responsible for loading and storing all data to disk."""
import sys
from dataclasses import dataclass

import dask.array as da
from sklearn.base import BaseEstimator, TransformerMixin

from src.pipeline.caching.util.store_raw import store_raw

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class CacheTIFBlock(BaseEstimator, TransformerMixin):
    """The caching pipeline is responsible for loading and storing the data to disk.

    :param data_path: The path to the data
    """

    data_path: str

    def fit(self, X: da.Array, y: da.Array | None = None) -> Self:  # noqa: ARG002
        """Do nothing. Exists for Pipeline compatibility.

        :param X: UNUSED data to fit.
        :param y: UNUSED target variable.
        :return: Itself.
        """
        return self

    def transform(self, X: da.Array) -> da.Array:
        """Transform the data.

        :param X: The data to transform
        :return: The transformed data
        """
        return store_raw(self.data_path, X)
