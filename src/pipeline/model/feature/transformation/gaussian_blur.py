"""Pipeline step to smooth the labels."""
import sys
import time
from dataclasses import dataclass

import dask.array as da
import dask_image.ndfilters._gaussian as gaf
import dask_image.ndfilters._utils as gau
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_utils.logger import logger

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class GaussianBlur(BaseEstimator, TransformerMixin):
    """Pipeline step to smooth the labels.

    :param smoothing: The smoothing factor
    """

    sigma: float = 0.5

    def fit(self, X: da.Array, y: da.Array | None = None) -> Self:  # noqa: ARG002
        """Do nothing. Exists for Pipeline compatibility.

        :param X: UNUSED data to fit.
        :param y: UNUSED target variable.
        :return: The fitted transformer.
        """
        return self

    def transform(self, X: da.Array) -> da.Array:
        """Transform the data.

        :param X: The data to transform
        :return: The transformed data
        """
        time_start = time.time()

        X = X.astype(np.float32)

        _, boundary = gau._get_depth_boundary(X.ndim, 5, "none")  # noqa: SLF001
        meta = X._meta  # noqa: SLF001

        result = X.map_overlap(
            gaf.dispatch_gaussian_filter(X),
            depth=0,
            boundary=boundary,
            dtype=X.dtype,
            meta=meta,
            sigma=self.sigma,
            order=0,
            mode="reflect",
            cval=0,
            truncate=4.0,
        )

        logger.info(f"Gaussian blur complete in: {time.time() - time_start} seconds.")

        return result
