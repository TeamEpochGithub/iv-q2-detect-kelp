"""Stratified k-fold cross-validation splitter for binary segmentation tasks."""
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Annotated

import dask.array as da
import numpy as np
import numpy.typing as npt
from annotated_types import Ge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import KBinsDiscretizer


@dataclass
class BinarySegmentationStratifiedKFold:
    """Stratified k-fold cross-validation splitter for binary segmentation tasks.

    :param n_splits: Number of folds. Must be at least 2.
    :param shuffle: Whether to shuffle the data before splitting into batches.
    :param random_state: Seed for the random number generator. Defaults to None.
    :param mean_coverages: The pre-computed mean coverage of each image.
    """

    mean_coverages: npt.NDArray[np.float_] | da.Array
    n_splits: Annotated[int, Ge(2)] = 5
    shuffle: bool = True
    random_state: int | None = None

    def split(self, X: npt.ArrayLike, y: npt.NDArray[np.bool_] | da.Array, groups: Iterable[int] | None = None) -> Iterator[tuple[list[int], list[int]]]:
        """Generate indices to split data into training and test set.

        :param X: The data to split.
        :param y: The target variable to try to predict in the case of supervised learning.
        :param groups: UNUSED Group labels for the samples used while splitting the dataset into train/test set.
        :return: The training and test set indices for that split.
        """
        kbd = KBinsDiscretizer(n_bins=self.n_splits, encode="ordinal", strategy="quantile")
        bins = kbd.fit_transform(self.mean_coverages.reshape(-1, 1)).flatten().astype(np.min_scalar_type(self.n_splits))

        kf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        yield from kf.split(X, y, groups=bins)
