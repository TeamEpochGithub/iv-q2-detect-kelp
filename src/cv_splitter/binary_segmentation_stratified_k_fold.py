"""Stratified k-fold cross-validation splitter for binary segmentation tasks."""
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Annotated

import dask.array as da
import numpy as np
import numpy.typing as npt
from annotated_types import Ge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer


@dataclass
class BinarySegmentationStratifiedKFold:
    """Stratified k-fold cross-validation splitter for binary segmentation tasks.

    :param n_splits: Number of folds. Must be at least 2.
    :param shuffle: Whether to shuffle the data before splitting into batches.
    :param random_state: Seed for the random number generator. Defaults to None.
    :param coverages: The pre-computed coverage of each image. If None, the coverage will be computed from the target.
    """

    n_splits: Annotated[int, Ge(2)] = 5
    shuffle: bool = True
    random_state: int | None = 42
    coverages: npt.NDArray[np.float_] | None = None

    _bins: npt.NDArray[np.uintp] = field(init=False)

    def split(self, X: npt.ArrayLike, y: npt.NDArray[np.bool_] | da.Array, groups: Iterable[int] | None = None) -> Iterator[tuple[list[int], list[int]]]:  # noqa: ARG002
        """Generate indices to split data into training and test set.

        :param X: The data to split.
        :param y: The target variable, used only for computing the coverages if not given.
        :param groups: UNUSED Group labels for the samples used while splitting the dataset into train/test set. Exists for compatibility.
        :return: The training and test set indices for that split.
        """
        if self.coverages is None:
            if isinstance(y, da.Array):
                y = y.compute()
            self.coverages = y.astype(np.float32).mean(axis=tuple(range(1, y.ndim)))

        # Discretize the coverages into bins
        kbd = KBinsDiscretizer(n_bins=self.n_splits, encode="ordinal", strategy="quantile")
        self._bins = kbd.fit_transform(self.coverages.reshape(-1, 1)).flatten().astype(np.min_scalar_type(self.n_splits))

        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        yield from kf.split(X, self._bins)
