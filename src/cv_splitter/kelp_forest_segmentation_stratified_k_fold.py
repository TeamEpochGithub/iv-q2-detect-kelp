"""A wrapper around BinarySegmentationStratifiedKFold that uses the metadata file specific to the Kelp Forest Segmentation competition data."""
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import dask.array as da
import numpy as np
import numpy.typing as npt
import pandas as pd
from annotated_types import Ge

from src.cv_splitter.binary_segmentation_stratified_k_fold import BinarySegmentationStratifiedKFold


@dataclass
class KelpForestSegmentationStratifiedKFold:
    """A wrapper around BinarySegmentationStratifiedKFold that uses the metadata file specific to the Kelp Forest Segmentation competition data.

    :param n_splits: Number of folds. Must be at least 2.
    :param shuffle: Whether to shuffle the data before splitting into batches.
    :param random_state: Seed for the random number generator. Defaults to None.
    :param metadata_path: Path to the metadata file.
    """

    metadata_path: Path
    n_splits: Annotated[int, Ge(2)] = 5
    shuffle: bool = True
    random_state: int | None = 42

    def split(self, X: npt.ArrayLike, y: npt.NDArray[np.bool_] | da.Array, groups: Iterable[int] | None = None) -> Iterator[tuple[list[int], list[int]]]:  # noqa: ARG002
        """Generate indices to split data into training and test set.

        :param X: The data to split.
        :param y: The target variable to try to predict in the case of supervised learning.
        :param groups: UNUSED Group labels for the samples used while splitting the dataset into train/test set. Exists for compatibility.
        :return: The training and test set indices for that split.
        """
        if self.metadata_path.exists():
            logging.info("Metadata file found at %s", self.metadata_path)
            metadata = pd.read_csv(self.metadata_path).sort_values(by="tile_id")

            kelp_coverages = metadata.query("in_train")["kelp"].to_numpy()
            yield from BinarySegmentationStratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state, coverages=kelp_coverages).split(X, y)
        else:
            logging.info("Metadata file not found at %s, using the target to compute the coverages", self.metadata_path)
            logging.info("Hint: You can generate the metadata file with `notebooks/create_metadataset.ipynb`")
            yield from BinarySegmentationStratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state).split(X, y)
