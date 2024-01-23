"""Feature engineering step that create features based on land and elevation."""
import sys
from dataclasses import dataclass
from typing import Literal

import dask.array as da
import numpy as np
import scipy
from numpy import typing as npt
from sklearn.base import BaseEstimator, TransformerMixin

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


def _distance(elev: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Compute the distance to land.

    :param elev: The elevation band (H,W)
    :return: The distance to land (H,W)
    """
    mask = elev <= 0
    if np.all(mask):
        return np.full_like(elev, 400)
    return scipy.ndimage.distance_transform_edt(mask)


def _closeness(elev: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Compute the closeness to land. Roughly the inverse of the distance to land.

    :param elev: The elevation band (H,W)
    :return: The closeness to land (H,W)
    """
    mask = elev <= 0
    if np.all(mask):
        return np.zeros_like(elev)
    dist = scipy.ndimage.distance_transform_edt(mask)
    return 1 / (1 + dist * 0.1)


def _binary(elev: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Compute a binary mask of land pixels.

    :param elev: The elevation band (H,W)
    :return: The binary mask (H,W)
    """
    return (elev > 0).astype(np.float32)


# map of mode to functions
mode_to_func = {
    "distance": _distance,
    "closeness": _closeness,
    "binary": _binary,
}


@dataclass
class Shore(BaseEstimator, TransformerMixin):
    """Shore can compute multiple features based on the elevation band.

    :param elevation_band: The index of the elevation band
    :param mode: The type of feature to compute, one of "distance", "closeness", "binary"
    """

    mode: Literal["distance", "closeness", "binary"] = "distance"
    elevation_band: int = 6

    def __post_init__(self) -> None:
        """Initialize the Shore block."""
        # Check if the mode is valid
        if self.mode not in mode_to_func:
            raise ValueError(f"Invalid mode '{self.mode}', must be one of {list(mode_to_func.keys())}")

    def fit(self, X: da.Array, y: da.Array | None = None) -> Self:  # noqa: ARG002
        """Fit the transformer, does not compute anything.

        :param X: The data to fit
        :param y: The target variable
        :return: The fitted transformer
        """
        return self

    def transform(self, X: da.Array, y: da.Array | None = None) -> da.Array:  # noqa: ARG002
        """Transform the data.

        :param X: The data to transform
        :param y: The target variable
        :return: The transformed data
        """
        # Re-chunk so that chunks contains complete images
        X = X.rechunk({0: "auto", 1: -1, 2: -1, 3: -1})

        # Map the function over each chunk
        return X.map_blocks(self.chunk_func, dtype="float32", chunks=(X.chunks[0], (X.chunks[1][0] + 1,), X.chunks[2], X.chunks[3]), meta=np.array((), dtype=np.float32))

    def chunk_func(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply the function to a chunk of data.

        :param x: The chunk of data
        :return: The transformed chunk
        """
        result = np.zeros((x.shape[0], 1, 350, 350), dtype="float32")
        for i in range(x.shape[0]):
            result[i] = mode_to_func[self.mode](x[i, self.elevation_band])
        return np.concatenate([x, result], axis=1)
