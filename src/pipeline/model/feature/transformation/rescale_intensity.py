"""Rescale the intensity of the labels."""
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

import dask.array as da
import numpy as np
import numpy.typing as npt
import skimage as ski
from annotated_types import Len
from sklearn.base import BaseEstimator, TransformerMixin

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class RescaleIntensity(TransformerMixin, BaseEstimator):
    """Pipeline step to perform rescaling the intensity of the labels.

    :param out_range: The output range. If a scalar, the output range is (out_range, 1 - out_range).

    Example:
    -------
    >>> y = np.array([[0, 1, 1], [1, 0, 0], [0, 1, 1]])
    >>> ri = RescaleIntensity(out_range=[0.1, 0.9])
    >>> ri.fit_transform(y)
    array([[0.1, 0.9, 0.9],
           [0.9, 0.1, 0.1],
           [0.1, 0.9, 0.9]])
    """

    # noinspection PyTypeHints
    out_range: float | Annotated[Sequence[float], Len(2, 2)]

    def fit(self, X: npt.ArrayLike | da.Array, y: npt.ArrayLike | da.Array | None = None) -> Self:  # noqa: ARG002
        """Do nothing. Exists for Pipeline compatibility.

        :param X: UNUSED data to fit
        :param y: UNUSED target variable
        :return: Itself
        """
        return self

    def transform(self, y: npt.NDArray[np.float_ | np.bool_] | da.Array) -> npt.NDArray[np.float_] | da.Array:
        """Rescale the intensity as described.

        :param y: The binary labels.
        :return: The rescaled labels.
        """
        if isinstance(self.out_range, Sequence):
            return ski.exposure.rescale_intensity(y, (0, 1), tuple(self.out_range))
        return ski.exposure.rescale_intensity(y, (0, 1), (self.out_range, 1 - self.out_range))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
