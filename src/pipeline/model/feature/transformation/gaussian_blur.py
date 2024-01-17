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

if sys.version_info < (3, 11):  # Self was added in Python 3.11
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class GaussianBlur(BaseEstimator, TransformerMixin):
    """Pipeline step to smooth the labels.

    :param smoothing: The smoothing factor
    """

    sigma: float = 0.5

    def fit(self, X: da.Array, y: da.Array | None = None) -> Self:
        """Fit the transformer.

        :param X: The data to fit
        :param y: The target variable
        :return: The fitted transformer
        """
        return self

    def transform(self, X: da.Array) -> da.Array:
        """Transform the data.

        :param X: The data to transform
        :param y: The target variable
        :return: The transformed data
        """
        time_start = time.time()

        X = X.astype(np.float32)

        depth, boundary = gau._get_depth_boundary(X.ndim, 5, "none")  # type: ignore

        result = X.map_overlap(
            gaf.dispatch_gaussian_filter(X), depth=0, boundary=boundary, dtype=X.dtype, meta=X._meta, sigma=self.sigma, order=0, mode="reflect", cval=0, truncate=4.0
        )  # type: ignore

        logger.info(f"Gaussian blur complete in: {time.time() - time_start} seconds.")

        return result

    def gaussian_kernel(self, size: int, sigma: float = 1) -> np.ndarray[float, float]:
        """Gaussian kernel returns a 2D Gaussian kernel.

        :param size: The size of the kernel
        :param sigma: The standard deviation of the kernel
        :return: The kernel
        """
        kernel_1d = np.linspace(-(size // 2), size // 2, size)
        for i in range(size):
            kernel_1d[i] = self.dnorm(kernel_1d[i], 0, sigma)
        kernel_2d = np.outer(kernel_1d.T, kernel_1d.T)

        kernel_2d *= 1.0 / kernel_2d.max()

        return kernel_2d

    def dnorm(self, x: np.ndarray, mu: float, sd: float) -> np.ndarray:
        """Density function of normal distribution.

        :param x: The input
        :param mu: The mean
        :param sd: The standard deviation
        """
        return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-(((x - mu) / sd) ** 2) / 2)

    def convolution(self, image: np.ndarray[float, float], kernel: np.ndarray[float, float], *, average: bool = False) -> np.ndarray[float, float]:
        """Convolves the image with the kernel.

        :param image: The image to convolve
        :param kernel: The kernel to convolve with
        :param average: Whether to average the result
        :return: The convolved image
        """
        image_row, image_col = image.shape
        kernel_row, kernel_col = kernel.shape

        output = np.zeros(image.shape)

        pad_height = int((kernel_row - 1) / 2)
        pad_width = int((kernel_col - 1) / 2)

        padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

        padded_image[pad_height : padded_image.shape[0] - pad_height, pad_width : padded_image.shape[1] - pad_width] = image

        for row in range(image_row):
            for col in range(image_col):
                output[row, col] = np.sum(kernel * padded_image[row : row + kernel_row, col : col + kernel_col])

        if average:
            output[row, col] /= kernel.shape[0] * kernel.shape[1]

        return output

    def gaussian_blur(self, image: np.ndarray[float, float]) -> np.ndarray[float, float]:
        """Smooth the image using a gaussian blur.

        :param image: The image to smooth
        :return: The smoothed image
        """
        if image is None:
            return image

        kernel_size = self.kernel_size
        image = image.squeeze()

        kernel = self.gaussian_kernel(kernel_size, sigma=1)
        convolution = self.convolution(image, kernel, average=True)
        return convolution
