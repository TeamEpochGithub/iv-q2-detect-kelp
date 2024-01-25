from unittest import TestCase

import dask.array as da
import numpy as np

from src.pipeline.model.feature.transformation.to_zero import ToZero


class TestToZero(TestCase):

    def test_error(self):
        X = np.random.rand(10, 7, 350, 350)
        X = da.from_array(X, chunks=(1, 1, 350, 350))
        error_range = [[0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.9, 0.89], [0.4, 0.6], [0.4, 0.6], [0.8, 0.9]]

        with self.assertRaises(ValueError):
            to_zero = ToZero(range_to_zero=error_range)
            to_zero.fit_transform(X)

    def test_to_zero_default(self):
        # Create a fake dataset
        X = np.random.rand(10, 7, 350, 350)

        # Add some random Nans
        X[0, 0, 0, 0] = np.nan
        X[0, 0, 0, 1] = np.nan
        X[0, 0, 1, 0] = np.nan
        X[0, 0, 1, 1] = np.nan

        ranges = [[0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.89], [0.4, 0.6], [0.4, 0.6], [0.8, 0.9]]

        to_zero = ToZero(range_to_zero=ranges)

        # Convert to dask array
        X = da.from_array(X, chunks=(1, 1, 350, 350))

        # Fit and transform
        X = to_zero.fit_transform(X)

        # Loop through the ranges and assert that the values are zero
        for c, band in enumerate(ranges):
            low = band[0]
            high = band[1]
            # Use assert from unittest
            low_count = ((X[:, c, :, :] < low) & (X[:, c, :, :] > 0.00001)).sum().compute()
            high_count = (X[:, c, :, :] > high).sum().compute()
            self.assertEqual(low_count, 0)
            self.assertEqual(high_count, 0)

        # check that there are no nans in the data
        assert not da.isnan(X).any()

        # check that the shape is correct
        assert X.shape == (10, 7, 350, 350)

    def test_to_zero_dont_do_nan(self):
        # Create a fake dataset
        X = np.random.rand(10, 7, 350, 350)

        # Add some random Nans
        X[0, 0, 0, 0] = np.nan
        X[0, 0, 0, 1] = np.nan
        X[0, 0, 1, 0] = np.nan
        X[0, 0, 1, 1] = np.nan

        ranges = [[0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.89], [0.4, 0.6], [0.4, 0.6], [0.8, 0.9]]

        to_zero = ToZero(range_to_zero=ranges, nan_to_zero=False)

        # Convert to dask array
        X = da.from_array(X, chunks=(1, 1, 350, 350))

        # Fit and transform
        X = to_zero.fit_transform(X)

        # Loop through the ranges and assert that the values are zero
        for c, band in enumerate(ranges):
            low = band[0]
            high = band[1]
            # Use assert from unittest
            low_count = ((X[:, c, :, :] < low) & (X[:, c, :, :] > 0.00001)).sum().compute()
            high_count = (X[:, c, :, :] > high).sum().compute()
            self.assertEqual(low_count, 0)
            self.assertEqual(high_count, 0)

        # Check that nans are still there
        assert da.isnan(X).any()

        # check that the shape is correct
        assert X.shape == (10, 7, 350, 350)

    def test_to_zero_custom_value(self):
        # Create a fake dataset with ints
        X = np.random.randint(0, 10, size=(10, 7, 350, 350))

        ranges = [[0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.89], [0.4, 0.6], [0.4, 0.6], [0.8, 0.9]]

        to_zero = ToZero(range_to_zero=ranges, nan_to_zero=False, nan_value=10)

        # Convert to dask array
        X = da.from_array(X, chunks=(1, 1, 350, 350))

        # Fit and transform
        X = to_zero.fit_transform(X)

        # Loop through the ranges and assert that the values are zero
        for c, band in enumerate(ranges):
            low = band[0]
            high = band[1]
            # Use assert from unittest
            low_count = ((X[:, c, :, :] < low) & (X[:, c, :, :] > 0.00001)).sum().compute()
            high_count = (X[:, c, :, :] > high).sum().compute()
            self.assertEqual(low_count, 0)
            self.assertEqual(high_count, 0)

        # Check that there are no 10s anymore
        assert (X == 10).any() == False

        # check that the shape is correct
        assert X.shape == (10, 7, 350, 350)
