from unittest import TestCase

import dask.array as da
import numpy as np

from src.pipeline.model.feature.transformation.clip import Clip


class TestClip(TestCase):

    def test_error(self):
        X = np.random.rand(10, 7, 350, 350)
        X = da.from_array(X, chunks=(1, 1, 350, 350))
        error_range = [[0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.9, 0.89], [0.4, 0.6], [0.4, 0.6], [0.8, 0.9]]

        with self.assertRaises(ValueError):
            to_zero = Clip(feature_ranges=error_range)
            to_zero.fit_transform(X)

    def test_stretch(self):
        # Create a fake dataset
        X = np.random.rand(10, 7, 350, 350)

        stretch_ranges = [[0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]

        # Create the transformer
        s = Clip(feature_ranges=stretch_ranges)

        X = da.from_array(X, chunks=(1, 1, 350, 350))

        # Cit and transform
        X = s.fit_transform(X)

        # Loop through the ranges and assert that the values are zero
        for c, band in enumerate(stretch_ranges):
            low = band[0]
            high = band[1]
            # Use assert from unittest
            self.assertTrue((X[:, c, :, :] >= low).all())
            self.assertTrue((X[:, c, :, :] <= high).all())
