from unittest import TestCase

from src.pipeline.model.feature.column.offset import Offset


class TestOffset(TestCase):

    def test_offset(self):

        # test the Offset class
        import numpy as np
        import dask.array as da

        # create a fake dataset
        X = np.random.rand(10, 7, 350, 350)

        # set elevation to 1 for the first image
        X[0, 6] = 1

        # convert to dask array
        X = da.from_array(X, chunks=(1, 1, 350, 350))

        # create the transformer
        offset = Offset(band=0, elevation=6)

        # fit and transform
        X = offset.fit_transform(X)

        # check that there are no nans in the data
        assert not da.isnan(X).any()

        # check that the shape is correct
        assert X.shape == (10, 8, 350, 350)