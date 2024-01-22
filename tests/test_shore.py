from unittest import TestCase

from dask import array as da

from src.pipeline.model.feature.column.shore import Shore


class TestShore(TestCase):
    def test_distance(self):

        shore = Shore(mode="distance")
        x = da.zeros((5, 7, 350, 350))
        x[0, 6, 0:50, 50:150] = 1
        out = shore.transform(x)
        self.assertEqual(out.shape, (5, 8, 350, 350))

    def test_closeness(self):
        shore = Shore(mode="closeness")
        x = da.zeros((5, 7, 350, 350))
        x[0, 6, 0:50, 50:150] = 1
        out = shore.transform(x)
        self.assertEqual(out.shape, (5, 8, 350, 350))

    def test_binary(self):
        shore = Shore(mode="binary")
        x = da.zeros((5, 7, 350, 350))
        x[0, 6, 0:50, 50:150] = 1
        out = shore.transform(x)
        self.assertEqual(out.shape, (5, 8, 350, 350))
