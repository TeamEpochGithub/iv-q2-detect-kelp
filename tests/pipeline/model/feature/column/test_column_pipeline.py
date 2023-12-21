from unittest import TestCase

import numpy as np
from src.pipeline.caching.column import CacheColumnBlock
from src.pipeline.model.feature.column.band_copy import BandCopy
from src.pipeline.model.feature.column.column import ColumnPipeline

from src.pipeline.model.feature.column.column_block import ColumnBlockPipeline


class Test(TestCase):
    def get_pipeline(self):
        band_copy_pipeline = BandCopy(1)

        cache = CacheColumnBlock(
            None, column=-1)
        column_block_pipeline = ColumnBlockPipeline(band_copy_pipeline, cache)
        column_pipeline = ColumnPipeline([column_block_pipeline])

        return column_pipeline

    def test_column_pipeline(self):
        cp = self.get_pipeline()
        self.assertNotEqual(cp, None)

        X = np.array([[1, 2], [3, 4]])
        X = cp.fit_transform(X)
        assert np.array_equal(X.compute(), np.array([[1, 2, 2], [3, 4, 4]]))


if __name__ == '__main__':
    test = Test()
    test.test_column_pipeline()
    print("Everything passed")
