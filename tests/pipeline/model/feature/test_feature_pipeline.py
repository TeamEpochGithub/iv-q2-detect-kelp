import time
from unittest import TestCase

from distributed import Client
import numpy as np
from src.pipeline.caching.column import CacheColumnBlock
from src.pipeline.model.feature.column.band_copy import BandCopy
from src.pipeline.model.feature.column.column import ColumnPipeline
from src.pipeline.model.feature.column.column_block import ColumnBlockPipeline

from src.pipeline.model.feature.feature import FeaturePipeline
from src.pipeline.model.feature.transformation.transformation import TransformationPipeline

import dask.array as da


class Test(TestCase):

    def get_pipeline(self):

        # Example test
        processed_path = None

        # Create the transformation pipeline
        from src.pipeline.model.feature.transformation.divider import Divider
        divider = Divider(2)

        transformation_pipeline = TransformationPipeline([divider])

        # Create the column pipeline
        band_copy_pipeline = BandCopy(1)

        cache = CacheColumnBlock(
            "../data/test", column=-1)
        column_block_pipeline = ColumnBlockPipeline(band_copy_pipeline, cache)
        column_pipeline = ColumnPipeline([column_block_pipeline])

        orig_time = time.time()
        # Create the feature pipeline
        feature_pipeline = FeaturePipeline(processed_path=processed_path,
                                           transformation_pipeline=transformation_pipeline, column_pipeline=column_pipeline)
        return feature_pipeline

    def test_repr(self):
        self.assertEqual(FeaturePipeline().__repr__(
        ), f"FeaturePipeline(processed_path=None, transformation_pipeline=None, column_pipeline=None)")

    def test_initialization(self):
        self.assertEqual(FeaturePipeline().__init__(), None)

    def test_get_pipeline(self):
        pipeline = self.get_pipeline()
        self.assertNotEqual(pipeline, None)

    def test_feature_pipeline(self):
        fp = self.get_pipeline()

        # Parse the raw data
        x = da.array([[1, 2], [3, 4]])
        x_fit = fp.fit_transform(x)
        np.testing.assert_array_equal(x_fit, np.array([[0.5, 1, 1], [1.5, 2, 2]]))


if __name__ == '__main__':
    Test().test_feature_pipeline()
    print("Everything passed")
