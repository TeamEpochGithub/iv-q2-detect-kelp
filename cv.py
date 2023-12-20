"""Run cross-validation on a model or ensemble."""
from dask_image.imread import imread

from distributed import Client
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from src.pipeline.model.feature.column.band_copy import BandCopy
from src.pipeline.model.feature.column.column import ColumnPipeline
from src.pipeline.model.feature.column.column_block import ColumnBlockPipeline
from src.pipeline.model.feature.feature import FeaturePipeline
from src.pipeline.model.feature.transformation.divider import Divider

from src.pipeline.model.feature.transformation.transformation import TransformationPipeline
from src.pipeline.model.model import ModelPipeline
from src.pipeline.model.model_loop.model_loop import ModelLoopPipeline
from src.pipeline.model.post_processing.post_processing import PostProcessingPipeline


if __name__ == '__main__':

    # Initialize dask client
    client = Client()

    # Paths
    processed_path = "data/processed"

    # Create the transformation pipeline
    divider = Divider(2)

    transformation_pipeline = TransformationPipeline([divider])

    # Create the column pipeline
    band_copy_pipeline = BandCopy(1)

    from src.pipeline.caching.column import CacheColumnBlock
    cache = CacheColumnBlock(
        "data/test", column=-1)
    column_block_pipeline = ColumnBlockPipeline(band_copy_pipeline, cache)
    column_pipeline = ColumnPipeline([column_block_pipeline])

    import time
    orig_time = time.time()
    # Create the feature pipeline
    fp = FeaturePipeline(processed_path=processed_path,
                                       transformation_pipeline=transformation_pipeline, column_pipeline=column_pipeline)
    feature_pipeline = fp.get_pipeline()

    # Get target pipeline TODO
    tp = None
    raw_target_path = 'data/raw/train_kelp'     # TODO remove
    y = imread(f"{raw_target_path}/*.tif")  # TODO remove

    # Get post processing pipeline TODO
    ppp = PostProcessingPipeline()

    # Read in the raw data
    raw_data_path = "data/raw/train_satellite"
    raw_target_path = "data/raw/train_kelp"
    X = imread(f"{raw_data_path}/*.tif").transpose(0,3,1,2)
    y = imread(f"{raw_target_path}/*.tif")

    # Create an array of indices # TODO split comes from config file
    split = 0.2
    x = feature_pipeline.fit_transform(X)
    indices = np.arange(x.shape[0])

    # Create a KFold object
    kf = StratifiedKFold(n_splits=5)

    # Split indices into train and test for each fold, create a stratification key from y
    stratification_key = y.compute().reshape(y.shape[0], -1).max(axis=1)
    for train_indices, test_indices in kf.split(X, stratification_key):
        print(f"Train indices: {train_indices}")
        print(f"Test indices: {test_indices}")

        fit_args = {}

        # Get model loop pipeline TODO
        mlp = ModelLoopPipeline(None, None)

        # Get model pipeline
        model_pipeline = ModelPipeline(fp, tp, mlp, ppp)

        mp = model_pipeline.get_pipeline()
        x = mp.fit_transform(X, y)
        print(x.shape)
