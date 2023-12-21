from dask_image.imread import imread

from distributed import Client
from src.logging_utils.section_separator import print_section_separator
from src.pipeline.model.feature.column.band_copy import BandCopy
from src.pipeline.model.feature.column.column import ColumnPipeline
from src.pipeline.model.feature.column.column_block import ColumnBlockPipeline
from src.pipeline.model.feature.feature import FeaturePipeline
from src.pipeline.model.feature.transformation.divider import Divider

from src.pipeline.model.feature.transformation.transformation import TransformationPipeline
from src.pipeline.model.model import ModelPipeline
from src.pipeline.model.model_loop.model_loop import ModelLoopPipeline
from src.pipeline.model.post_processing.post_processing import PostProcessingPipeline

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':

    # Coloured logs
    import coloredlogs
    coloredlogs.install()

    # Initialize dask client
    client = Client()

    # Print section separator
    print_section_separator("Q2 Detect Kelp States -- Submit")

    # Paths
    processed_path = "data/test"

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

    # Get model loop pipeline TODO
    mlp = ModelLoopPipeline(None, None)

    # Get post processing pipeline TODO
    ppp = PostProcessingPipeline()

    # Get model pipeline
    model_pipeline = ModelPipeline(fp, tp, mlp, ppp)

    # Read in the raw data
    raw_data_path = "data/raw/test_satellite"
    X = imread(f"{raw_data_path}/*.tif").transpose(0, 3, 1, 2)

    # Fit the model pipeline
    mp = model_pipeline.get_pipeline()

    # Transform the model pipeline
    # TODO remove and replace with predict, x = mp.predict(X)
    x = mp.fit_transform(X)
    print(x.shape)
