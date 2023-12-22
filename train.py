"""Train.py is the main script for training the model and will take in the raw data and output a trained model."""
import time
from dask_image.imread import imread

from distributed import Client
import numpy as np
from sklearn import set_config
from sklearn.model_selection import train_test_split
from src.logging_utils.logger import logger
from src.pipeline.model.feature.column.band_copy import BandCopy
from src.pipeline.model.feature.column.column import ColumnPipeline
from src.pipeline.model.feature.column.column_block import ColumnBlockPipeline
from src.pipeline.model.feature.feature import FeaturePipeline
from src.pipeline.model.feature.transformation.divider import Divider

from src.pipeline.model.feature.transformation.transformation import TransformationPipeline
from src.pipeline.model.model import ModelPipeline
from src.pipeline.model.model_loop.model_loop import ModelLoopPipeline
from src.pipeline.model.post_processing.post_processing import PostProcessingPipeline

from src.logging_utils.section_separator import print_section_separator

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':

    # Coloured logs
    import coloredlogs
    coloredlogs.install()

    # Print section separator
    print_section_separator("Q2 Detect Kelp States -- Training")

    # Initialize dask client
    client = Client()

    # Log client information
    logger.info(f"Client: {client}")

    # Setup the pipeline
    logger.info("Setting up the pipeline")
    orig_time = time.time()

    ###############################
    # TODO: Use config to create the classes

    processed_path = "data/processed"
    raw_data_path = "data/raw/train_satellite"
    raw_target_path = "data/raw/train_kelp"
    split = 0.2

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

    # Create the feature pipeline
    feature_pipeline = FeaturePipeline(processed_path=processed_path,
                                       transformation_pipeline=transformation_pipeline, column_pipeline=column_pipeline)

    # Get target pipeline TODO
    tp = None
    raw_target_path = 'data/raw/train_kelp'     # TODO remove
    y = imread(f"{raw_target_path}/*.tif")  # TODO remove

    # Get model loop pipeline TODO
    mlp = ModelLoopPipeline(None, None)

    # Get post processing pipeline TODO
    ppp = PostProcessingPipeline()

    # Get model pipeline
    model_pipeline_object = ModelPipeline(feature_pipeline, tp, mlp, ppp)

    ################################

    model_pipeline = model_pipeline_object.get_pipeline()
    logger.info(f"Pipeline setup time: {time.time() - orig_time} seconds")
    logger.debug(f"Pipeline: {model_pipeline}")

    # Save the pipeline to html file
    set_config(display='diagram')

    # Get the HTML representation of the pipeline
    pipeline_html = model_pipeline._repr_html_()

    # Write the HTML to a file
    with open('logging/pipeline.html', 'w', encoding='utf-8') as f:
        f.write(pipeline_html)

    # Read in the raw data
    logger.info("Reading in the raw feature and target data")
    X = imread(f"{raw_data_path}/*.tif").transpose(0, 3, 1, 2)
    y = imread(f"{raw_target_path}/*.tif")
    logger.info(f"Raw data shape: {X.shape}")
    logger.info(f"Raw target shape: {y.shape}")

    # Create an array of indices # TODO split comes from config file
    logger.info("Splitting the data into train and test sets")

    # Suppress logger messages while getting the indices to avoid clutter in the log file
    logger.setLevel("ERROR")
    x = feature_pipeline.fit_transform(X)
    logger.setLevel("INFO")
    indices = np.arange(x.shape[0])

    # Split indices into train and test
    train_indices, test_indices = train_test_split(indices, test_size=split)

    logger.debug(f"Train indices: {train_indices}")
    logger.debug(f"Test indices: {test_indices}")

    # fit_args: dict[tuple[str, Any]] = {}

    # Transform the model pipeline
    x = model_pipeline.fit_transform(X, y)
    print(x.shape)
