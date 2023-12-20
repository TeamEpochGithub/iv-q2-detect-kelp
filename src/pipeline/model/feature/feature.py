# This file contains the feature processing pipeline, which is used to process the raw data into features that can be used by the model.
# It returns two things: the raw feature dataframe and the pipeline object.
# The pipeline object is used to transform the test data in the same way as the training data.

# Import libraries
from typing import Any
from distributed import Client
from src.logging_utils.logger import logger
from sklearn.pipeline import Pipeline
from src.pipeline.caching.tif import CacheTIFBlock
from src.pipeline.model.feature.column.column import ColumnPipeline
from src.pipeline.model.feature.error import FeaturePipelineError
from joblib import hash
from src.pipeline.model.feature.transformation.transformation import TransformationPipeline


class FeaturePipeline():
    """
    This class is used to create the feature pipeline.
    :param raw_data_path: path to the raw data
    :param processed_path: path to the processed data
    :param transformation_steps: list of transformation steps
    :param column_steps: list of column steps
    """

    def __init__(self, processed_path: str | None = None, transformation_pipeline: TransformationPipeline | None = None, column_pipeline: ColumnPipeline | None = None) -> None:
        """
        Initialize the class.

        :param processed_path: path to the processed data
        :param transformation_pipeline: the transformation pipeline
        :param column_pipeline: the column pipeline
        """

        # Set the parameters
        self.processed_path = processed_path
        self.transformation_pipeline = transformation_pipeline
        self.column_pipeline = column_pipeline

    def get_pipeline(self) -> Pipeline:
        """
        This function returns the feature pipeline.
        :return: Pipeline object
        """

        steps = []

        # Create the transformation pipeline
        transformation_hash = "raw"
        if self.transformation_pipeline:
            transformation_hash = hash(self.transformation_pipeline)
            transformation = (str(self.transformation_pipeline),
                              self.transformation_pipeline.get_pipeline())
            steps.append(transformation)
        else:
            logger.debug("No transformation steps were provided")

        # Full path
        path = self.processed_path + '/' + transformation_hash

        # Add the store pipeline
        if self.processed_path:
            store = ('store_processed', CacheTIFBlock(path))
            steps.append(store)

        # Create the column pipeline
        if self.column_pipeline:
            self.column_pipeline.set_path(path)
            column = (str(self.column_pipeline),
                      self.column_pipeline.get_pipeline())
            steps.append(column)
        else:
            logger.debug("No column steps were provided")

        mem = self.processed_path + '/' + transformation_hash + \
            '/pipeline_cache' if self.processed_path else None

        return Pipeline(steps=steps, memory=mem)


if __name__ == "__main__":
    # Example test
    raw_data_path = "data/raw/train_satellite"
    processed_path = "data/processed"
    features_path = "data/features"

    # Create the transformation pipeline
    from src.pipeline.model.feature.transformation.divider import Divider
    divider = Divider(2)

    transformation_pipeline = TransformationPipeline([divider])

    # Create the column pipeline
    from src.pipeline.model.feature.column.band_copy import BandCopy
    from src.pipeline.caching.column import CacheColumnBlock
    from src.pipeline.model.feature.column.column_block import ColumnBlockPipeline
    band_copy_pipeline = BandCopy(1)

    from src.pipeline.caching.column import CacheColumnBlock
    cache = CacheColumnBlock(
        "data/test", column=-1)
    column_block_pipeline = ColumnBlockPipeline(band_copy_pipeline, cache)
    column_pipeline = ColumnPipeline([column_block_pipeline])

    client = Client()
    import time
    orig_time = time.time()
    # Create the feature pipeline
    feature_pipeline = FeaturePipeline(processed_path=processed_path,
                                       transformation_pipeline=transformation_pipeline, column_pipeline=column_pipeline)
    pipeline = feature_pipeline.get_pipeline()

    # Parse the raw data
    orig_time = time.time()
    from dask_image.imread import imread
    x = imread(f"{raw_data_path}/*.tif").transpose(0,3,1,2)
    images = pipeline.fit_transform(x)
    print(time.time() - orig_time)

    # Display the first image
    image1 = images[0].compute()

    # Display all bands of the first image in multiple plots on the same figure
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 15))
    for i in range(8):
        plt.subplot(1, 8, i+1)
        plt.imshow(image1[i])
    plt.show()

    print(images.shape)
