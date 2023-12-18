# This file contains the feature processing pipeline, which is used to process the raw data into features that can be used by the model.
# It returns two things: the raw feature dataframe and the pipeline object.
# The pipeline object is used to transform the test data in the same way as the training data.

# Import libraries
from typing import Any
from distributed import Client
from src.logging_utils.logger import logger
from sklearn.pipeline import Pipeline
from src.pipeline.caching.tif import CacheTIFPipeline
from src.pipeline.model.feature.column.get_columns import get_columns
from src.pipeline.model.feature.error import FeaturePipelineError
from src.pipeline.model.feature.transformation.get_transformations import get_transformations
from joblib import hash


class FeaturePipeline():
    """
    This class is used to create the feature pipeline.
    :param raw_data_path: path to the raw data
    :param processed_path: path to the processed data
    :param transformation_steps: list of transformation steps
    :param column_steps: list of column steps
    """

    def __init__(self, raw_data_path: str, processed_path: str | None = None, transformation_steps: list[dict[str, Any]] = [], column_steps: list[dict[str, Any]] = []):

        if not raw_data_path:
            logger.error("raw_data_path is required")
            raise FeaturePipelineError("raw_data_path is required")

        # Set paths to self
        self.raw_data_path = raw_data_path
        self.processed_path = processed_path
        self.transformation_steps = transformation_steps
        self.column_steps = column_steps

    def get_pipeline(self) -> Pipeline:
        """
        This function returns the feature pipeline.
        :return: Pipeline object
        """

        steps = []

        # Create the raw data parser
        parser = ('raw_data_parser', CacheTIFPipeline(self.raw_data_path))
        steps.append(parser)

        # Create the transformation pipeline
        transformation_hash = "raw"
        if self.transformation_steps:
            transformation_pipeline = get_transformations(
                self.transformation_steps)
            if transformation_pipeline:
                transformations = ('transformations', transformation_pipeline)
                steps.append(transformations)

                # Get the hash of the transformation pipeline
                transformation_hash = hash(transformation_pipeline)
                logger.debug(
                    f"Transformation pipeline hash: {transformation_hash}")
        else:
            logger.info("No transformation steps were provided")

        # Add the store pipeline
        store = ('store_processed', CacheTIFPipeline(
            self.processed_path + '/' + transformation_hash))
        steps.append(store)

        if self.column_steps:
            column_pipeline = get_columns(
                self.column_steps, self.processed_path + '/' + transformation_hash)
            if column_pipeline:
                columns = ('columns', column_pipeline)
                steps.append(columns)
        if not self.processed_path:
            logger.info(
                "No processed path was provided, returning pipeline without caching")
            return Pipeline(steps)
        else:
            pipeline = Pipeline(steps=steps, memory=self.processed_path +
                                '/' + transformation_hash + '/pipeline_cache')

        return pipeline


if __name__ == "__main__":
    # Example test
    raw_data_path = "data/raw/train_satellite"
    processed_path = "data/processed"
    features_path = "data/features"
    transform_steps = [{'type': 'divider', 'divider': 65500}]
    columns = [{'type': 'band_copy', 'band': 0},
               {'type': 'band_copy', 'band': 2}]

    client = Client()
    import time
    orig_time = time.time()
    # Create the feature pipeline
    feature_pipeline = FeaturePipeline(
        raw_data_path, processed_path, transformation_steps=transform_steps, column_steps=columns)
    pipeline = feature_pipeline.get_pipeline()

    # Parse the raw data
    orig_time = time.time()
    images = pipeline.fit_transform(None)
    print(time.time() - orig_time)

    # Display the first image
    image1 = images[0].compute()

    # Display all bands of the first image in multiple plots on the same figure
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 15))
    for i in range(9):
        plt.subplot(1, 9, i+1)
        plt.imshow(image1[i])
    plt.show()

    print(images.shape)
