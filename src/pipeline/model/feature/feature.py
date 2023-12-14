# This file contains the feature processing pipeline, which is used to process the raw data into features that can be used by the model.
# It returns two things: the raw feature dataframe and the pipeline object.
# The pipeline object is used to transform the test data in the same way as the training data.

# Import libraries
from distributed import Client
from src.logging_utils.logger import logger
import os
from sklearn.pipeline import Pipeline
from src.pipeline.model.feature.column.get_columns import get_columns
from src.pipeline.model.feature.transformation.get_transformations import get_transformations
from src.pipeline.parse.tif import ParseTIFPipeline
from src.pipeline.store.tif import StoreTIFPipeline


class FeaturePipeline():
    """
    This class is used to create the feature pipeline.
    :param raw_data_path: path to the raw data
    :param processed_path: path to the processed data
    :param features_path: path to the features
    :param transformation_steps: list of transformation steps
    :param column_steps: list of column steps
    """

    def __init__(self, raw_data_path: str, processed_path=None, features_path: str = None, transformation_steps: list[dict] = None, column_steps: list[dict] = None):

        if not raw_data_path:
            logger.error("raw_data_path is required")
            raise FeaturePipelineError("raw_data_path is required")

        # Set paths to self
        self.raw_data_path = raw_data_path
        self.processed_path = processed_path
        self.features_path = features_path
        self.transformation_steps = transformation_steps
        self.column_steps = column_steps
        self.steps = steps

    def get_pipeline(self) -> Pipeline:
        """
        This function returns the feature pipeline.
        :return: Pipeline object
        """

        steps = []

        # Get the raw data paths
        raw_data_paths = self.get_raw_data_paths()

        # Create the raw data parser
        parser = ('raw_data_parser', ParseTIFPipeline(raw_data_paths))
        steps.append(parser)

        # Create the transformation pipeline
        if self.transformation_steps:
            transformation_pipeline = get_transformations(
                self.transformation_steps)
            if transformation_pipeline:
                transformations = ('transformations', transformation_pipeline)
                steps.append(transformations)
        else:
            logger.info("No transformation steps were provided")

        # Create processed paths
        processed_paths = self.get_processed_data_paths()

        # Add the store pipeline
        store = ('store_processed', StoreTIFPipeline(processed_paths))
        steps.append(store)

        if self.column_steps:
            column_pipeline = get_columns(self.column_steps)
            if column_pipeline:
                columns = ('columns', column_pipeline)
                steps.append(columns)

        pipeline = Pipeline(steps=steps)

        return pipeline

    def get_raw_data_paths(self) -> list[str]:
        """
        This function returns the raw data paths.
        :return: list of raw data paths
        """
        # Get the raw data paths
        raw_data_paths = os.listdir(self.raw_data_path)

        # Iterate over the raw data paths and create the full path
        for i, raw_data_path in enumerate(raw_data_paths):
            raw_data_paths[i] = os.path.join(self.raw_data_path, raw_data_path)

        # Sort the raw data paths
        raw_data_paths.sort()

        return raw_data_paths

    def get_processed_data_paths(self) -> list[str]:
        """
        This function returns the processed data paths.
        :return: list of processed data paths
        """
        # Get the names of each file
        names = os.listdir(self.raw_data_path)

        # If processed path does not exist, create it
        if not os.path.exists(self.processed_path):
            os.makedirs(self.processed_path)

        # Create the processed data paths
        processed_data_paths = names

        # Iterate over the processed data paths and create the full path
        for i, name in enumerate(names):
            processed_data_paths[i] = os.path.join(self.processed_path, name)

        # Sort the processed data paths
        processed_data_paths.sort()

        return processed_data_paths


if __name__ == "__main__":
    # Example test
    raw_data_path = "data/raw/train_satellite"
    processed_path = "data/processed/test"
    features_path = "data/features"
    steps = []

    client = Client()
    import time
    orig_time = time.time()
    # Create the feature pipeline
    feature_pipeline = FeaturePipeline(
        raw_data_path, processed_path, features_path, steps)

    from sklearn.pipeline import Pipeline
    pipeline = feature_pipeline.get_pipeline()

    # Parse the raw data
    orig_time = time.time()
    images = pipeline.fit_transform(None)
    print(time.time() - orig_time)

    # images = feature_pipeline.parse_raw()
    print(time.time() - orig_time)

    print(images.shape)


class FeaturePipelineError(Exception):
    # Define the error message
    def __init__(self, message):
        self.message = message

    # Define the string representation
    def __str__(self):
        return self.message

    # Define the representation
    def __repr__(self):
        return self.message

    # Define the error name
    def __name__(self):
        return "FeaturePipelineError"
