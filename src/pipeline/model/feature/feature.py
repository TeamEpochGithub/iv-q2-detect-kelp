# This file contains the feature processing pipeline, which is used to process the raw data into features that can be used by the model.
# It returns two things: the raw feature dataframe and the pipeline object.
# The pipeline object is used to transform the test data in the same way as the training data.

# Import libraries
from tqdm import tqdm
from distributed import Client
import rasterio
from src.logging_utils.logger import logger
import os
import dask.array as da
import dask
import dask.diagnostics as diag
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from src.pipeline.parse.tif import ParseTIFPipeline

# Define the feature pipeline
class FeaturePipeline():
    def __init__(self, raw_data_path: str, processed_path=None, features_path: str = None, transformation_steps: list = None, column_steps: list = None):

        if not raw_data_path:
            logger.error("raw_data_path is required")
            raise FeaturePipelineError("raw_data_path is required")

        # Set paths to self
        self.raw_data_path = raw_data_path
        self.processed_path = processed_path
        self.features_path = features_path
        self.steps = steps

    def get_pipeline(self) -> Pipeline:
        """
        This function returns the feature pipeline.
        :return: Pipeline object
        """

        steps = []
        parser = ('raw_data_parser', ParseTIFPipeline(self.raw_data_path))
        steps.append(parser)

        transformations = ('transformations', self.transformation_pipeline())
        #steps.append(transformations)

        columns = ('columns', self.column_pipeline())
        #steps.append(columns)

        pipeline = Pipeline(steps=steps)

        return pipeline

    def transformation_pipeline(self):
        """
        This function creates the transformation pipeline.
        :return: None
        """
        # TODO: Create the transformation pipeline
        pass

    def column_pipeline(self):
        """
        This function creates the column pipeline.
        :return: None
        """
        # TODO: Create the column pipeline
        pass


if __name__ == "__main__":
    # Example test
    raw_data_path = "data/raw/train_satellite"
    processed_path = "data/processed"
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

    #images = feature_pipeline.parse_raw()

    images = images[:1000].compute()
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
