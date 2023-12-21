# This file contains the feature processing pipeline, which is used to process the raw data into features that can be used by the model.
# It returns two things: the raw feature dataframe and the pipeline object.
# The pipeline object is used to transform the test data in the same way as the training data.

# Import libraries
from src.logging_utils.logger import logger
from sklearn.pipeline import Pipeline
from src.pipeline.caching.tif import CacheTIFBlock
from src.pipeline.model.feature.column.column import ColumnPipeline
from joblib import hash
from src.pipeline.model.feature.transformation.transformation import TransformationPipeline


class FeaturePipeline():
    """This class is used to create the feature pipeline.

    :param processed_path: path to the processed data
    :param transformation_pipeline: transformation pipeline
    :param column_pipeline: column pipeline
    """

    def __init__(
            self,
            processed_path: str | None = None,
            transformation_pipeline: TransformationPipeline | None = None,
            column_pipeline: ColumnPipeline | None = None
    ) -> None:
        """
        Initialize the class.

        :param processed_path: path to the processed data
        :param transformation_pipeline: transformation pipeline
        :param column_pipeline: column pipeline
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
        path = None
        if self.processed_path:
            path = self.processed_path + '/' + transformation_hash
            store = ('store_processed', CacheTIFBlock(path))
            steps.append(store)

        # Create the column pipeline
        if self.column_pipeline:
            if path:
                self.column_pipeline.set_path(path)
            column = (str(self.column_pipeline),
                      self.column_pipeline.get_pipeline())
            steps.append(column)
        else:
            logger.debug("No column steps were provided")

        mem = self.processed_path + '/' + transformation_hash + \
            '/pipeline_cache' if self.processed_path else None

        return Pipeline(steps=steps, memory=mem)

    def __str__(self) -> str:
        """String representation of the class.

        :return: String representation of the class
        """
        return "FeaturePipeline"

    def __repr__(self) -> str:
        """String representation of the class.

        :return: String representation of the class
        """
        return f"FeaturePipeline(processed_path='{self.processed_path}',transformation_pipeline={self.transformation_pipeline},column_pipeline={self.column_pipeline})"
