"""Feature processing pipeline."""
from joblib import hash
from sklearn.pipeline import Pipeline

from src.logging_utils.logger import logger
from src.pipeline.caching.tif import CacheTIFBlock
from src.pipeline.model.feature.column.column import ColumnPipeline
from src.pipeline.model.feature.transformation.transformation import TransformationPipeline


class FeaturePipeline(Pipeline):
    """Feature pipeline class is used to create the feature pipeline.

    :param processed_path: path to the processed data
    :param transformation_pipeline: transformation pipeline
    :param column_pipeline: column pipeline
    """

    def __init__(
        self,
        processed_path: str | None = None,
        transformation_pipeline: TransformationPipeline | None = None,
        column_pipeline: ColumnPipeline | None = None,
    ) -> None:
        """Initialize the class.

        :param processed_path: path to the processed data
        :param transformation_pipeline: transformation pipeline
        :param column_pipeline: column pipeline
        :param is_train: whether the pipeline is for training or not
        """
        # Set the parameters
        self.processed_path = processed_path
        self.transformation_pipeline = transformation_pipeline
        self.column_pipeline = column_pipeline
        
        # Create hash
        if self.processed_path:
            self.transformation_hash = hash(self.transformation_pipeline)

        super().__init__(self._get_steps(), memory=self._get_memory())

    def _get_steps(self) -> list[tuple[str, Pipeline]]:
        """_get_steps function returns the steps for the pipeline.

        :return: list of steps
        """
        steps = []
        if self.transformation_pipeline:
            steps.append((str(self.transformation_pipeline), self.transformation_pipeline))
        else:
            logger.debug("No transformation steps were provided")

        if self.processed_path:
            steps.append(("store_processed", CacheTIFBlock(self.processed_path + "/" + self.transformation_hash)))

        if self.column_pipeline:
            if self.processed_path:
                self.column_pipeline.set_path(self.processed_path + "/" + self.transformation_hash)
            steps.append((str(self.column_pipeline), self.column_pipeline))
        else:
            logger.debug("No column steps were provided")

        return steps

    def _get_memory(self) -> str | None:
        """_get_memory function returns the memory location for the pipeline.

        :return: memory location
        """
        if self.processed_path:
            return self.processed_path + "/" + self.transformation_hash + "/pipeline_cache"
        return None

    def __str__(self) -> str:
        """__str__ returns string representation of the class.

        :return: String representation of the class
        """
        return "FeaturePipeline"

    def __repr__(self) -> str:
        """__repr__ returns the full representation of the class.

        :return: Full representation of the class
        """
        return (
            f"FeaturePipeline("
            f"processed_path={self.processed_path}, "
            f"transformation_pipeline={self.transformation_pipeline}, "
            f"column_pipeline={self.column_pipeline})"
        )
