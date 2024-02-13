"""Feature processing pipeline."""
from dataclasses import dataclass

from joblib import hash
from sklearn.pipeline import Pipeline

from src.logging_utils.logger import logger
from src.pipeline.caching.tif import CacheTIFBlock
from src.pipeline.model.feature.column.column import ColumnPipeline
from src.pipeline.model.feature.transformation.transformation import TransformationPipeline


@dataclass
class FeaturePipeline(Pipeline):
    """Feature pipeline class is used to create the feature pipeline.

    :param processed_path: path to the processed data
    :param transformation_pipeline: transformation pipeline
    :param column_pipeline: column pipeline
    """

    processed_path: str | None = None
    transformation_pipeline: TransformationPipeline | None = None
    column_pipeline: ColumnPipeline | None = None

    def __post_init__(self) -> None:
        """Post init function."""
        # Create hash
        self.load_from_cache = False
        if self.processed_path:
            self.transformation_hash = hash(self.transformation_pipeline)
            self.load_from_cache = True

        self.set_hash("")
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, Pipeline]]:
        """_get_steps function returns the steps for the pipeline.

        :return: list of steps
        """
        steps = []
        if self.transformation_pipeline:
            steps.append((str(self.transformation_pipeline), self.transformation_pipeline))
        else:
            logger.debug("No transformation steps were provided")

        if self.processed_path and self.load_from_cache:
            steps.append(("store_processed", CacheTIFBlock(self.processed_path + "/" + self.transformation_hash)))

        if self.column_pipeline:
            if self.processed_path and self.load_from_cache:
                self.column_pipeline.set_path(self.processed_path + "/" + self.transformation_hash)
            steps.append((str(self.column_pipeline), self.column_pipeline))
        else:
            logger.debug("No column steps were provided")

        # Add step to cache the processed data
        if self.processed_path and self.load_from_cache:
            steps.append(("store_pipeline", CacheTIFBlock(self.processed_path + "/" + self.prev_hash)))

        return steps

    def set_load_from_cache(self, *, load_from_cache: bool) -> None:
        """set_load_from_cache function sets the load from cache flag for the pipeline.

        :param load_from_cache: load from cache flag
        """
        self.load_from_cache = load_from_cache

        # Update the steps in the pipeline after changing the load from cache flag
        super().__init__(self._get_steps())

    def set_hash(self, prev_hash: str) -> str:
        """set_hash function sets the hash for the pipeline.

        :param prev_hash: previous hash
        :return: hash
        """
        feature_hash = prev_hash
        if self.transformation_pipeline:
            feature_hash = self.transformation_pipeline.set_hash(feature_hash)
        if self.column_pipeline:
            feature_hash = self.column_pipeline.set_hash(feature_hash)

        self.prev_hash = feature_hash

        return feature_hash
