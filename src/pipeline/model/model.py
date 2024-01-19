"""ModelPipeline is the class used to create the model pipeline."""

import sys
import time
from dataclasses import dataclass

from src.logging_utils.logger import logger

if sys.version_info < (3, 11):  # Self was added in Python 3.11
    pass
else:
    pass


import dask.array as da
from joblib import hash
from sklearn.pipeline import Pipeline

from src.pipeline.model.feature.feature import FeaturePipeline
from src.pipeline.model.model_loop.model_loop import ModelLoopPipeline
from src.pipeline.model.post_processing.post_processing import PostProcessingPipeline


@dataclass
class ModelPipeline(Pipeline):
    """ModelPipeline is the class used to create the model pipeline.

    :param feature_pipeline: The feature pipeline
    :param target_pipeline: The target pipeline
    :param model_loop_pipeline: The model loop pipeline
    :param post_processing_pipeline: The post processing pipeline
    """

    feature_pipeline: FeaturePipeline | None = None
    target_pipeline: FeaturePipeline | None = None
    model_loop_pipeline: ModelLoopPipeline | None = None
    post_processing_pipeline: PostProcessingPipeline | None = None

    def __post_init__(self) -> None:
        """Post init function."""
        self.set_hash("")
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, Pipeline]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        steps = []

        if self.feature_pipeline:
            steps.append(("feature_pipeline_step", self.feature_pipeline))
        if self.model_loop_pipeline:
            steps.append(("model_loop_pipeline_step", self.model_loop_pipeline))
        if self.post_processing_pipeline:
            steps.append(("post_processing_pipeline_step", self.post_processing_pipeline))

        return steps

    def fit_transform(self, X: da.Array, y: da.Array | None = None, **fit_params: str) -> da.Array:
        """Fit and transform the data.

        :param X: Data to fit and transform
        :param y: Target data
        :param fit_params: Fit parameters
        :return: Fitted and transformed data
        """
        start_time = time.time()
        X = super().fit_transform(X, y, **fit_params)
        logger.info(f"Fitted model pipeline in {time.time() - start_time} seconds")
        return X

    def get_target_pipeline(self) -> FeaturePipeline | None:
        """Get the target pipeline.

        :return: The target pipeline
        """
        if self.target_pipeline:
            return self.target_pipeline

        return None

    def set_hash(self, prev_hash: str) -> str:
        """Set the hashes of the pipelines."""
        model_hash = prev_hash

        if self.feature_pipeline:
            model_hash = self.feature_pipeline.set_hash(model_hash)
        if self.target_pipeline:
            model_hash = hash(self.target_pipeline.set_hash(prev_hash) + model_hash)
        if self.model_loop_pipeline:
            model_hash = self.model_loop_pipeline.set_hash(model_hash)
        if self.post_processing_pipeline:
            model_hash = self.post_processing_pipeline.set_hash(model_hash)

        self.prev_hash = model_hash

        return model_hash
