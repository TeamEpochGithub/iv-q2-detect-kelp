"""ModelPipeline is the class used to create the model pipeline."""
from typing import Any

from sklearn.pipeline import Pipeline

from src.pipeline.model.feature.feature import FeaturePipeline
from src.pipeline.model.model_loop.model_loop import ModelLoopPipeline
from src.pipeline.model.post_processing.post_processing import PostProcessingPipeline
from src.pipeline.model.target.target import TargetPipeline


class ModelPipeline:
    """ModelPipeline is the class used to create the model pipeline.

    :param feature_pipeline: The feature pipeline
    :param target_pipeline: The target pipeline
    :param model_loop_pipeline: The model loop pipeline
    :param post_processing_pipeline: The post processing pipeline
    """

    def __init__(
        self,
        feature_pipeline: FeaturePipeline | None = None,
        target_pipeline: TargetPipeline | None = None,
        model_loop_pipeline: ModelLoopPipeline | None = None,
        post_processing_pipeline: PostProcessingPipeline | None = None,
    ) -> None:
        """Initialize the class.

        :param feature_pipeline: The feature pipeline
        :param target_pipeline: The target pipeline
        :param model_loop_pipeline: The model loop pipeline
        :param post_processing_pipeline: The post processing pipeline
        """
        self.feature_pipeline = feature_pipeline
        self.target_pipeline = target_pipeline
        self.model_loop_pipeline = model_loop_pipeline
        self.post_processing_pipeline = post_processing_pipeline

    def get_pipeline(self) -> Pipeline:
        """Get the pipeline.

        :return: Pipeline object
        """
        steps: list[tuple[str, Any]] = []

        if self.feature_pipeline:
            steps.append((str(self.feature_pipeline), self.feature_pipeline))
        if self.target_pipeline:
            steps.append(("target_pipeline", self.target_pipeline.get_pipeline()))
        if self.model_loop_pipeline:
            steps.append(("model_loop_pipeline", self.model_loop_pipeline.get_pipeline()))
        if self.post_processing_pipeline:
            steps.append(("post_processing_pipeline", self.post_processing_pipeline.get_pipeline()))

        return Pipeline(steps)

    def __str__(self) -> str:
        """__str__ returns string representation of the class.

        :return: String representation of the class
        """
        return "ModelPipeline"

    def __repr__(self) -> str:
        """Representation of the class.

        :return: Representation of the class
        """
        return (
            "ModelPipeline("
            f"feature_pipeline={self.feature_pipeline.__repr__()},"
            f"target_pipeline={self.target_pipeline.__repr__()},"
            f"model_loop_pipeline={self.model_loop_pipeline.__repr__()},"
            f"post_processing_pipeline={self.post_processing_pipeline.__repr__()})"
        )
