"""ModelPipeline is the class used to create the model pipeline."""

from dataclasses import dataclass

from sklearn.pipeline import Pipeline

from src.pipeline.model.feature.feature import FeaturePipeline
from src.pipeline.model.model_loop.model_loop import ModelLoopPipeline
from src.pipeline.model.post_processing.post_processing import PostProcessingPipeline
from src.pipeline.model.target.target import TargetPipeline


@dataclass
class ModelPipeline(Pipeline):
    """ModelPipeline is the class used to create the model pipeline.

    :param feature_pipeline: The feature pipeline
    :param target_pipeline: The target pipeline
    :param model_loop_pipeline: The model loop pipeline
    :param post_processing_pipeline: The post processing pipeline
    """

    feature_pipeline: FeaturePipeline | None = None
    target_pipeline: TargetPipeline | None = None
    model_loop_pipeline: ModelLoopPipeline | None = None
    post_processing_pipeline: PostProcessingPipeline | None = None

    def __post_init__(self) -> None:
        """Post init function."""
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, Pipeline]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        steps = []

        if self.feature_pipeline:
            steps.append(("feature_pipeline_step", self.feature_pipeline))
        if self.target_pipeline:
            steps.append(("target_pipeline_step", self.target_pipeline))
        if self.model_loop_pipeline:
            steps.append(("model_loop_pipeline_step", self.model_loop_pipeline))
        if self.post_processing_pipeline:
            steps.append(("post_processing_pipeline_step", self.post_processing_pipeline))

        return steps

    def load_models(self, model_hashes: list[str]) -> None:
        """Load the models from the model hashes.

        :param model_hashes: The model hashes
        """
        # self.model_loop_pipeline.load_models(model_hashes)

    def load_scalers(self, scaler_hashes: list[str | None]) -> None:
        """Load the scalers from the scaler hashes.

        :param scaler_hashes: The scaler hashes
        """
        # self.model_loop_pipeline.load_scalers(scaler_hashes)
