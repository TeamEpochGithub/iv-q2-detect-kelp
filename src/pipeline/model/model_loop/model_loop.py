"""Model loop pipeline."""

from dataclasses import dataclass

from sklearn.pipeline import Pipeline

from src.pipeline.model.model_loop.model_blocks.model_blocks import ModelBlocksPipeline
from src.pipeline.model.model_loop.pretrain.pretrain import PretrainPipeline


@dataclass
class ModelLoopPipeline(Pipeline):
    """Model loop pipeline.

    :param pretrain_pipeline: Pretrain pipeline.
    :param model_blocks_pipeline: Model blocks pipeline.
    """

    pretrain_pipeline: PretrainPipeline | None = None
    model_blocks_pipeline: ModelBlocksPipeline | None = None

    def __post_init__(self) -> None:
        """Post init function."""
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, Pipeline]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        steps = []

        if self.pretrain_pipeline:
            steps.append(("pretrain_pipeline_step", self.pretrain_pipeline))
        if self.model_blocks_pipeline:
            steps.append(("model_blocks_pipeline_step", self.model_blocks_pipeline))

        return steps

    def load_model(self, model_hash: str) -> None:
        """Load the model from the model hash.

        :param model_hash: The model hash
        """
        if self.model_blocks_pipeline:
            self.model_blocks_pipeline.load_model(model_hash)

    def load_scaler(self, scaler_hash: str) -> None:
        """Load the scaler from the scaler hash.

        :param scaler_hash: The scaler hash
        """
        if self.pretrain_pipeline:
            self.pretrain_pipeline.load_scaler(scaler_hash)

    def save_model(self, model_hash: str) -> None:
        """Save the model to the model hash.

        :param model_hash: The model hash
        """
        if self.model_blocks_pipeline:
            self.model_blocks_pipeline.save_model(model_hash)

    def save_scaler(self, scaler_hash: str) -> None:
        """Save the scaler to the scaler hash.

        :param scaler_hash: The scaler hash
        """
        if self.pretrain_pipeline:
            self.pretrain_pipeline.save_scaler(scaler_hash)
