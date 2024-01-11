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
        self.set_hash("")
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

    def set_hash(self, prev_hash: str = "") -> str:
        """Set the hash.

        :param prev_hash: Previous hash
        :return: Hash
        """
        model_loop_hash = prev_hash

        if self.pretrain_pipeline:
            model_loop_hash = self.pretrain_pipeline.set_hash(model_loop_hash)
        if self.model_blocks_pipeline:
            model_loop_hash = self.model_blocks_pipeline.set_hash(model_loop_hash)

        self.prev_hash = model_loop_hash

        return model_loop_hash
