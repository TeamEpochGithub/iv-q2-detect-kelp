"""Model loop pipeline."""
from sklearn.pipeline import Pipeline

from src.pipeline.model.model_loop.model_blocks.model_blocks import ModelBlocksPipeline
from src.pipeline.model.model_loop.pretrain.pretrain import PretrainPipeline


class ModelLoopPipeline(Pipeline):
    """Model loop pipeline.

    :param pretrain_pipeline: Pretrain pipeline.
    :param model_blocks_pipeline: Model blocks pipeline.
    """

    def __init__(self, pretrain_pipeline: PretrainPipeline | None, model_blocks_pipeline: ModelBlocksPipeline | None) -> None:
        """Model loop pipeline.

        :param pretrain_pipeline: Pretrain pipeline.
        :param model_blocks_pipeline: Model blocks pipeline.
        """
        self.pretrain_pipeline = pretrain_pipeline
        self.model_blocks_pipeline = model_blocks_pipeline
        super().__init__(self._get_steps())

    def _get_steps(self):
        """Get the pipeline steps.

        :return: list of steps
        """
        steps = []

        if self.pretrain_pipeline:
            steps.append(("pretrain_pipeline_step", self.pretrain_pipeline))
        if self.model_blocks_pipeline:
            steps.append(("model_blocks_pipeline_step", self.model_blocks_pipeline))

        return steps

