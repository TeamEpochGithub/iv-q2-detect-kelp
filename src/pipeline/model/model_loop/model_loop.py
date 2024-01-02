"""Model loop pipeline."""
from sklearn.pipeline import Pipeline

from src.pipeline.model.model_loop.model_blocks.model_blocks import ModelBlocksPipeline
from src.pipeline.model.model_loop.pretrain.pretrain import PretrainPipeline


class ModelLoopPipeline:
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

    def get_pipeline(self, *, cache_model: bool = True) -> Pipeline | None:
        """Get the pipeline.

        :param cache_model: Whether to cache the model.
        :return: Pipeline object.
        """
        steps = []

        if self.pretrain_pipeline:
            steps.append(("pretrain_pipeline", self.pretrain_pipeline.get_pipeline()))
        if self.model_blocks_pipeline:
            steps.append(("model_blocks_pipeline", self.model_blocks_pipeline))

        if steps:
            return Pipeline(steps=steps, memory="tm" if cache_model else None)

        return None
