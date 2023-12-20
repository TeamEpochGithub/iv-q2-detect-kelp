from sklearn.pipeline import Pipeline

from src.pipeline.model.model_loop.model_blocks.model_fit_block import ModelBlock


class ModelBlocksPipeline():
    """
    This class is used to create the model blocks pipeline.
    :param model_blocks: list of model blocks
    """
    def __init__(self, model_blocks: list[ModelBlock]) -> None:
        """
        initialize the ModelBlocksPipeline
        :param model_blocks: list of model blocks
        """
        self.model_blocks = model_blocks

    def get_pipeline(self) -> None | Pipeline:
        """
        get the pipeline
        :return: Pipeline object
        """
        steps = []

        for model_block in self.model_blocks:
            steps.append((str(model_block), model_block))

        if steps:
            return Pipeline(steps=steps)
        else:
            return None
