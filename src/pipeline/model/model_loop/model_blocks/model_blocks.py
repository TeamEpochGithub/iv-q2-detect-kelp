"""Model blocks pipeline."""
from sklearn.pipeline import Pipeline

from src.pipeline.model.model_loop.model_blocks.model_fit_block import ModelBlock


class ModelBlocksPipeline(Pipeline):
    """ModelBlocksPipeline class is used to create the model blocks pipeline.

    :param model_blocks: list of model blocks
    """

    def __init__(self, model_blocks: list[ModelBlock]) -> None:
        """Initialize the ModelBlocksPipeline.

        :param model_blocks: list of model blocks
        """
        self.model_blocks = model_blocks
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, ModelBlock]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        # Use list comprehension to get the steps
        return [(str(model_block), model_block) for model_block in self.model_blocks]
