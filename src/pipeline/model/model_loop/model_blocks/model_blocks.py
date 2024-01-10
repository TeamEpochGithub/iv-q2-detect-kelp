"""Model blocks pipeline."""
from dataclasses import dataclass

from sklearn.pipeline import Pipeline

from src.pipeline.model.model_loop.model_blocks.torch_block import TorchBlock


@dataclass
class ModelBlocksPipeline(Pipeline):
    """ModelBlocksPipeline class is used to create the model blocks pipeline.

    :param model_blocks: list of model blocks
    """

    model_blocks: list[TorchBlock]

    def __post_init__(self) -> None:
        """Post init function."""
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, TorchBlock]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        # Use list comprehension to get the steps
        return [(str(model_block), model_block) for model_block in self.model_blocks]

    def load_model(self, model_hash: str) -> None:
        """Load the model from the model hash.

        :param model_hash: The model hash
        """
        for model_block in self.model_blocks:
            model_block.load_model(model_hash)
