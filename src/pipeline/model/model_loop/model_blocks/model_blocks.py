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
        self.set_hash("")
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, TorchBlock]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        # Use list comprehension to get the steps
        return [(str(model_block), model_block) for model_block in self.model_blocks]

    def set_hash(self, prev_hash: str) -> str:
        """Set the hash.

        :param prev_hash: Previous hash
        :return: Hash
        """
        model_blocks_hash = prev_hash

        for model_block in self.model_blocks:
            model_blocks_hash = model_block.set_hash(model_blocks_hash)

        self.prev_hash = model_blocks_hash

        return model_blocks_hash
