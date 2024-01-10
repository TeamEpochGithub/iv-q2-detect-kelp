"""Pretrain pipeline class."""
from dataclasses import dataclass
from typing import Any

from sklearn.pipeline import Pipeline

from src.pipeline.model.model_loop.pretrain.scaler_block import ScalerBlock


@dataclass
class PretrainPipeline(Pipeline):
    """Class used to create the pretrain pipeline.

    :param steps: list of steps
    """

    steps: list[ScalerBlock]

    def __post_init__(self) -> None:
        """Post init function."""
        self.set_hash("")
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, ScalerBlock]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        # Use list comprehension to get the steps
        # if isinstance(self.steps[0], tuple):
        #     return self.steps
        # else:
        return [(str(step), step) for step in self.steps]

    def set_hash(self, prev_hash: str) -> str:
        """Set the hash.

        :param prev_hash: Previous hash
        :return: Hash
        """
        pretrain_hash = prev_hash
        for step in self.steps:
            pretrain_hash = step.set_hash(pretrain_hash)

        self.prev_hash = pretrain_hash

        return pretrain_hash