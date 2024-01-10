"""Pretrain pipeline class."""
from dataclasses import dataclass
from typing import Any

from sklearn.pipeline import Pipeline


@dataclass
class PretrainPipeline(Pipeline):
    """Class used to create the pretrain pipeline.

    :param steps: list of steps
    """

    steps: list[Any]

    def __post_init__(self) -> None:
        """Post init function."""
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, Any]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        # Use list comprehension to get the steps
        # if isinstance(self.steps[0], tuple):
        #     return self.steps
        # else:
        return [(str(step), step) for step in self.steps]

    def load_scaler(self, scaler_hash: str) -> None:
        """Load the scaler from the scaler hash.

        :param scaler_hash: The scaler hash
        """
        for step in self.steps:
            if hasattr(step, "load_scaler"):
                step.load_scaler(scaler_hash)
