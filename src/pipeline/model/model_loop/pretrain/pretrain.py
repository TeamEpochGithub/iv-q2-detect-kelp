"""Pretrain pipeline class."""
from typing import Any

from sklearn.pipeline import Pipeline


class PretrainPipeline(Pipeline):
    """Class used to create the pretrain pipeline.

    :param steps: list of steps
    """

    def __init__(self, steps: list[Any]) -> None:
        """Initialize the PretrainPipeline.

        :param steps: list of steps
        """
        self.steps = steps
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
