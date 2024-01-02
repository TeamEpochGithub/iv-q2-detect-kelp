"""Pretrain pipeline class."""
from typing import Any

from sklearn.pipeline import Pipeline


class PretrainPipeline:
    """Class used to create the pretrain pipeline.

    :param steps: list of steps
    """

    def __init__(self, steps: list[Any]) -> None:
        """Initialize the PretrainPipeline.

        :param steps: list of steps
        """
        self.steps = steps

    def get_pipeline(self) -> Pipeline | None:
        """Get the pipeline.

        :return: Pipeline object
        """
        if self.steps:
            return Pipeline(steps=self.steps)

        return None
