"""PostProcessingPipeline class."""
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from typing import Any

@dataclass
class PostProcessingPipeline(Pipeline):
    """PostProcessingPipeline is the class used to create the post processing pipeline."""

    steps: list[Any]

    def __post_init__(self) -> None:
        """Post init function."""
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, Pipeline]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        # TODO(Jasper): Implement post processing pipeline steps
        return [(str(step), step) for step in self.steps]
