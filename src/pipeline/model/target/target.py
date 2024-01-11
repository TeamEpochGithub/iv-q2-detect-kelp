"""TargetPipeline class sets up the target pipeline."""
from dataclasses import dataclass

from sklearn.pipeline import Pipeline


@dataclass
class TargetPipeline(Pipeline):
    """TargetPipeline is the class used to create the target pipeline.

    :param raw_target_path: The raw target path
    :param processed_path: The processed path
    :param transformation_steps: The transformation steps
    :param column_steps: The column steps
    """

    raw_target_path: str
    processed_path: str
    transformation_steps: list[str]
    column_steps: list[str]

    def __post_init__(self) -> None:
        """Post init function."""
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, Pipeline]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        # TODO(Jasper): Implement target pipeline steps
        return []
