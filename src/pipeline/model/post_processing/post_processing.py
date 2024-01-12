"""PostProcessingPipeline class."""
from dataclasses import dataclass

from sklearn.pipeline import Pipeline


@dataclass
class PostProcessingPipeline(Pipeline):
    """PostProcessingPipeline is the class used to create the post processing pipeline."""

    def __post_init__(self) -> None:
        """Post init function."""
        super().__init__(self._get_steps())
        self.set_hash("")

    def _get_steps(self) -> list[tuple[str, Pipeline]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        # TODO(Jasper): Implement post processing pipeline steps
        return []

    def set_hash(self, prev_hash: str) -> str:
        """Set the hash.

        :param prev_hash: Previous hash
        :return: Hash
        """
        # TODO(Jasper): Implement post processing pipeline hash
        return prev_hash
