"""TransformationPipeline."""
from dataclasses import dataclass

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from joblib import hash


@dataclass
class TransformationPipeline(Pipeline):
    """TransformationPipeline class extends the sklearn Pipeline class.

    :param transformations: list of transformations
    """

    transformations: list[BaseEstimator]

    def __post_init__(self) -> None:
        """Post init function."""
        self.set_hash("")
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, BaseEstimator | Pipeline]]:
        """Get the transformation pipeline steps.

        :return: list of steps
        """
        # Use list comprehension to get the steps
        return [(str(transformation), transformation) for transformation in self.transformations]

    def set_hash(self, prev_hash: str) -> str:
        """set_hash function sets the hash for the pipeline.

        :param prev_hash: previous hash
        :return: hash
        """
        transformation_hash = prev_hash
        for transformation in self.transformations:
            transformation_hash = hash(
                str(transformation) + transformation_hash)

        self.prev_hash = transformation_hash
        return transformation_hash
