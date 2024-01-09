"""TransformationPipeline."""
from dataclasses import dataclass

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


@dataclass
class TransformationPipeline(Pipeline):
    """TransformationPipeline class extends the sklearn Pipeline class.

    :param transformations: list of transformations
    """

    transformations: list[BaseEstimator]

    def __post_init__(self) -> None:
        """Post init function."""
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, BaseEstimator | Pipeline]]:
        """Get the transformation pipeline steps.

        :return: list of steps
        """
        # Use list comprehension to get the steps
        return [(str(transformation), transformation) for transformation in self.transformations]
