"""TransformationPipeline."""
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class TransformationPipeline(Pipeline):
    """TransformationPipeline class extends the sklearn Pipeline class.

    :param transformations: list of transformations
    """

    def __init__(self, transformations: list[BaseEstimator]) -> None:
        """Initialize the TransformationPipeline.

        :param transformations: list of transformations
        """
        self.transformations = transformations
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, BaseEstimator | Pipeline]]:
        """Get the transformation pipeline steps.

        :return: list of steps
        """
        # Use list comprehension to get the steps
        return [(str(transformation), transformation) for transformation in self.transformations]

    def __str__(self) -> str:
        """__str__ returns string representation of the TransformationPipeline.

        :return: String representation of the TransformationPipeline
        """
        return "TransformationPipeline"


if __name__ == "__main__":
    from src.pipeline.model.feature.transformation.divider import Divider

    divider = Divider(2)

    transformation_pipeline = TransformationPipeline([divider])
    import numpy as np

    X = np.array([1, 2, 3, 4, 5])
    X = transformation_pipeline.fit_transform(X)
