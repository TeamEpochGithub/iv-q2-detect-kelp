from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class TransformationPipeline(Pipeline):
    """
    TransformationPipeline

    :param transformations: list of transformations
    """

    def __init__(self, transformations: list[BaseEstimator]) -> None:
        """
        Initialize the TransformationPipeline

        :param transformations: list of transformations
        """
        self.transformations = transformations
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, BaseEstimator | Pipeline]]:
        """
        Get the transformation pipeline steps

        :return: list of steps
        """
        steps = []
        for transformation in self.transformations:
            if transformation:
                steps.append((str(transformation), transformation))
        return steps

    def __str__(self) -> str:
        """
        String representation of the TransformationPipeline

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
    print(X)
