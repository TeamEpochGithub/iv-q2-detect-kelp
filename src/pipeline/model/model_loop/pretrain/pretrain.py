from typing import Any

from sklearn.pipeline import Pipeline


class PretrainPipeline():
    """
    This class is used to create the pretrain pipeline.
    :param steps: list of steps
    """
    def __init__(self, steps: list[Any]):
        """
        initialize the PretrainPipeline
        :param steps: list of steps
        """
        self.steps = steps

    def get_pipeline(self) -> Pipeline | None:
        """
        get the pipeline
        :return: Pipeline object
        """
        if self.steps:
            return Pipeline(steps=self.steps)
        else:
            return None
