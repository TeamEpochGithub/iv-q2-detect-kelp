from sklearn.pipeline import Pipeline

# TODO: Remove these classes
from sklearn.base import BaseEstimator 
class PretrainPipeline():
    def __init__(self, pretrain_blocks: list[BaseEstimator]) -> None:
        """
        Initialize the class.

        :param pretrain_blocks: The pretrain blocks
        """
        pass
    
    def get_pipeline(self) -> Pipeline | None:
        """
        PretrainPipeline is the class used to create the pretrain pipeline.

        :return: Pipeline object
        """
        pass

class ModelBlocksPipeline():
    def __init__(self, model_blocks: list[BaseEstimator]) -> None:
        """
        Initialize the class.

        :param model_blocks: The model blocks
        """
        pass
    
    def get_pipeline(self) -> Pipeline | None:
        """
        ModelBlocksPipeline is the class used to create the model blocks pipeline.

        :return: Pipeline object
        """
        pass
# TODO: Remove till here

class ModelLoopPipeline():
    def __init__(self, pretrain_pipeline: PretrainPipeline, model_blocks_pipeline: ModelBlocksPipeline) -> None:
        """
        Initialize the class.

        :param pretrain_pipeline: The pretrain pipeline
        :param model_blocks_pipeline: The model blocks pipeline
        """
        self.pretrain_pipeline = pretrain_pipeline
        self.model_blocks_pipeline = model_blocks_pipeline
    
    def get_pipeline(self) -> Pipeline | None:
        """
        ModelLoopPipeline is the class used to create the model loop pipeline.

        :return: Pipeline object
        """
        steps = []
        if self.pretrain_pipeline:
            steps.append(('pretrain_pipeline', self.pretrain_pipeline.get_pipeline()))
        if self.model_blocks_pipeline:
            steps.append(('model_blocks_pipeline', self.model_blocks_pipeline.get_pipeline()))

        if steps:
            return Pipeline(steps)
    
    def __str__(self) -> str:
        """
        String representation of the class.

        :return: String representation of the class
        """
        return "ModelLoopPipeline"
