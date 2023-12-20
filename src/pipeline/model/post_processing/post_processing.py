from sklearn.pipeline import Pipeline


class PostProcessingPipeline():
    """
    PostProcessingPipeline is the class used to create the post processing pipeline.
    """
    def __init__(self) -> None:
        """
        Initialize the class.
        """
        # TODO create post processing pipeline
        pass

    def get_pipeline(self) -> Pipeline:
        """
        This function returns the post processing pipeline.

        :return: Pipeline object
        """
        steps = []

        # TODO add steps to pipeline

        if steps:
            return Pipeline(steps=steps)
        
    def __str__(self) -> str:
        """
        String representation of the class.

        :return: String representation of the class
        """
        return "PostProcessingPipeline"