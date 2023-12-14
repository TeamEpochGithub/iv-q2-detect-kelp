from sklearn.pipeline import Pipeline

from src.pipeline.model.feature.transformation.divider import Divider
from src.pipeline.model.feature.transformation.error import TransformationPipelineError
from src.logging_utils.logger import logger


def get_transformations(transformation_steps: list[dict]) -> Pipeline:
    """
    This function creates the transformation pipeline.
    :param transformation_steps: list of transformation steps
    :return: transformation pipeline
    """

    # Create the transformation pipeline
    steps = [match(transformation_step) for transformation_step in transformation_steps]

    # Create the transformation pipeline
    if not steps:
        return None

    return Pipeline(steps)


def match(transformation_step: dict) -> tuple[str, any]:
    """
    This function matches the transformation steps to the correct function.
    :param transformation_steps: list of transformation steps
    :return: transformation pipeline
    """

    # Check if type is defined
    if not transformation_step.get("type"):
        logger.error("type is required")
        raise TransformationPipelineError("type is required")

    transformation_type = transformation_step.get("type")
    transformation_args = transformation_step.copy()
    transformation_args.pop("type")
    # Check if the type is valid
    match transformation_type:
        case "divider":
            return "divider", Divider(**transformation_args)
        case _:
            logger.error(f"Unknown transformation type: {transformation_type}")
            raise TransformationPipelineError(
                f"Unknown transformation type: {transformation_type}"
            )
