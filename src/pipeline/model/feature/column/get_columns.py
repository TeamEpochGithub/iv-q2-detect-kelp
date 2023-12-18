from typing import Any
from sklearn.pipeline import Pipeline
from src.logging_utils import logger
from src.pipeline.model.feature.column.band_copy import BandCopyPipeline

from src.pipeline.model.feature.column.error import ColumnPipelineError


def get_columns(column_steps: list[dict[str, Any]], processed_path: str = "") -> Pipeline:
    """
    This function creates the column pipeline.
    :param column_steps: list of column steps
    :return: column pipeline
    """
    # TODO: Create the column pipeline
    steps = [match(column_step, processed_path=processed_path)
             for column_step in column_steps]

    # Create the column pipeline
    if not steps:
        return None
    else:
        return Pipeline(steps, memory=processed_path + "/column_pipeline/")


def match(column_step: dict[str, Any], processed_path: str = "") -> tuple[str, Any]:
    """
    This function matches the column steps to the correct function.
    :param column_step: column step
    :return: column pipeline
    """

    # Check if type is defined
    if not column_step.get("type"):
        logger.error("type is required")
        raise ColumnPipelineError("type is required")

    column_type = column_step.get("type")
    column_args = column_step.copy()
    column_args["processed_path"] = processed_path
    column_args.pop("type")
    # Check if the type is valid
    match column_type:
        case "band_copy":
            pipeline = BandCopyPipeline(**column_args)
            return f"band_copy_{column_args['band']}", pipeline.get_pipeline()
        case _:
            logger.error(f"Unknown column type: {column_type}")
            raise ColumnPipelineError(
                f"Unknown column type: {column_type}"
            )
