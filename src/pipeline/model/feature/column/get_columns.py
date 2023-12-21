"""Module for the function to create the column pipeline."""
from typing import Any

from sklearn.pipeline import Pipeline

from src.pipeline.model.feature.column.band_copy import BandCopyPipeline
from src.pipeline.model.feature.column.error import ColumnPipelineError


def get_columns(column_steps: list[dict[str, Any]], processed_path: str | None = None) -> Pipeline | None:
    """Create the column pipeline.

    :param column_steps: list of column steps
    :param processed_path: path to the processed data
    :return: column pipeline
    """
    # Match the column steps to the correct function
    steps = [match(column_step, processed_path=processed_path) for column_step in column_steps]

    # Create the column pipeline
    if not steps:
        return None

    if processed_path:
        pipeline_path = processed_path + "/column_pipeline"
    else:
        pipeline_path = None
    return Pipeline(steps, memory=pipeline_path)


def match(column_step: dict[str, Any], processed_path: str | None = None) -> tuple[str, Any]:
    """Match the column steps to the correct function.

    :param column_step: column step
    :param processed_path: path to the processed data
    :return: column pipeline
    """
    # Check if type is defined
    if not column_step.get("type"):
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
            raise ColumnPipelineError(f"Unknown column type: {column_type}")
