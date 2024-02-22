"""Generate the model parameters for training and cross validation."""
from omegaconf import DictConfig

from src.pipeline.ensemble.ensemble_base import EnsembleBase
from src.pipeline.model.model import ModelPipeline
from src.utils.flatten_dict import flatten_dict


def generate_train_params(cfg: DictConfig, model_pipeline: ModelPipeline | EnsembleBase, train_indices: list[int], test_indices: list[int]) -> dict[str, str]:
    """Generate the model parameters.

    :param cfg: The configuration
    :param model_pipeline: The model pipeline
    :param train_indices: The train indices
    :param test_indices: The test indices
    :return: The model parameters
    """
    if "model" in cfg:
        return generate_model_params(model_pipeline, train_indices, test_indices, cfg.cache_size, save=True)
    if "ensemble" in cfg:
        return generate_ensemble_params(model_pipeline, train_indices, test_indices, cfg.cache_size, save=True)
    raise ValueError("No model or ensemble found in config")


def generate_cv_params(cfg: DictConfig, model_pipeline: ModelPipeline | EnsembleBase, train_indices: list[int], test_indices: list[int]) -> dict[str, str]:
    """Generate the model parameters.

    :param cfg: The configuration
    :param model_pipeline: The model pipeline
    :param train_indices: The train indices
    :param test_indices: The test indices
    :return: The model parameters
    """
    if "model" in cfg:
        return generate_model_params(model_pipeline, train_indices, test_indices, cfg.cache_size, save=False, save_pretrain_with_split=True)
    if "ensemble" in cfg:
        return generate_ensemble_params(model_pipeline, train_indices, test_indices, cfg.cache_size, save=False, save_pretrain_with_split=True)
    raise ValueError("No model or ensemble found in config")


def generate_model_params(
    model_pipeline: ModelPipeline,
    train_indices: list[int] | None = None,
    test_indices: list[int] | None = None,
    cache_size: int = -1,
    *,
    save: bool = True,
    save_pretrain: bool = True,
    save_pretrain_with_split: bool = False,
) -> dict[str, str]:
    """Generate the model parameters.

    :param model_pipeline: The model pipeline
    :param train_indices: The train indices
    :param test_indices: The test indices
    :param cache_size: The cache size
    :param save: Whether to save the model or not
    :param save_pretrain: Whether to save the pretrain or not
    :param save_pretrain_with_split: Whether to save the pretrain with the split or not
    :return: The model parameters
    """
    new_params = (
        {
            "model_loop_pipeline_step": {
                "pretrain_pipeline_step": {
                    name: {"train_indices": train_indices, "save_pretrain": save_pretrain, "save_pretrain_with_split": save_pretrain_with_split}
                    for name, _ in model_pipeline.model_loop_pipeline.named_steps.pretrain_pipeline_step.steps
                }
                if "pretrain_pipeline_step" in model_pipeline.model_loop_pipeline.named_steps
                else {},
                "model_blocks_pipeline_step": {
                    name: {"train_indices": train_indices, "test_indices": test_indices, "cache_size": cache_size, "save_model": save}
                    for name, _ in model_pipeline.model_loop_pipeline.named_steps.model_blocks_pipeline_step.steps
                }
                if "model_blocks_pipeline_step" in model_pipeline.model_loop_pipeline.named_steps
                else {},
            }
            if "model_loop_pipeline_step" in model_pipeline.named_steps and model_pipeline.model_loop_pipeline
            else {},
            "post_processing_pipeline_step": {name: {"test_indices": test_indices} for name, _ in model_pipeline.named_steps.post_processing_pipeline_step.steps}
            if "post_processing_pipeline_step" in model_pipeline.named_steps
            else {},
        }
        if model_pipeline
        else {}
    )

    return flatten_dict(new_params)


def generate_ensemble_params(
    ensemble_pipeline: EnsembleBase,
    train_indices: list[int] | None = None,
    test_indices: list[int] | None = None,
    cache_size: int = -1,
    *,
    save: bool = True,
    save_pretrain: bool = True,
    save_pretrain_with_split: bool = False,
) -> dict[str, str]:
    """Generate the model parameters.

    :param ensemble_pipeline: The ensemble pipeline
    :param train_indices: The train indices
    :param test_indices: The test indices
    :param cache_size: The cache size
    :param save: Whether to save the model or not
    :param save_pretrain: Whether to save the pretrain or not
    :param save_pretrain_with_split: Whether to save the pretrain with the split or not
    :return: The model parameters
    """
    new_params = (
        {
            name: {
                "model_loop_pipeline_step": {
                    "pretrain_pipeline_step": {
                        name: {"train_indices": train_indices, "save_pretrain": save_pretrain, "save_pretrain_with_split": save_pretrain_with_split}
                        for name, _ in model_pipeline.model_loop_pipeline.named_steps.pretrain_pipeline_step.steps
                    }
                    if "pretrain_pipeline_step" in model_pipeline.model_loop_pipeline.named_steps
                    else {},
                    "model_blocks_pipeline_step": {
                        name: {"train_indices": train_indices, "test_indices": test_indices, "cache_size": cache_size, "save_model": save}
                        for name, _ in model_pipeline.model_loop_pipeline.named_steps.model_blocks_pipeline_step.steps
                    }
                    if "model_blocks_pipeline_step" in model_pipeline.model_loop_pipeline.named_steps
                    else {},
                }
                if "model_loop_pipeline_step" in model_pipeline.named_steps and model_pipeline.model_loop_pipeline
                else {},
                "post_processing_pipeline_step": {name: {"test_indices": test_indices} for name, _ in model_pipeline.named_steps.post_processing_pipeline_step.steps}
                if "post_processing_pipeline_step" in model_pipeline.named_steps
                else {},
            }
            for name, model_pipeline in ensemble_pipeline.models.items()
        }
        if ensemble_pipeline
        else {}
    )

    new_params["train_indices"] = train_indices  # type: ignore[assignment]
    new_params["test_indices"] = test_indices  # type: ignore[assignment]

    return flatten_dict(new_params)
