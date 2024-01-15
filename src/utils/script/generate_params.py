"""Generate the model parameters for training and cross validation."""
from omegaconf import DictConfig

from src.pipeline.ensemble.ensemble import EnsemblePipeline
from src.pipeline.model.model import ModelPipeline
from src.utils.flatten_dict import flatten_dict


def generate_train_params(cfg: DictConfig, model_pipeline: ModelPipeline | EnsemblePipeline, train_indices: list[int], test_indices: list[int]) -> dict[str, str]:
    """Generate the model parameters.

    :param cfg: The configuration
    :return: The model parameters
    """
    if "model" in cfg:
        params = generate_model_params(model_pipeline, train_indices, test_indices, cfg.cache_size, save=True)
    elif "ensemble" in cfg:
        params = generate_ensemble_params(model_pipeline, train_indices, test_indices, cfg.cache_size, save=True)

    return params


def generate_cv_params(cfg: DictConfig, model_pipeline: ModelPipeline | EnsemblePipeline, train_indices: list[int], test_indices: list[int]) -> dict[str, str]:
    """Generate the model parameters.

    :param cfg: The configuration
    :return: The model parameters
    """
    if "model" in cfg:
        params = generate_model_params(model_pipeline, train_indices, test_indices, cfg.cache_size, save=False)
    elif "ensemble" in cfg:
        params = generate_ensemble_params(model_pipeline, train_indices, test_indices, cfg.cache_size, save=False)

    return params


def generate_model_params(
    model_pipeline: ModelPipeline, train_indices: list[int] | None = None, test_indices: list[int] | None = None, cache_size: int = -1, *, save: bool = True
) -> dict[str, str]:
    """Generate the model parameters.

    :param train_indices: The train indices
    :param test_indices: The test indices
    :param cache_size: The cache size
    :param save: Whether to save the model or not
    :return: The model parameters
    """
    new_params = (
        {
            "model_loop_pipeline_step": {
                "pretrain_pipeline_step": {
                    name: {"train_indices": train_indices, "save_pretrain": save} for name, _ in model_pipeline.model_loop_pipeline.named_steps.pretrain_pipeline_step.steps
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
            "post_processing_pipeline_step":{
                name: {"test_indices": test_indices}
                for name, _ in model_pipeline.named_steps.post_processing_pipeline_step.steps
            }
        }
        if model_pipeline
        else {}
    )

    return flatten_dict(new_params)


def generate_ensemble_params(
    ensemble_pipeline: EnsemblePipeline, train_indices: list[int] | None = None, test_indices: list[int] | None = None, cache_size: int = -1, *, save: bool = True
) -> dict[str, str]:
    """Generate the model parameters.

    :param train_indices: The train indices
    :param test_indices: The test indices
    :param cache_size: The cache size
    :param save: Whether to save the model or not
    :return: The model parameters
    """
    new_params = (
        {
            name: {
                "model_loop_pipeline_step": {
                    "pretrain_pipeline_step": {
                        name: {"train_indices": train_indices, "save_pretrain": save}
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
                else {}
            }
            for name, model_pipeline in ensemble_pipeline.steps
        }
        if ensemble_pipeline
        else {}
    )

    return flatten_dict(new_params)
