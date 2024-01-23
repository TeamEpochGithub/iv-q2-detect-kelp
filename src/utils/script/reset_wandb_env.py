"""Reset wandb environment variables."""
import os


def reset_wandb_env() -> None:
    """Reset wandb environment variables."""
    # if "WANDB_RUN_ID" in os.environ:
    #     del os.environ["WANDB_RUN_ID"]

    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
        "WANDB_SWEEP_ID",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]
