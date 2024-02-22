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
    }
    for k in os.environ:
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]
