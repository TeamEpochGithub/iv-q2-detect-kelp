"""Test instantiating a hydra config object"""

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="conf/models", config_name="unet2M")
def my_app(cfg: DictConfig) -> None:
    """Test instantiating a hydra config object

    :param cfg: hydra config object"""
    pipeline = instantiate(cfg.pipeline)
    import os
    print(os.getcwd())
    print(pipeline)


if __name__ == "__main__":
    my_app()
