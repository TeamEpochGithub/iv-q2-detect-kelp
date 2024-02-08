"""Sweep.py is used to run a sweep on a model pipeline with K fold split. Entry point for Hydra which loads the config file."""
import copy
import multiprocessing
import os
import warnings
from contextlib import nullcontext
from multiprocessing import Queue
from pathlib import Path
from typing import NamedTuple

import dask.array as da
import hydra
import randomname
from distributed import Client
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig

import wandb
from src.config.cross_validation_config import CVConfig
from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator
from src.utils.script.generate_params import generate_cv_params
from src.utils.script.lock import Lock
from src.utils.script.reset_wandb_env import reset_wandb_env
from src.utils.seed_torch import set_torch_seed
from src.utils.setup import setup_config, setup_pipeline, setup_train_data, setup_wandb


class Worker(NamedTuple):
    """A worker.

    :param queue: The queue to get the data from
    :param process: The process
    """

    queue: Queue  # type: ignore[type-arg]
    process: multiprocessing.Process


class WorkerInitData(NamedTuple):
    """The data to initialize a worker.

    :param cfg: The configuration
    :param output_dir: The output directory
    :param wandb_group_name: The wandb group name
    :param i: The fold number
    :param train_indices: The train indices
    :param test_indices: The test indices
    :param X: The X data
    :param y: The y data
    """

    cfg: DictConfig
    output_dir: Path
    wandb_group_name: str
    i: int
    train_indices: list[int]
    test_indices: list[int]
    X: da.Array
    y: da.Array


class WorkerDoneData(NamedTuple):
    """The data a worker returns.

    :param sweep_score: The sweep score
    """

    sweep_score: float


warnings.filterwarnings("ignore", category=UserWarning)
# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"

# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_cv", node=CVConfig)


@hydra.main(version_base=None, config_path="conf", config_name="cv")
def run_cv(cfg: DictConfig) -> None:  # TODO(Jeffrey): Use CVConfig instead of DictConfig
    """Do cv on a model pipeline with K fold split. Entry point for Hydra which loads the config file."""
    # Run the cv config with a dask client, and optionally a lock
    optional_lock = Lock if not cfg.allow_multiple_instances else nullcontext
    with optional_lock(), Client() as client:
        logger.info(f"Client: {client}")
        run_cv_cfg(cfg)


def run_cv_cfg(cfg: DictConfig) -> None:
    """Do cv on a model pipeline with K fold split."""
    print_section_separator("Q2 Detect Kelp States -- CV")

    import coloredlogs

    coloredlogs.install()

    # Set seed
    set_torch_seed()

    # Check for missing keys in the config file
    setup_config(cfg)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Lazily read the raw data with dask, and find the shape after processing
    X, y = setup_train_data(cfg.raw_data_path, cfg.raw_target_path)

    # Set up Weights & Biases group name
    wandb_group_name = randomname.get_name()

    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start
    sweep_q: Queue[WorkerDoneData] = multiprocessing.Queue()
    workers = []
    for _ in range(cfg.splitter.n_splits):
        q: Queue[WorkerInitData] = multiprocessing.Queue()
        p = multiprocessing.Process(target=try_fold_run, kwargs={"sweep_q": sweep_q, "worker_q": q})
        p.start()
        workers.append(Worker(queue=q, process=p))

    # Initialize wandb
    sweep_run = setup_wandb(cfg, "sweep", output_dir, name=wandb_group_name, group=wandb_group_name)

    metrics = []
    failed = False  # If any worker fails, stop the run
    for num, (train_indices, test_indices) in enumerate(instantiate(cfg.splitter).split(X, y)):
        set_torch_seed()
        worker = workers[num]

        # If failed, stop the run
        if failed:
            logger.debug(f"Stopping worker {num}")
            worker.process.terminate()
            continue

        # Start worker
        worker.queue.put(
            WorkerInitData(
                cfg=cfg,
                output_dir=output_dir,
                wandb_group_name=wandb_group_name,
                i=num,
                train_indices=train_indices,
                test_indices=test_indices,
                X=X,
                y=y,
            ),
        )
        # Get metric from worker
        result = sweep_q.get()
        # Wait for worker to finish
        worker.process.join()
        # Log metric to sweep_run
        metrics.append(result.sweep_score)

        # If failed, stop the run by setting failed to True
        if result.sweep_score == -0.1:
            logger.error("Worker failed")
            failed = True
            continue

        # If score is too low, stop the run by setting failed to True
        if result.sweep_score < 0.25:
            logger.debug("Worker score too low, stopping run")
            failed = True
            continue

    if sweep_run is not None:
        sweep_run.log({"sweep_score": sum(metrics) / len(metrics)})
    wandb.join()


def try_fold_run(sweep_q: Queue, worker_q: Queue) -> None:  # type: ignore[type-arg]
    """Run a fold, and catch exceptions.

    :param sweep_q: The queue to put the result in
    :param worker_q: The queue to get the data from
    """
    try:
        fold_run(sweep_q, worker_q)
    except Exception as e:  # noqa: BLE001
        logger.error(e)
        wandb.join()
        sweep_q.put(WorkerDoneData(sweep_score=-0.1))


def fold_run(sweep_q: Queue, worker_q: Queue) -> None:  # type: ignore[type-arg]
    """Run a fold.

    :param sweep_q: The queue to put the result in
    :param worker_q: The queue to get the data from
    """
    # Get the data from the queue
    worker_data = worker_q.get()
    cfg = worker_data.cfg
    output_dir = worker_data.output_dir
    wandb_group_name = worker_data.wandb_group_name
    i = worker_data.i
    train_indices = worker_data.train_indices
    test_indices = worker_data.test_indices
    X = worker_data.X
    y = worker_data.y

    score = _one_fold(cfg, output_dir, i, wandb_group_name, train_indices, test_indices, X, y)

    sweep_q.put(WorkerDoneData(sweep_score=score))


def _one_fold(cfg: DictConfig, output_dir: Path, fold: int, wandb_group_name: str, train_indices: list[int], test_indices: list[int], X: da.Array, y: da.Array) -> float:
    """Run one fold of cv.

    :param cfg: The configuration
    :param output_dir: The output directory
    :param fold: The fold number
    :param wandb_group_name: The wandb group name
    :param train_indices: The train indices
    :param test_indices: The test indices
    :param X: The X data
    :param y: The y data

    :return: The score
    """
    # https://github.com/wandb/wandb/issues/5119
    # This is a workaround for the issue where sweeps override the run id annoyingly
    reset_wandb_env()

    # Print section separator
    print_section_separator(f"CV - Fold {fold}")
    logger.info(f"Train/Test size: {len(train_indices)}/{len(test_indices)}")

    if cfg.wandb.enabled:
        wandb_fold_run = setup_wandb(cfg, "cv", output_dir, name=f"{wandb_group_name}_{fold}", group=wandb_group_name)

    logger.info("Creating clean pipeline for this fold")
    model_pipeline = setup_pipeline(cfg, output_dir, is_train=True)

    # Generate the parameters for training
    fit_params = generate_cv_params(cfg, model_pipeline, train_indices, test_indices)

    # Fit the pipeline
    target_pipeline = model_pipeline.get_target_pipeline()
    original_y = copy.deepcopy(y)

    if target_pipeline is not None:
        print_section_separator("Target pipeline")
        y = target_pipeline.fit_transform(y)

    # Fit the pipeline and get predictions
    predictions = model_pipeline.fit_transform(X, y, **fit_params)
    scorer = instantiate(cfg.scorer)
    score = scorer(original_y[test_indices].compute(), predictions[test_indices])
    logger.info(f"Score: {score}")
    if wandb_fold_run is not None:
        wandb_fold_run.log({"Score": score})

    logger.info("Finishing wandb run")
    wandb.join()
    return score


if __name__ == "__main__":
    run_cv()
