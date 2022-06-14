import os
from typing import Callable, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

import wandb
from rtk_mult_clf import utils

from .datamodules.datamodule_sklearn import SklearnRTKDataModule
from .metrics import opt_metrics
from .predictor.engine import train_model, valid_model


def resolve_tuple(*args):
    return tuple(args)


OmegaConf.register_new_resolver("as_tuple", resolve_tuple)

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains the training pipeline. Can additionally evaluate model on a
    testset, using best weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    wandb.init(**config.logger.init)

    log.info(f"Instantiating estimator <{config.model._target_}>")
    model: BaseEstimator = instantiate(config.model)

    log.info(
        f"Instantiating data transformer"
        f" <{config.data_transformer._target_}>"
    )
    pipeline: Pipeline = instantiate(
        config.data_transformer, _recursive_=False
    )

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: SklearnRTKDataModule = instantiate(config.datamodule)

    train_model(model, pipeline, datamodule)

    log.info(f"Optimized metric: <{config.get('optimized_metric')}>")
    optimized_metric: Callable = opt_metrics.get(
        config.get("optimized_metric"), None
    )
    metric_aggregation: str = config.get("metric_aggregation")
    if not optimized_metric:
        raise ValueError("Flag optimized_metric not defined!")

    if not metric_aggregation:
        raise ValueError("Flag metric_aggregation not defined!")

    score: float = valid_model(
        model, pipeline, datamodule, optimized_metric, metric_aggregation
    )

    folder_path: str = os.path.join(
        os.environ["PROJECT_PATH_ROOT"], config.checkpoints.path_to_checkpoints
    )

    os.makedirs(folder_path, exist_ok=True)

    save_path: str = os.path.join(
        folder_path,
        "_".join(
            [
                config.logger.init.name,
                config.get("optimized_metric"),
                str(round(score, 4)),
                ".pkl",
            ]
        ),
    )

    utils.save_model(
        {"model": model, "pipeline": pipeline}, save_path=save_path
    )

    utils.log_info_error_analysis(
        model, pipeline, datamodule, config.logger.init.name, save_path, score
    )

    # Return metric score for hyperparameter optimization
    return score
