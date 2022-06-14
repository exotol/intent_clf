import os
from typing import Any, Dict, List

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from rtk_mult_clf import utils

from .datamodules.datamodule_sklearn import SklearnRTKDataModule


def resolve_tuple(*args):
    return tuple(args)


OmegaConf.register_new_resolver("as_tuple", resolve_tuple)

log = utils.get_logger(__name__)


def test(config: DictConfig) -> None:
    """Contains the training pipeline. Can additionally evaluate model on a
    testset, using best weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    if not os.path.isabs(config.predictions_path):
        config.predictions_path = os.path.join(
            os.environ["PROJECT_PATH_ROOT"], config.predictions_path
        )

    if not os.path.isabs(config.ckpt_path):
        config.ckpt_path = os.path.join(
            os.environ["PROJECT_PATH_ROOT"], config.ckpt_path
        )

    checkpoint: Dict[str, Any] = utils.load_model(config.ckpt_path)

    model: BaseEstimator = checkpoint["model"]
    log.info(f"Instantiating estimator <{model.__class__.__name__}>")

    pipeline: Pipeline = checkpoint["pipeline"]
    log.info(
        f"Instantiating data transformer" f" <{pipeline.__class__.__name__}>"
    )

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: SklearnRTKDataModule = instantiate(config.datamodule)

    datamodule.setup(stage="test")
    x_test = datamodule.get_test_data()
    predictions: List[int] = model.predict(pipeline.transform(x_test))
    x_test["y"] = predictions

    predictions_path: str = os.path.join(config.predictions_path, config.name)
    os.makedirs(predictions_path, exist_ok=True)
    file_name_full: str = os.path.join(
        predictions_path, "Bardakov_AA_full.csv"
    )
    file_name_task: str = os.path.join(predictions_path, "Bardakov_AA.csv")
    x_test.to_csv(file_name_full, index=False)
    x_test[["id", "y"]].to_csv(file_name_task, index=False)
