from typing import Callable, List

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from rtk_mult_clf import SklearnRTKDataModule, utils

log = utils.get_logger(__name__)


def train_model(
    model: BaseEstimator,
    pipeline: Pipeline,
    datamodule: SklearnRTKDataModule,
) -> None:
    log.info("Model fitting start")
    datamodule.setup()

    x_train, y_train = datamodule.get_train_data()

    pipeline.fit(x_train)

    model.fit(pipeline.transform(x_train), y_train)
    log.info("Model fitting end")


def valid_model(
    model: BaseEstimator,
    pipeline: Pipeline,
    datamodule: SklearnRTKDataModule,
    optimized_metric: Callable,
    metric_aggregation: str,
) -> float:
    log.info(f"Validation process {type(model)} start")
    x_valid, y_valid = datamodule.get_val_data()
    predictions: List[int] = model.predict(pipeline.transform(x_valid))

    score: float = optimized_metric(
        y_valid, predictions, average=metric_aggregation
    )

    log.info(f"Validation score: <{score}>")
    log.info(f"Validation process {type(model)} end")
    return score
