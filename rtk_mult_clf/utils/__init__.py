import logging
import os
import pickle
import subprocess
import warnings
from argparse import Namespace
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import rich.syntax
import rich.tree
import seaborn as sns
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

import wandb
from rtk_mult_clf import SklearnRTKDataModule


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU
    # process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


def extras(config: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags.

    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library if <config.print_config=True>
    if config.get("print_config"):
        log.info("Printing config tree with Rich! <config.print_config=True>")
        print_config(config, resolve=True)


@rank_zero_only
def print_config(
    config: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config
                     components are printed.
        resolve (bool, optional): Whether to resolve reference
                    fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    for field in print_order:
        queue.append(field) if field in config else log.info(
            f"Field '{field}' not found in config"
        )

    for field in config:
        if field not in queue:
            queue.append(field)

    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    path_to_log: str = os.path.join(config.current_work_dir, "config_tree.log")
    with open(path_to_log, "w") as file:
        rich.print(tree, file=file)


@rank_zero_only
def log_hyper_parameters(
    config: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionaly saves:
    - number of model parameters
    """

    if not trainer.logger:
        return

    hparams: Dict[str, Any] = {
        "model": config["model"],
        "model/params/total": sum(p.numel() for p in model.parameters()),
        "model/params/trainable": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "model/params/non_trainable": sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        ),
        "datamodule": config["datamodule"],
        "trainer": config["trainer"],
    }

    # choose which parts of hydra config will be saved to loggers

    # save number of model parameters

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # send hparams to all loggers
    trainer.logger.log_hyperparams(Namespace(**hparams))


def finish(
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


def log_wandb_confusion_matrix(
    model: BaseEstimator,
    pipeline: Pipeline,
    datamodule: SklearnRTKDataModule,
    experiment_name: str,
) -> None:
    x_valid, y_valid = datamodule.get_val_data()

    predictions: List[float] = model.predict(pipeline.transform(x_valid))
    confusion_matrix = metrics.confusion_matrix(
        y_true=y_valid, y_pred=predictions
    )
    plt.figure(figsize=(14, 8))
    sns.set(font_scale=1.4)
    sns.heatmap(confusion_matrix, annot=True, annot_kws={"size": 8}, fmt="g")
    wandb.log(
        {f"confusion_matrix/{experiment_name}": wandb.Image(plt)}, commit=False
    )
    # reset plot
    plt.clf()


def log_wandb_precision_recall(
    model: BaseEstimator,
    pipeline: Pipeline,
    datamodule: SklearnRTKDataModule,
    experiment_name: str,
    metric_aggregation: str,
) -> None:
    x_valid, y_valid = datamodule.get_val_data()
    predictions: List[float] = model.predict(pipeline.transform(x_valid))

    f1: float = metrics.f1_score(
        y_valid, predictions, average=metric_aggregation
    )
    recall: float = metrics.recall_score(
        y_valid, predictions, average=metric_aggregation
    )
    precision: float = metrics.precision_score(
        y_valid, predictions, average=metric_aggregation
    )
    data: List[List[float]] = [[f1], [precision], [recall]]

    # set figure size
    plt.figure(figsize=(14, 3))

    # set labels size
    sns.set(font_scale=1.2)

    # set font size
    sns.heatmap(
        data,
        annot=True,
        annot_kws={"size": 10},
        fmt=".3f",
        yticklabels=["F1", "Precision", "Recall"],
    )

    log_key: str = f"f1_p_r_heatmap/{experiment_name}/{metric_aggregation}"
    wandb.log(
        {log_key: wandb.Image(plt)},
        commit=False,
    )

    # reset plot
    plt.clf()


def log_wandb_classification_report(
    model: BaseEstimator,
    pipeline: Pipeline,
    datamodule: SklearnRTKDataModule,
    experiment_name: str,
) -> None:
    x_valid, y_valid = datamodule.get_val_data()
    predictions: List[float] = model.predict(pipeline.transform(x_valid))

    clf_report = metrics.classification_report(
        y_valid,
        predictions,
        labels=model.classes_,
        target_names=model.classes_,
        output_dict=True,
    )

    # set figure size
    plt.figure(figsize=(14, 6))

    # set labels size
    sns.set(font_scale=1.2)

    sns.heatmap(
        pd.DataFrame(clf_report),
        annot=True,
        annot_kws={"size": 10},
        fmt=".3f",
    )
    plt.xticks(rotation=15)
    wandb.log(
        {f"clf_report/{experiment_name}": wandb.Image(plt)}, commit=False
    )

    # reset plot
    plt.clf()


def log_wandb_error_predictions(
    model: BaseEstimator,
    pipeline: Pipeline,
    datamodule: SklearnRTKDataModule,
    experiment_name: str,
) -> None:
    x_valid, y_valid = datamodule.get_val_data()
    predictions: List[float] = model.predict(pipeline.transform(x_valid))
    if hasattr(model, 'predict_proba'):
        prob_predictions: List[List[float]] = model.predict_proba(
            pipeline.transform(x_valid)
        )
    else:
        prob_predictions: List[List[float]] = model.decision_function(
            pipeline.transform(x_valid)
        )
    pd.options.mode.chained_assignment = None
    logic_index: pd.SupportsIndex = predictions != y_valid
    error_report: pd.DataFrame = x_valid.loc[logic_index]
    error_report["target"] = y_valid.loc[logic_index]
    error_report["predictions"] = predictions[logic_index]
    columns: List[str] = [
        "prob_0",
        "prob_1",
        "prob_2",
        "prob_3",
        "prob_4",
        "prob_5",
        "prob_6",
        "prob_7",
        "prob_8",
        "prob_9",
        "prob_10",
    ]
    error_report[columns] = prob_predictions[logic_index, :]

    wandb.log({f"error_report/{experiment_name}": error_report}, commit=False)


def log_wandb_artifact(experiment_name: str, path_to_save_local: str) -> None:
    ckpts: wandb.Artifact = wandb.Artifact(experiment_name, type="checkpoints")
    ckpts.add_file(path_to_save_local)
    wandb.log_artifact(ckpts)


def log_code_to_git(experiment_name: str, score: float) -> None:
    command = ["git", "add", "*"]
    not_ignored: bool = subprocess.run(command).returncode == 1
    log.warning("Git add processed with error: {}".format(not_ignored))

    experiment_msg: str = "Experiment {}, " "target score: {}".format(
        experiment_name, score
    )
    command = ["git", "commit", "-m", experiment_msg]
    not_ignored: bool = subprocess.run(command).returncode == 1
    log.warning("Git commit processed with error: {}".format(not_ignored))


def log_info_error_analysis(
    model: BaseEstimator,
    pipeline: Pipeline,
    datamodule: SklearnRTKDataModule,
    experiment_name: str,
    path_to_save_local: str,
    score: float,
) -> None:
    log.info("Info for Error Analysis!")
    log_wandb_confusion_matrix(model, pipeline, datamodule, experiment_name)
    log_wandb_precision_recall(
        model, pipeline, datamodule, experiment_name, "micro"
    )
    log_wandb_precision_recall(
        model, pipeline, datamodule, experiment_name, "macro"
    )
    log_wandb_classification_report(
        model,
        pipeline,
        datamodule,
        experiment_name,
    )
    log_wandb_error_predictions(
        model,
        pipeline,
        datamodule,
        experiment_name,
    )
    log_wandb_artifact(experiment_name, path_to_save_local)
    log_code_to_git(experiment_name, score)


def save_model(model: Dict[str, Any], save_path: str) -> str:
    with open(save_path, "wb") as output_stream:
        pickle.dump(model, output_stream)
    return save_path


def load_model(load_path: str) -> Optional[Dict[str, Any]]:
    with open(load_path, "rb") as input_stream:
        model = pickle.load(input_stream)
    return model
