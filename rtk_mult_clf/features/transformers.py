from __future__ import annotations

from typing import Any, Optional, Union

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline


def make_pipeline(steps_config: DictConfig) -> Pipeline:
    """Creates a pipeline with all the preprocessing steps specified in
    `steps_config`, ordered in a sequential manner
    Args:
        steps_config (DictConfig): the config containing the instructions for
                            creating the feature selectors or transformers
    Returns:
        [sklearn.pipeline.Pipeline]: a pipeline with all the
        preprocessing steps, in a sequential manner
    """
    steps = []

    for step_config in steps_config:
        # retrieve the name and parameter dictionary of the current steps
        step_name, step_params = list(step_config.items())[0]

        # instantiate the pipeline step, and append to the list of steps
        pipeline_step = (step_name, hydra.utils.instantiate(step_params))
        steps.append(pipeline_step)

    return Pipeline(steps)


class CountVectorizerDF:
    def __init__(self, column_name: str, **kwargs: Any):
        self.count_vectorizer: CountVectorizer = CountVectorizer(**kwargs)
        self.column_name: str = column_name

    def fit(
        self,
        data: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> CountVectorizerDF:
        # y: Optional[Union[pd.Series, np.ndarray]]
        # необходим по требования Pipeline
        self.count_vectorizer.fit(data[self.column_name].values)
        return self

    def transform(self, data: pd.DataFrame) -> Any:
        return self.count_vectorizer.transform(data[self.column_name].values)


class TfIdfVectorizerDF:
    def __init__(self, column_name: str, **kwargs: Any):
        self.tfidf_vectorizer: TfidfVectorizer = TfidfVectorizer(**kwargs)
        self.column_name: str = column_name

    def fit(
        self,
        data: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> TfIdfVectorizerDF:
        # y: Optional[Union[pd.Series, np.ndarray]]
        # необходим по требования Pipeline
        self.tfidf_vectorizer.fit(data[self.column_name].values)
        return self

    def transform(self, data: pd.DataFrame) -> Any:
        return self.tfidf_vectorizer.transform(data[self.column_name].values)
