from __future__ import annotations

from typing import Any, List, Optional, Union

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
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


class TextPreprocessTransformerDF:
    def __init__(self, column_name: str, **kwargs: Any):
        self.column_name: str = column_name
        self.stop_words: List[str] = kwargs.get("stop_words", [])

    def fit(
        self,
        data: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> TextPreprocessTransformerDF:
        # y: Optional[Union[pd.Series, np.ndarray]]
        # необходим по требования Pipeline
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.column_name] = data[self.column_name].apply(
            lambda sent: self.process_text(sent, self.stop_words)
        )
        return data

    @classmethod
    def process_text(cls, text: str, stop_words: List[str]) -> str:
        return " ".join(
            [word for word in text.split() if word not in stop_words]
        )


class IdentityTransformer:
    def __init__(self, column_name: str, **kwargs: Any):
        self.column_name: str = column_name

    def fit(
        self,
        data: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> IdentityTransformer:
        # y: Optional[Union[pd.Series, np.ndarray]]
        # необходим по требования Pipeline
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[[self.column_name]]


class LaBSEVectorizer:

    def __init__(self, column_name: str, **kwargs: Any):
        self.column_name: str = column_name
        path_to_model: str = kwargs.get(
            "path_to_model",
            'sentence-transformers/LaBSE'
        )
        self.model: SentenceTransformer = SentenceTransformer(
            path_to_model
        )

    def fit(
        self,
        data: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> IdentityTransformer:
        # y: Optional[Union[pd.Series, np.ndarray]]
        # необходим по требования Pipeline
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        text_list: List[str] = [text for text in data[self.column_name].values]
        return self.model.encode(text_list, show_progress_bar=True)
