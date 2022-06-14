import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

Dataset = Union[pd.DataFrame, np.ndarray]
Target = Union[pd.DataFrame, np.ndarray]


class SklearnRTKDataModule:
    def __init__(
        self,
        target_column: str,
        data_columns: List[str],
        data_dir: str = "data/raw/",
        path_to_test: str = "data/raw/",
        stratify: bool = False,
        test_size: float = 0.3,
        shuffle: bool = True,
        random_state: int = 100500,
    ):
        self.target_column: str = target_column
        self.data_columns: List[str] = data_columns
        self.data_dir: str = os.path.join(
            os.environ["PROJECT_PATH_ROOT"], data_dir
        )
        self.path_to_test: str = os.path.join(
            os.environ["PROJECT_PATH_ROOT"], path_to_test
        )
        self.stratify: bool = stratify
        self.test_size: float = test_size
        self.shuffle: bool = shuffle
        self.random_state: int = random_state

        self.train_data: Optional[Tuple[Dataset, Target]] = None
        self.val_data: Optional[Tuple[Dataset, Target]] = None
        self.test_data: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or not stage:
            data_train: pd.DataFrame = pd.read_excel(
                os.path.join(self.data_dir, "train.xlsx")
            )
            target: Target = data_train[self.target_column]
            data_train.drop(self.target_column, inplace=True, errors="ignore")
            trn_idx, val_idx = train_test_split(
                data_train.index.values,
                stratify=target if self.stratify else None,
                shuffle=self.shuffle,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            x_train: Dataset = data_train.iloc[trn_idx]
            y_train: Target = target.iloc[trn_idx]
            x_val: Dataset = data_train.iloc[val_idx]
            y_val: Target = target.iloc[val_idx]

            self.train_data = (x_train[self.data_columns], y_train)
            self.val_data = (x_val[self.data_columns], y_val)
        elif stage == "test":
            self.test_data = pd.read_excel(
                os.path.join(self.data_dir, "test.xlsx")
            )

    def get_train_data(self) -> Tuple[Dataset, Target]:
        return self.train_data

    def get_val_data(self) -> Tuple[Dataset, Target]:
        return self.val_data

    def get_test_data(self) -> Dataset:
        return self.test_data
