from typing import Callable, Dict

from sklearn import metrics as mcs

opt_metrics: Dict[str, Callable] = {
    "precision": mcs.precision_score,
    "recall": mcs.recall_score,
    "f1": mcs.f1_score,
}
