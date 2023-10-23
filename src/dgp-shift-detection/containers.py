import numpy as np

from dataclasses import dataclass


@dataclass
class EvaluationData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_clean: np.ndarray
    y_clean: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
