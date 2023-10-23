import os
import torch

from mlp import MLP
from typing import Dict
from collections import OrderedDict


def parse_experiment_arguments(path: str) -> Dict[str, str]:
    # Split model name
    path = os.path.normpath(path)
    path = path.split(os.sep)[-1]

    # Split key-value pairs
    key_value_pairs = path.split("_")
    key_value_pairs = [key_value_pair.split("=") for key_value_pair in key_value_pairs]
    parameters = {key: value for key, value in key_value_pairs}
    return parameters


def get_model_weights(path: str) -> OrderedDict:
    # Load model
    model = MLP.load_from_checkpoint(path, map_location=torch.device("cpu"))
    model.eval()

    # Get model weights
    return model.get_weights()
