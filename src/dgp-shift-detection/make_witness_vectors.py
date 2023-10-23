import os
import torch
import argparse
import numpy as np

from mlp import MLP
from tqdm import trange
from typing import Union
from torch import Tensor
from functools import partial

from witness_functions import dnp_witness_function
from witness_functions import logit_witness_function
from witness_functions import softmax_witness_function
from witness_functions import magdiff_witness_function
from witness_functions import input_activation_witness_function
from witness_functions import topological_uncertainty_witness_function


all_witness_functions = {
    "softmax": softmax_witness_function,
    "logit": logit_witness_function,
    "magdiff": magdiff_witness_function,
    "dgp": partial(dnp_witness_function, variance_correction=False, normalize=False),
    "dgp_normalize": partial(
        dnp_witness_function, variance_correction=False, normalize=True
    ),
    "dgp_var": partial(dnp_witness_function, variance_correction=True, normalize=False),
    "dgp_var_normalize": partial(
        dnp_witness_function, variance_correction=True, normalize=True
    ),
    "dgp_approximate": partial(
        dnp_witness_function,
        variance_correction=False,
        normalize=False,
        approximate=True,
    ),
    "dgp_var_approximate": partial(
        dnp_witness_function,
        variance_correction=True,
        normalize=False,
        approximate=True,
    ),
    "topological": topological_uncertainty_witness_function,
    "topological_var": partial(
        topological_uncertainty_witness_function, variance_correction=True
    ),
    "input_activation": input_activation_witness_function,
}


def call_witness_function(
    x: Union[Tensor, np.ndarray], witness_function, batch_size: int
) -> Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    witness_vectors = []
    for batch_start in trange(0, x.shape[0], batch_size, leave=False):
        batch_end = batch_start + batch_size
        batch_x = x[batch_start:batch_end]
        with torch.no_grad():
            witness_vectors.append(witness_function(x=batch_x))
    if all([isinstance(w, np.ndarray) for w in witness_vectors]):
        return np.concatenate(witness_vectors, axis=0)
    else:
        return [
            np.concatenate(layer_witness_vectors, axis=0)
            for layer_witness_vectors in zip(*witness_vectors)
        ]


def load_mlp_model(path: str) -> MLP:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MLP.load_from_checkpoint(path, map_location=device)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Make Witness Vectors")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--shift", type=str, required=True)
    parser.add_argument("--intensity", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--method", type=str, default="softmax", required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse Arguments
    arguments = parse_arguments()

    # Load Model
    model_path = arguments.model_path
    if not model_path.endswith(".ckpt"):
        model_path = os.path.join(model_path, "last.ckpt")

    model = load_mlp_model(model_path)

    # Load Data
    data = np.load(arguments.dataset, allow_pickle=True)
    data = data.item()

    # Extract dataset parameters
    dataset_name = data["parameters"]["dataset_name"]
    num_train_samples = data["parameters"]["num_train_samples"]
    num_test_samples = data["parameters"]["num_test_samples"]

    if arguments.train_samples is not None:
        num_train_samples = arguments.train_samples

    # Extract Train Data
    x_train = data["x_train"][:num_train_samples]
    y_train = data["y_train"][:num_train_samples]

    # Extract Clean Test Data
    x_clean = data["x_clean"]
    y_clean = data["y_clean"]

    # Extract Shifted Test Data
    shifted_data = data["shifted_data"][arguments.shift][arguments.intensity]

    # Get Witness Function
    witness_function = all_witness_functions[arguments.method]
    witness_function = partial(witness_function, model=model)

    # Get Train Witness Vectors
    witness_vectors_train = call_witness_function(
        x_train,
        witness_function=witness_function,
        batch_size=arguments.batch_size,
    )

    # Get Clean Witness Vectors
    witness_vectors_clean = call_witness_function(
        x_clean, witness_function=witness_function, batch_size=arguments.batch_size
    )

    # Get Shifted Witness Vectors
    witness_vectors_shifted = call_witness_function(
        x=shifted_data["x"],
        witness_function=witness_function,
        batch_size=arguments.batch_size,
    )

    # Save Data
    save_path = os.path.join(
        "./saved_data",
        f"witness_vectors",
        dataset_name,
        arguments.model_name,
        arguments.shift,
        arguments.method,
    )
    os.makedirs(save_path, exist_ok=True)

    save_file_name = f"intensity={arguments.intensity}-train={num_train_samples}.npy"
    save_path = os.path.join(save_path, save_file_name)

    save_data = {
        "parameters": {
            "dataset_name": dataset_name,
            "num_train_samples": num_train_samples,
            "num_test_samples": num_test_samples,
            "model_name": arguments.model_name,
            "shift": arguments.shift,
            "intensity": arguments.intensity,
            "seed": data["parameters"]["seed"],
            "method": arguments.method,
        },
        "witness_vectors_train": witness_vectors_train,
        "witness_vectors_clean": witness_vectors_clean,
        "witness_vectors_shifted": witness_vectors_shifted,
        "y_train": y_train,
        "y_clean": y_clean,
        "y_shifted": y_clean,
    }

    np.save(save_path, save_data, allow_pickle=True)
