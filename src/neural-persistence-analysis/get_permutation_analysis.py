import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from dnp import get_dnp
from typing import Dict
from copy import deepcopy
from typing import OrderedDict
from metrics import get_sortedness
from collections import defaultdict
from metrics import get_neural_persistence
from analysis_utils import get_model_weights
from analysis_utils import parse_experiment_arguments


def make_permutation(
    weights: OrderedDict[int, np.ndarray]
) -> OrderedDict[int, np.ndarray]:
    # Flatten weights
    flattened_weights = {
        layer_index: w.reshape(-1).copy() for layer_index, w in weights.items()
    }

    # Shuffle weights and unflatten
    shuffled_weights = OrderedDict()
    for layer_index, w in flattened_weights.items():
        np.random.shuffle(w)
        shuffled_weights[layer_index] = w.reshape(weights[layer_index].shape)

    return shuffled_weights


def get_metrics(weights: OrderedDict[int, np.ndarray]) -> Dict[str, float]:
    # Get metrics
    metrics = dict()

    # Sortedness
    sortedness = get_sortedness(weights=weights)
    for layer_index, sortedness_value in enumerate(sortedness):
        metrics[f"layer_{layer_index}_sortedness"] = sortedness_value

    # Neural persistence
    global_persistence, local_persistences = get_neural_persistence(weights=weights)
    metrics["neural_persistence"] = global_persistence
    for layer_index, local_persistence in enumerate(local_persistences):
        metrics[f"layer_{layer_index}_persistence"] = local_persistence

    # DNP
    for key, value in get_dnp(weights=weights).items():
        metrics[key] = value

    # Return metrics
    return metrics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_permutations", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Parse experiment arguments
    experiment_arguments = parse_experiment_arguments(args.path)

    # Initialize results
    permutation_results = []
    permutation_results_expected = []

    # Get model weight distributions for all runs
    for run_directory in tqdm(os.listdir(args.path)):
        run_id = int(run_directory.split("_")[-1])
        run_path = os.path.join(args.path, run_directory, "last.ckpt")

        model_weights = get_model_weights(run_path)
        unpermuted_metrics = get_metrics(model_weights)
        model_permutation_results = []

        for permutation_index in range(args.num_permutations):
            # Make permutation
            permutation = make_permutation(model_weights)

            # Compute metrics for permuted weights
            permutation_metrics = get_metrics(permutation)
            permutation_metrics = {
                f"permuted_{key}": value for key, value in permutation_metrics.items()
            }
            permutation_metrics_delta = {
                f"{key}_delta": value - unpermuted_metrics[key[len("permuted_") :]]
                for key, value in permutation_metrics.items()
            }
            permutation_metrics.update(permutation_metrics_delta)

            # Add original metrics
            permutation_metrics.update(deepcopy(unpermuted_metrics))

            # Add permutation index
            permutation_metrics["permutation_index"] = permutation_index

            # Save permutation metrics
            model_permutation_results.append(permutation_metrics)

        # Get distribution of permutation metrics
        aggregated_permutation_metrics = defaultdict(list)
        for permutation_metrics in model_permutation_results:
            for key, value in permutation_metrics.items():
                aggregated_permutation_metrics[key].append(value)

        expected_permutation_metrics = dict()
        for key, values in aggregated_permutation_metrics.items():
            values = np.array(values)
            expected_permutation_metrics[f"{key}_expected"] = np.mean(values)
            expected_permutation_metrics[f"{key}_std"] = np.std(values)
            expected_permutation_metrics[f"{key}_min"] = np.min(values)
            expected_permutation_metrics[f"{key}_max"] = np.max(values)
            expected_permutation_metrics[f"{key}_abs_expected"] = np.mean(
                np.abs(values)
            )

        # Save
        experiment_arguments_copy = deepcopy(experiment_arguments)
        experiment_arguments_copy["run_id"] = run_id

        expected_permutation_metrics.update(experiment_arguments_copy)
        permutation_results_expected.append(expected_permutation_metrics)

        # Save permutation results
        for permutation_metrics in model_permutation_results:
            permutation_metrics.update(experiment_arguments_copy)
            permutation_results.append(permutation_metrics)

    # Save results
    permutation_results_df = pd.DataFrame.from_records(permutation_results)
    permutation_results_expected_df = pd.DataFrame.from_records(
        permutation_results_expected
    )

    save_file_name = "_".join(
        [f"{key}={value}" for key, value in experiment_arguments.items()]
    )
    permutation_results_save_file_name = f"permutation_results_{save_file_name}.csv"
    permutation_result_save_path = os.path.join(
        args.output_path, permutation_results_save_file_name
    )

    permutation_results_expected_save_file_name = (
        f"expected_permutation_results_expected_{save_file_name}.csv"
    )
    expected_permutation_results_save_path = os.path.join(
        args.output_path, permutation_results_expected_save_file_name
    )

    os.makedirs(args.output_path, exist_ok=True)
    permutation_results_df.to_csv(permutation_result_save_path, index=False)
    permutation_results_expected_df.to_csv(
        expected_permutation_results_save_path, index=False
    )
