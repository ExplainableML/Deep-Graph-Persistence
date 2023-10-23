import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List
from typing import Dict
from itertools import product
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from witness_functions import distance_from_center_by_groups


def multidimensional_ks_test(
    x_clean: np.ndarray, x_test: np.ndarray, alpha: float = 0.05
) -> bool:
    # x_clean: shape [n_samples, n_features]
    # x_test: shape [n_samples, n_features]
    num_features = x_clean.shape[1]
    assert num_features == x_test.shape[1], "Number of features must be equal"

    # Perform KS test for each feature separately
    p_vals = []
    for k in range(num_features):
        feature_p_value = ks_2samp(x_clean[:, k], x_test[:, k]).pvalue
        p_vals.append(feature_p_value)

    # Adjust alpha for multiple testing (Bonferroni correction)
    alpha = alpha / num_features

    # Return True if any p-value is smaller than alpha
    reject = min(p_vals) < alpha
    return reject


def get_difference_to_train_mean(
    x_train: np.ndarray, y_train: np.ndarray, x_clean: np.ndarray, x_test: np.ndarray
) -> bool:
    # x_train: shape [n_samples, n_features]
    # y_train: shape [n_samples]
    # x_clean: shape [n_samples, n_features]
    # x_test: shape [n_samples, n_features]

    # Get group-wise differences for all clean samples
    clean_differences = distance_from_center_by_groups(x_train, y_train, x_clean)

    # Get group-wise differences for all test samples
    test_differences = distance_from_center_by_groups(x_train, y_train, x_test)

    # Perform KS test for each feature separately
    return multidimensional_ks_test(clean_differences, test_differences)


def get_metrics(x_train, y_train, x_clean, x_test):
    # x_train: shape [n_samples, n_features]
    # y_train: shape [n_samples]
    # x_clean: shape [n_samples, n_features]
    # x_test: shape [n_samples, n_features]
    if isinstance(x_train, np.ndarray):
        num_features = x_train.shape[1]
        assert num_features == x_clean.shape[1], "Number of features must be equal"
        assert num_features == x_test.shape[1], "Number of features must be equal"
    else:
        num_features = np.inf

    metrics = dict()

    # Difference to Train Mean
    difference_to_train_mean = get_difference_to_train_mean(
        x_train, y_train, x_clean, x_test
    )
    metrics["difference_to_train_mean"] = difference_to_train_mean

    # If the number of features is sufficiently small, we can also perform a KS test
    # (this mainly applies to logit and softmax baselines)
    if num_features < 100:
        # KS Test
        ks_test = multidimensional_ks_test(x_clean, x_test)
        metrics["ks_test"] = ks_test

    return metrics


def _bootstrap_iteration(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_clean: np.ndarray,
    x_test: np.ndarray,
    num_train_samples: int,
    num_test_samples: int,
):
    # Helper function to select samples
    def select_samples(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x[indices, :].copy()
        elif isinstance(x, list):
            return [x_i[indices, :].copy() for x_i in x]
        else:
            raise TypeError(f"Expected x to be np.ndarray or list, got {type(x)}")

    # Get number of samples
    def _get_num_samples(x):
        if isinstance(x, np.ndarray):
            return x.shape[0]
        elif isinstance(x, list):
            return x[0].shape[0]

    num_train_samples_total = _get_num_samples(x_train)
    num_test_samples_total = _get_num_samples(x_test)
    num_clean_samples_total = _get_num_samples(x_clean)

    # Sample training data
    train_indices = np.random.choice(
        num_train_samples_total, size=(num_train_samples,), replace=False
    )
    x_train = select_samples(x_train, train_indices)
    y_train = y_train[train_indices]

    # Sample test data
    test_indices = np.random.choice(
        num_test_samples_total, size=(num_test_samples,), replace=False
    )
    x_test = select_samples(x_clean, test_indices)

    # Sample clean data
    clean_indices = np.random.choice(
        num_clean_samples_total, size=(num_test_samples,), replace=False
    )
    x_clean = select_samples(x_clean, clean_indices)

    # Get metrics
    metrics = get_metrics(x_train, y_train, x_clean, x_test)
    return metrics


def bootstrap(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_clean: np.ndarray,
    x_test: np.ndarray,
    num_train_samples: int,
    num_test_samples: int,
    iterations: int,
) -> List[Dict[str, float]]:
    metrics = []

    for iteration in range(iterations):
        iteration_metrics = _bootstrap_iteration(
            x_train,
            y_train,
            x_clean,
            x_test,
            num_train_samples,
            num_test_samples,
        )
        iteration_metrics["iteration"] = iteration
        iteration_metrics["num_train_samples"] = num_train_samples
        iteration_metrics["num_test_samples"] = num_test_samples
        metrics.append(iteration_metrics)

    return metrics


def load_witness_vectors(
    dataset: str,
    hidden: int,
    layers: int,
    run: int,
    shift: str,
    intensity: int,
    method: str,
):
    root_paths = [
        "./saved_data/witness_vectors",
    ]
    for root_path in root_paths:
        witness_vector_path = os.path.join(
            root_path,
            dataset,
            f"data={dataset}-hidden={hidden}-layers={layers}-run={run}",
            shift,
            method,
            f"intensity={intensity}-train=1000.npy",
        )

        try:
            data = np.load(witness_vector_path, allow_pickle=True)
            data = data.item()
            break
        except FileNotFoundError:
            continue
    else:
        parameters = ", ".join(
            [dataset, str(hidden), str(layers), str(run), shift, str(intensity)]
        )
        raise FileNotFoundError(f"Could not find witness vectors: {parameters}")

    x_train = data["witness_vectors_train"]
    y_train = data["y_train"]
    x_clean = data["witness_vectors_clean"]
    x_test = data["witness_vectors_shifted"]

    # if isinstance(x_train, list):
    #    x_train = x_train[-1]
    # if isinstance(x_clean, list):
    #    x_clean = x_clean[-1]
    # if isinstance(x_test, list):
    #    x_test = x_test[-1]

    return x_train, y_train, x_clean, x_test


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Witness Vectors")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
    parser.add_argument("--hidden", type=int, required=True, help="Hidden dimension")
    parser.add_argument("--layers", type=int, required=True, help="Num Layers")
    parser.add_argument("--run", type=int, required=True, help="Run")
    parser.add_argument("--shift", type=str, required=True, help="Shift")
    parser.add_argument("--intensity", type=int, help="Intensity")
    parser.add_argument("--method", type=str, required=True, help="Method")

    args = parser.parse_args()
    return args


def evaluation() -> None:
    args = parse_arguments()

    # Prepare save path
    result_save_path = os.path.join(
        "./calibration_results",
        args.dataset,
        args.method,
        args.shift,
        str(args.intensity),
    )
    os.makedirs(result_save_path, exist_ok=True)
    result_file_name = f"hidden={args.hidden}-layers={args.layers}-run={args.run}.csv"

    # Load witness vectors
    x_train, y_train, x_clean, x_test = load_witness_vectors(
        dataset=args.dataset,
        hidden=args.hidden,
        layers=args.layers,
        run=args.run,
        shift=args.shift,
        intensity=args.intensity,
        method=args.method,
    )

    train_samples_sizes = [1000]
    test_sample_sizes = [10, 20, 50, 100, 200]
    iterations = 100

    results = []

    grid = list(product(train_samples_sizes, test_sample_sizes))
    for train_sample_size, test_sample_size in tqdm(grid):
        metrics = bootstrap(
            x_train,
            y_train,
            x_clean,
            x_test,
            train_sample_size,
            test_sample_size,
            iterations,
        )
        results.extend(metrics)

    results_df = pd.DataFrame.from_records(results)
    # Make hyperparameters columns
    for key, value in vars(args).items():
        results_df[key] = value

    # Save results
    os.makedirs(result_save_path, exist_ok=True)
    results_df.to_csv(os.path.join(result_save_path, result_file_name), index=False)


if __name__ == "__main__":
    evaluation()
