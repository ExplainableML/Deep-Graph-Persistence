import os
import torch
import random
import argparse
import numpy as np

from tqdm import tqdm
from load_data import load_dataset
from data_shifts import registered_shifts
from shift_configurations import registered_configurations


assert all(shift_name in registered_configurations for shift_name in registered_shifts)


def generate_shifts(dataset_name: str, x: np.ndarray, y: np.ndarray):
    # Initialize shifted data dictionary
    shifts = {}
    shift_generators = tqdm(
        registered_shifts.items(),
        desc="Generating Shifts",
        total=len(registered_shifts),
    )

    # Iterate over all shifts
    for shift_name, shift_generator in shift_generators:
        shifts[shift_name] = []
        # Load configurations for shift
        configurations = registered_configurations[shift_name](dataset_name)

        # Iterate over all configurations
        for configuration in tqdm(
            configurations, leave=False, desc=f"Generating {shift_name} Shifts"
        ):
            # Generate shifted data
            x_shift, y_shift = shift_generator(x, y, **configuration)

            # Save shifted data
            shifts[shift_name].append(
                {
                    "shift_name": shift_name,
                    "configuration": configuration,
                    "x": x_shift,
                    "y": y_shift,
                }
            )

    return shifts


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Make Data Shifts")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num-train-samples", type=int, required=True)
    parser.add_argument("--num-test-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    # Parse Arguments
    arguments = parse_arguments()
    root_path = os.path.join("./saved_data", "shifted_data", arguments.dataset)

    # Set Seed
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    random.seed(arguments.seed)

    # Load Data
    data = load_dataset(
        dataset_name=arguments.dataset,
        num_train_samples=arguments.num_train_samples,
        num_test_samples=arguments.num_test_samples,
    )

    # Get Shifted Data
    shifted_data = generate_shifts(
        dataset_name=arguments.dataset,
        x=data.x_clean,
        y=data.y_clean,
    )

    # Combine Data
    data = {
        "parameters": {
            "dataset_name": arguments.dataset,
            "num_train_samples": arguments.num_train_samples,
            "num_test_samples": arguments.num_test_samples,
            "seed": arguments.seed,
        },
        "x_train": data.x_train,
        "y_train": data.y_train,
        "x_clean": data.x_clean,
        "y_clean": data.y_clean,
        "shifted_data": shifted_data,
    }

    # Save Data
    os.makedirs(root_path, exist_ok=True)
    dataset_file_name = (
        f"train={arguments.num_train_samples}_"
        + f"test={arguments.num_test_samples}_"
        + f"seed={arguments.seed}.npy"
    )
    dataset_file_name = os.path.join(root_path, dataset_file_name)

    np.save(dataset_file_name, data, allow_pickle=True)
