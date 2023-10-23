import os
import pandas as pd

from tqdm import tqdm
from copy import deepcopy
from analysis_utils import parse_experiment_arguments

path_to_logs = "./logs"
path_to_permutation_analysis = "./permutation_analysis"
path_to_weight_distribution_analysis = "./weight_distributions"

if __name__ == "__main__":
    # Load Experiment Logs
    experiment_logs = []

    for experiment in tqdm(os.listdir(path_to_logs)):
        experiment_parameters = parse_experiment_arguments(experiment)

        experiment_path = os.path.join(path_to_logs, experiment)
        for run in os.listdir(experiment_path):
            # Read log file
            log_file = os.path.join(
                experiment_path, run, experiment, "version_0", "metrics.csv"
            )
            df = pd.read_csv(log_file)

            # Sort by step
            df = df.sort_values(by="step", ignore_index=True)
            df = df.reset_index(drop=False)

            # Add experiment parameters
            experiment_parameters_copy = deepcopy(experiment_parameters)
            experiment_parameters_copy["run"] = int(run.split("_")[-1])
            experiment_parameters_copy = pd.DataFrame.from_records(
                [experiment_parameters_copy]
            )
            experiment_parameters_copy = pd.concat(
                [experiment_parameters_copy] * len(df), ignore_index=True
            )
            df = pd.concat([experiment_parameters_copy, df], axis=1)

            # Save experiment logs
            experiment_logs.append(df)

    experiment_logs = pd.concat(experiment_logs, ignore_index=True)
    experiment_logs.to_csv("./experiment_logs.csv", index=False)

    # Load Permutation Analysis Logs
    permutation_analysis_logs = []
    permutation_analysis_expected_logs = []

    for result_file in tqdm(os.listdir(path_to_permutation_analysis)):
        results = pd.read_csv(os.path.join(path_to_permutation_analysis, result_file))

        if "expected" in result_file:
            permutation_analysis_expected_logs.append(results)
        else:
            permutation_analysis_logs.append(results)

    permutation_analysis_logs = pd.concat(permutation_analysis_logs, ignore_index=True)
    permutation_analysis_logs.to_csv("./permutation_analysis_logs.csv", index=False)

    permutation_analysis_expected_logs = pd.concat(
        permutation_analysis_expected_logs, ignore_index=True
    )
    permutation_analysis_expected_logs.to_csv(
        "./permutation_analysis_expected_logs.csv", index=False
    )

    # Load Weight Distribution Analysis Logs
    weight_distribution_analysis_logs = []

    for result_file in tqdm(os.listdir(path_to_weight_distribution_analysis)):
        results = pd.read_csv(
            os.path.join(path_to_weight_distribution_analysis, result_file)
        )
        weight_distribution_analysis_logs.append(results)

    weight_distribution_analysis_logs = pd.concat(
        weight_distribution_analysis_logs, ignore_index=True
    )
    weight_distribution_analysis_logs.to_csv(
        "./weight_distribution_analysis_logs.csv", index=False
    )
