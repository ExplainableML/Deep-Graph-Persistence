import os
import argparse
import numpy as np
import pandas as pd

from typing import Dict
from typing import Tuple
from copy import deepcopy
from scipy.stats import beta
from scipy.stats import truncnorm
from scipy.stats import truncexpon
from scipy.stats import truncpareto
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from analysis_utils import get_model_weights
from analysis_utils import parse_experiment_arguments


def fit_beta(m: np.ndarray) -> Tuple[float, float, float]:
    # Clip values to avoid numerical issues
    vals = np.clip(m.reshape(-1), 1e-6, 1 - 1e-6)

    # Define loss function for beta distribution
    def loss_beta(args):
        return -beta.logpdf(vals, a=np.exp(args[0]), b=np.exp(args[1])).sum()

    # Find best beta parameters
    best_beta_params = minimize(
        loss_beta, x0=np.array([0.0, 0.0]), method="Nelder-Mead"
    )

    # Apply exponential to get beta parameters
    a = np.exp(best_beta_params.x[0])
    b = np.exp(best_beta_params.x[1])

    # Evaluate loss
    loss = loss_beta(best_beta_params.x)

    return a, b, loss


def fit_truncpareto(m: np.ndarray) -> Tuple[float, float]:
    # Clip values to avoid numerical issues
    vals = np.clip(m.reshape(-1), 0.0, 1.0)

    # Define loss function for truncated pareto distribution
    def loss_truncpareto(log_b: float):
        return -truncpareto.logpdf(vals, b=np.exp(log_b), c=2.0, loc=-1.0).sum()

    # Find best parameters
    best_pareto_params = minimize_scalar(loss_truncpareto)
    b = np.exp(best_pareto_params.x)
    loss = best_pareto_params.fun

    return b, loss


def fit_truncnorm(m: np.ndarray) -> Tuple[float, float]:
    # Clip values to avoid numerical issues
    vals = np.clip(m.reshape(-1), 0.0, 1.0)

    # Define loss function for truncated normal distribution
    def loss_truncnorm(log_scale: float):
        scale = np.exp(log_scale)
        return -truncnorm.logpdf(
            vals, a=0.0, b=(1.0 / scale), loc=0.0, scale=scale
        ).sum()

    # Find best parameters
    best_truncnorm_params = minimize_scalar(loss_truncnorm)
    scale = np.exp(best_truncnorm_params.x)
    loss = best_truncnorm_params.fun

    return scale, loss


def fit_truncexpon(m: np.ndarray) -> Tuple[float, float]:
    # Clip values to avoid numerical issues
    vals = np.clip(m.reshape(-1), 0.0, 1.0)

    # Define loss function for truncated exponential distribution
    def loss_truncexpon(log_scale: float):
        scale = np.exp(log_scale)
        return -truncexpon.logpdf(vals, b=(1.0 / scale), loc=0.0, scale=scale).sum()

    # Find best parameters
    best_truncexpon_params = minimize_scalar(loss_truncexpon)
    scale = np.exp(best_truncexpon_params.x)
    loss = best_truncexpon_params.fun

    return scale, loss


def get_weight_distributions_of_matrix(weights: np.ndarray) -> Dict[str, float]:
    beta_a, beta_b, beta_loss = fit_beta(weights)
    pareto_b, pareto_loss = fit_truncpareto(weights)
    norm_scale, norm_loss = fit_truncnorm(weights)
    exp_scale, exp_loss = fit_truncexpon(weights)

    return {
        "beta_a": beta_a,
        "beta_b": beta_b,
        "beta_loss": beta_loss,
        "pareto_b": pareto_b,
        "pareto_loss": pareto_loss,
        "norm_scale": norm_scale,
        "norm_loss": norm_loss,
        "exp_scale": exp_scale,
        "exp_loss": exp_loss,
    }


def get_weight_distributions_of_model(path: str) -> Dict[str, float]:
    # Load model
    weights = get_model_weights(path)
    # Normalize weights
    weights = {layer_index: np.abs(weight) for layer_index, weight in weights.items()}
    max_abs_weight = max([np.max(weight) for weight in weights.values()])
    weights = {
        layer_index: weight / max_abs_weight for layer_index, weight in weights.items()
    }

    # Get weight distributions
    weight_distributions = {}
    for layer_index, weight in weights.items():
        weight_distributions[layer_index] = get_weight_distributions_of_matrix(weight)

    return weight_distributions


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Parse experiment arguments
    experiment_arguments = parse_experiment_arguments(args.path)

    # Initialize results
    results = []

    # Get model weight distributions for all runs
    for run_directory in os.listdir(args.path):
        run_id = int(run_directory.split("_")[-1])
        run_path = os.path.join(args.path, run_directory, "last.ckpt")

        # Get model weight distributions
        weight_distributions = get_weight_distributions_of_model(run_path)

        # Save model weight distributions
        for layer_index, weight_distribution in weight_distributions.items():
            arguments_copy = deepcopy(experiment_arguments)
            arguments_copy["run_id"] = run_id
            arguments_copy["layer_index"] = layer_index
            arguments_copy.update(weight_distribution)
            results.append(arguments_copy)

    # Save results
    df = pd.DataFrame.from_records(results)

    save_file_name = "_".join(
        [f"{key}={value}" for key, value in experiment_arguments.items()]
    )
    save_file_name = f"weight_distributions_{save_file_name}.csv"
    save_path = os.path.join(args.output_path, save_file_name)
    os.makedirs(args.output_path, exist_ok=True)
    df.to_csv(save_path, index=False)
