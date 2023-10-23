import torch
import gudhi as gd
import numpy as np

from mlp import MLP
from typing import List
from torch import Tensor


def get_mlp_activations(model: MLP, x: Tensor) -> List[Tensor]:
    activations = []
    x = x.reshape(x.shape[0], -1)

    # Get activations
    for linear, nonlinearity in zip(model.linear_layers, model.nonlinearities):
        activations.append(x)
        x = linear(x)
        x = nonlinearity(x)

    return activations


def build_mlp_activation_graph(model: MLP, x: Tensor) -> Tensor:
    activations = []
    weights = []
    x = x.reshape(x.shape[0], -1)

    # Get activations and weights
    for linear, nonlinearity in zip(model.linear_layers, model.nonlinearities):
        activations.append(x)
        weights.append(linear.weight)
        x = linear(x)
        x = nonlinearity(x)

    # Build activation graph
    activation_graph = []
    for activation, weight in zip(activations, weights):
        activation_graph.append(activation.unsqueeze(1) * weight.unsqueeze(0))

    return activation_graph


def build_dnp_matrix(weights: List[np.ndarray]) -> Tensor:
    weights = [weight for weight in reversed(weights)]
    collapsed_weights = weights[0]

    for weight in weights[1:]:
        collapsed_weights_expanded = np.expand_dims(collapsed_weights, 1)
        weight_expanded = np.expand_dims(weight, -1)

        collapsed_weights = np.minimum(collapsed_weights_expanded, weight_expanded)
        collapsed_weights = np.max(collapsed_weights, axis=2)

    return torch.from_numpy(collapsed_weights).float()


def get_mst_weights_from_bipartite_graph(weights: np.ndarray) -> np.ndarray:
    """Taken from TU code: https://github.com/tlacombe/topologicalUncertainty/"""
    # Build simpex tree
    G = gd.SimplexTree()

    for i in np.arange(0, weights.shape[0] + weights.shape[1]):
        G.insert([i], filtration=-np.inf)

    for i in np.arange(0, weights.shape[0]):
        for j in np.arange(weights.shape[0], weights.shape[0] + weights.shape[1]):
            G.insert([i, j], filtration=-np.abs(weights[i, j - weights.shape[0]]))

    # Get MST
    G.compute_persistence(min_persistence=-1.0)
    dgm0 = G.persistence_intervals_in_dimension(0)[:, 1]
    dgm0 = -dgm0[np.where(np.isfinite(dgm0))]

    return dgm0


def get_approximate_mst_weights_from_bipartite_graph(weights: np.ndarray) -> np.ndarray:
    max_over_rows = np.max(weights, axis=1)
    max_over_cols = np.max(weights, axis=0)
    return np.concatenate([max_over_rows, max_over_cols])
