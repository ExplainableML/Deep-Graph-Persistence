import torch
import numpy as np

from mlp import MLP
from typing import List
from torch import Tensor
from utils import build_dnp_matrix
from collections.abc import Iterable
from utils import get_mlp_activations
from utils import build_mlp_activation_graph
from utils import get_mst_weights_from_bipartite_graph
from utils import get_approximate_mst_weights_from_bipartite_graph


def logit_witness_function(model: MLP, x: Tensor) -> np.ndarray:
    return model(x).detach().cpu().numpy()


def softmax_witness_function(model: MLP, x: Tensor) -> np.ndarray:
    return torch.softmax(model(x), dim=1).detach().cpu().numpy()


def magdiff_witness_function(model: MLP, x: Tensor) -> np.ndarray:
    activation_graph = build_mlp_activation_graph(model, x)
    activation_graph = activation_graph[-1]
    activation_graph = activation_graph.reshape(activation_graph.shape[0], -1)
    activation_graph = activation_graph.detach().cpu().numpy()
    return activation_graph


def dnp_witness_function(
    model: MLP,
    x: Tensor,
    variance_correction: bool,
    normalize: bool = False,
    approximate: bool = False,
) -> np.ndarray:
    # Build activation graph
    activation_graph = build_mlp_activation_graph(model, x)
    num_layers = len(activation_graph)

    # Normalize activation graph
    def normalize_layer(layer: Tensor) -> Tensor:
        if variance_correction:
            layer = layer - layer.mean()
            layer = layer / (layer.std() + 1e-8)

        layer = torch.abs(layer)
        return layer

    # Split into samples
    activation_graph = [
        [activations[sample_index] for activations in activation_graph]
        for sample_index in range(x.shape[0])
    ]

    # Normalize each layer
    activation_graph = [
        [normalize_layer(layer) for layer in sample_layers]
        for sample_layers in activation_graph
    ]

    # Map to [0, 1]
    if normalize:
        maximum_activations = [
            np.max([layer.max() for layer in sample_layers])
            for sample_layers in activation_graph
        ]
        activation_graph = [
            [layer / maximum_activation for layer in sample_layers]
            for sample_layers, maximum_activation in zip(
                activation_graph, maximum_activations
            )
        ]

    # Recombine samples
    activation_graph = [
        torch.stack(
            [
                sample_activations[layer_index]
                for sample_activations in activation_graph
            ],
        ).transpose(1, 2)
        for layer_index in range(num_layers)
    ]

    # Build DNP matrix
    activation_graph = [layer.detach().cpu().numpy() for layer in activation_graph]
    dnp_matrix = build_dnp_matrix(activation_graph)

    # Get MST weights from each sample individually
    mst_weights = []
    for sample_dnp_matrix in dnp_matrix:
        sample_dnp_matrix = sample_dnp_matrix.detach().cpu().numpy()
        if approximate:
            sample_mst_weights = get_approximate_mst_weights_from_bipartite_graph(
                sample_dnp_matrix
            )
        else:
            sample_mst_weights = get_mst_weights_from_bipartite_graph(sample_dnp_matrix)
        mst_weights.append(sample_mst_weights)

    mst_weights = np.stack(mst_weights)
    return mst_weights


def input_activation_witness_function(model: MLP, x: Tensor) -> List[np.ndarray]:
    activations = get_mlp_activations(model, x)
    activations = activations[1:]
    activations = [activation.detach().cpu().numpy() for activation in activations]
    return activations


def topological_uncertainty_witness_function(
    model: MLP, x: Tensor, variance_correction: bool = False
) -> List[np.ndarray]:
    def _layer_persistence_diagrams(layer: List[np.ndarray]) -> List[np.ndarray]:
        # Get PD for each sample individually
        layer_persistence_diagrams = []

        for sample_layer in layer:
            if variance_correction:
                sample_layer = sample_layer - sample_layer.mean()
                sample_layer = sample_layer / (sample_layer.std() + 1e-8)
            # Extract Maximum Spanning Tree
            mst_weights = get_mst_weights_from_bipartite_graph(sample_layer)

            # Save PD
            layer_persistence_diagrams.append(mst_weights)

        layer_persistence_diagrams = np.stack(layer_persistence_diagrams)
        return layer_persistence_diagrams

    # Build Activation Graph
    activation_graph = build_mlp_activation_graph(model, x)

    # Get PD for each layer
    persistence_diagrams = []

    for layer in activation_graph:
        # Convert to numpy
        layer = layer.detach().cpu().numpy()

        # Get PD for each sample individually
        layer_persistence_diagrams = _layer_persistence_diagrams(layer)

        # Save Layer PDs
        persistence_diagrams.append(layer_persistence_diagrams)

    return persistence_diagrams


def distance_from_center(x_reference: np.ndarray, x: np.ndarray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return np.linalg.norm(x_reference.mean(axis=0) - x, axis=1)
    elif isinstance(x, Iterable):
        elementwise_distances = [
            distance_from_center(x_reference_i, x_i)
            for x_reference_i, x_i in zip(x_reference, x)
        ]
        elementwise_distances = np.stack(elementwise_distances)
        return np.mean(elementwise_distances, axis=0)
    else:
        raise TypeError(f"Expected x to be np.ndarray or Iterable, got {type(x)}")


def distance_from_center_by_groups(
    x_reference: np.ndarray, y_reference: np.ndarray, x: np.ndarray
) -> np.ndarray:
    if isinstance(x, np.ndarray):
        distances = []
        for group in np.sort(np.unique(y_reference)):
            group_x_reference = x_reference[y_reference == group]
            group_distances = distance_from_center(group_x_reference, x)
            distances.append(group_distances)
        return np.stack(distances, axis=1)

    elif isinstance(x, Iterable):
        elementwise_distances = [
            distance_from_center_by_groups(x_reference_i, y_reference, x_i)
            for x_reference_i, x_i in zip(x_reference, x)
        ]
        elementwise_distances = np.stack(elementwise_distances)
        return np.mean(elementwise_distances, axis=0)
    else:
        raise TypeError(f"Expected x to be np.ndarray or Iterable, got {type(x)}")
