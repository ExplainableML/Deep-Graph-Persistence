import numpy as np

from typing import List
from typing import Tuple
from scipy.stats import kendalltau
from tda import PerLayerCalculation
from collections import OrderedDict


def get_neural_persistence(weights: OrderedDict) -> Tuple[float, List[float]]:
    plc = PerLayerCalculation()
    persistence = plc(weights)

    global_persistence = persistence["global"][
        "accumulated_total_persistence_normalized"
    ]
    local_persistences = [
        persistence[layer_index]["total_persistence_normalized"]
        for layer_index in sorted(weights.keys())
    ]

    return global_persistence, local_persistences


def get_variance(weights: OrderedDict) -> float:
    weights_flattened = [weight.reshape(-1) for weight in weights.values()]
    weights_combined = np.concatenate(weights_flattened)
    weights_normalised = np.abs(weights_combined)
    weights_normalised = weights_normalised / np.max(weights_normalised)

    return np.var(weights_normalised).item()


def _diagonal_flatten(m: np.ndarray) -> np.ndarray:
    # Flatten matrix along diagonals from top left to bottom right
    xs = np.arange(m.shape[0])
    ys = np.arange(m.shape[1])

    grid = xs[:, None] + ys[None, :]
    sorted_idx = np.argsort(grid.reshape(-1))
    return m.reshape(-1)[sorted_idx]


def stochastic_num_inversions(lst, k=100000):
    lst = np.array(lst)
    index_pairs = np.random.randint(0, len(lst) - 1, size=(k, 2))
    no_collisions = index_pairs[:, 0] != index_pairs[:, 1]
    index_pairs = index_pairs[no_collisions]
    index_pairs = np.sort(index_pairs, axis=1)
    swap = lst[index_pairs[:, 0]] > lst[index_pairs[:, 1]]
    return 1 - np.mean(swap)


def _get_sortedness_of_matrix(m: np.ndarray) -> float:
    # Presort matrix
    m = np.abs(m)
    m = m / np.max(m)

    m = m[:, np.argsort(np.mean(m, axis=0))]
    m = m[np.argsort(np.mean(m, axis=1))]

    num_entries = np.prod(m.shape)

    # Return Kendall's Tau as sortedness metric
    return max(
        kendalltau(np.arange(num_entries), _diagonal_flatten(m)).statistic,
        kendalltau(np.arange(num_entries), m.reshape(-1)).statistic,
        kendalltau(np.arange(num_entries), m.T.reshape(-1)).statistic,
    )


def get_sortedness(weights: OrderedDict) -> List[float]:
    return [
        _get_sortedness_of_matrix(weights[layer_index].copy())
        for layer_index in sorted(weights.keys())
    ]
