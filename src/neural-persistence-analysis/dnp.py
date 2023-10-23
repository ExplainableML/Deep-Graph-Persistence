import numpy as np

from typing import Dict
from collections import OrderedDict
from dnp_tda import PerLayerCalculation
from tda import PerLayerCalculation as OriginalPerLayerCalculation


def prep_weightsum_ratio_weights(weights):
    weights = [(weight - np.mean(weight)) / np.std(weight) for weight in weights]
    weightsum_ratios = [np.abs(weight) for weight in weights]

    W = np.max([np.max(weight) for weight in weightsum_ratios])
    weightsum_ratios = [weight / W for weight in weightsum_ratios]

    return weightsum_ratios


def prep_normalized_weights(weights):
    weights = [(weight - np.mean(weight)) / np.std(weight) for weight in weights]
    normalized_weights = [np.abs(weight) for weight in weights]
    return normalized_weights


def prep_normalized_scaled_weights(weights):
    normalized_weights = prep_normalized_weights(weights)
    W = np.max([np.max(weight) for weight in normalized_weights])
    normalized_scaled_weights = [weight / W for weight in normalized_weights]
    return normalized_scaled_weights


def get_summarization_matrix(weights: list[np.ndarray]) -> np.ndarray:
    collapsed_weights = weights[0]

    for weight in weights[1:]:
        collapsed_weights_expanded = np.expand_dims(collapsed_weights, -1)
        weight_expanded = np.expand_dims(weight, 0)

        collapsed_weights = np.minimum(collapsed_weights_expanded, weight_expanded)
        collapsed_weights = np.max(collapsed_weights, axis=1)

    return collapsed_weights


def get_dnp(weights: OrderedDict) -> Dict[str, float]:
    mlp_weights = list(weights.values())
    mlp_weights = [weight.T for weight in mlp_weights]
    plc = PerLayerCalculation()
    plc_original = OriginalPerLayerCalculation()

    # Calculate DNP without special normalisation
    vanilla_dnp = get_summarization_matrix(mlp_weights)
    vanilla_dnp = plc_original({"S": vanilla_dnp})
    vanilla_dnp = vanilla_dnp["global"]["accumulated_total_persistence_normalized"]

    # Calculate DNP with weightsum ratio normalisation
    normalized_scaled_weights = prep_normalized_scaled_weights(mlp_weights)
    S_normabs = get_summarization_matrix(normalized_scaled_weights)

    plc_results = plc({"S": S_normabs}, scale=False)
    dnp_prev = plc_results["global"]["accumulated_total_persistence"]
    dnp_prev_norm = plc_results["global"]["accumulated_total_persistence_normalized"]

    plc_results = plc({"S": S_normabs}, scale=True)
    dnp_exact = plc_results["global"]["accumulated_total_persistence"]
    dnp_exact_norm = plc_results["global"]["accumulated_total_persistence_normalized"]

    return {
        "dnp_prev": dnp_prev,
        "dnp_prev_norm": dnp_prev_norm,
        "dnp_exact": dnp_exact,
        "dnp_exact_norm": dnp_exact_norm,
        "vanilla_dnp": vanilla_dnp,
    }
