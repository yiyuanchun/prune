from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.cegar.pgd_search import (
    PGDConfig, DEFAULT_EXTRA_REFINEMENT_MERGES, search_counterexample,
    add_search_arguments, search_options,
)

from core.nnet.read_nnet import (
    _build_network_from_weights_and_biases,
    network_from_nnet_file,
    write_nnet_file,
)
from core.pre_process.crown_bounds import (
    HiddenLayerCrownBounds,
    compute_crown_hidden_layer_bounds,
    compute_crown_output_bounds,
)
from core.pre_process.dead_relu_pruning import (
    apply_input_bounds_from_property,
    classify_crown_output_bounds,
    prune_dead_relu_neurons,
)
from core.utils.verification_properties_utils import (
    is_satisfying_assignment,
    read_test_property,
)


@dataclass
class RawCegarState:
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    metadata: dict[str, Any]
    input_bounds: tuple[list[float], list[float]]
    labels: dict[int, list[str]] = field(default_factory=dict)
    ids_by_layer: list[list[str]] = field(default_factory=list)
    mapping: dict[str, Any] = field(default_factory=dict)
    merge_log_stack: list[dict[str, Any]] = field(default_factory=list)
    merge_log: list[dict[str, Any]] = field(default_factory=list)
    refinement_log: list[dict[str, Any]] = field(default_factory=list)
    next_split_id: int = 0
    next_merge_id: int = 0


def _as_float_array_list(values: list[Any]) -> list[np.ndarray]:
    return [np.asarray(value, dtype=float) for value in values]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(_to_jsonable(payload), json_file, indent=2, sort_keys=True)


def _print_to_console(message: str) -> None:
    output_stream = sys.__stdout__ if getattr(sys, "__stdout__", None) is not None else sys.stdout
    print(message, file=output_stream, flush=True)


def _network_structure_for_log(network) -> str:
    layer_parts = []
    for layer_index, layer in enumerate(network.layers):
        layer_type = getattr(layer, "type_name", "layer")
        layer_parts.append("{}:{}={}".format(layer_index, layer_type, len(layer.nodes)))
    return " -> ".join(layer_parts)


def _network_weights_biases(network) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return _as_float_array_list(network.generate_weights()), _as_float_array_list(network.generate_biases())


def _input_bounds_from_property(test_property: dict[str, Any], input_size: int) -> tuple[list[float], list[float]]:
    lower = [0.0] * input_size
    upper = [0.0] * input_size
    seen = set()
    for variable, bounds in test_property.get("input", []):
        index = int(variable)
        lower[index] = float(bounds["Lower"])
        upper[index] = float(bounds["Upper"])
        seen.add(index)
    missing = [index for index in range(input_size) if index not in seen]
    if missing:
        raise ValueError("Input bounds are missing for variables: {}".format(missing))
    return lower, upper


def build_network_from_state(state: RawCegarState):
    network = _build_network_from_weights_and_biases(
        weights=[weight.tolist() for weight in state.weights],
        biases=[bias.tolist() for bias in state.biases],
        acasxu_net=None,
    )
    network.nnet_metadata = copy.deepcopy(state.metadata)
    lower_bounds, upper_bounds = state.input_bounds
    for index, node in enumerate(network.layers[0].nodes):
        node.lower_bound = float(lower_bounds[index])
        node.upper_bound = float(upper_bounds[index])
    return network


def write_nnet(network_or_state: Any, nnet_file: str | os.PathLike[str]) -> None:
    if isinstance(network_or_state, RawCegarState):
        weights = [weight.tolist() for weight in network_or_state.weights]
        biases = [bias.tolist() for bias in network_or_state.biases]
        metadata = network_or_state.metadata
    else:
        weights = network_or_state.generate_weights()
        biases = network_or_state.generate_biases()
        metadata = getattr(network_or_state, "nnet_metadata", {})
    Path(nnet_file).parent.mkdir(parents=True, exist_ok=True)
    write_nnet_file(str(nnet_file), weights=weights, biases=biases, metadata=metadata)


def forward_nnet(network_or_state: Any, input_values: dict[int, float] | list[float] | np.ndarray) -> np.ndarray:
    if isinstance(network_or_state, RawCegarState):
        weights = network_or_state.weights
        biases = network_or_state.biases
    else:
        weights, biases = _network_weights_biases(network_or_state)

    if isinstance(input_values, dict):
        vector = np.asarray(
            [float(value) for _, value in sorted(input_values.items(), key=lambda item: int(item[0]))],
            dtype=float,
        )
    else:
        vector = np.asarray(input_values, dtype=float)

    current = vector
    for weight, bias in zip(weights[:-1], biases[:-1]):
        current = np.maximum(np.dot(weight, current) + bias, 0.0)
    return np.dot(weights[-1], current) + biases[-1]


def _state_from_network(network, test_property: dict[str, Any]) -> RawCegarState:
    weights, biases = _network_weights_biases(network)
    metadata = copy.deepcopy(getattr(network, "nnet_metadata", {}))
    input_bounds = _input_bounds_from_property(test_property, len(network.layers[0].nodes))
    ids_by_layer = [
        [node.name for node in layer.nodes]
        for layer in network.layers
    ]
    hidden_layer_indices = get_last_hidden_layer_indices_from_state(weights, count=2)
    labels = {
        layer_index: ["unknown"] * len(ids_by_layer[layer_index])
        for layer_index in hidden_layer_indices
    }
    reference_hidden_sizes = {
        str(layer_index): len(ids_by_layer[layer_index])
        for layer_index in hidden_layer_indices
    }
    return RawCegarState(
        weights=weights,
        biases=biases,
        metadata=metadata,
        input_bounds=input_bounds,
        labels=labels,
        ids_by_layer=ids_by_layer,
        mapping={
            "splits": [],
            "merges": [],
            "merge_reference_hidden_sizes": reference_hidden_sizes,
            "current_ids_by_layer": copy.deepcopy(ids_by_layer),
        },
    )


def _sync_mapping_ids(state: RawCegarState) -> None:
    state.mapping["current_ids_by_layer"] = copy.deepcopy(state.ids_by_layer)


def _labels_for_json(labels: dict[int, list[str]]) -> dict[str, list[str]]:
    return {str(layer_index): list(layer_labels) for layer_index, layer_labels in labels.items()}


def _write_stage_logs(state: RawCegarState, output_dir: Path, cegar_log: dict[str, Any] | None = None) -> None:
    _sync_mapping_ids(state)
    _write_json(output_dir / "labels.json", _labels_for_json(state.labels))
    _write_json(output_dir / "mapping.json", state.mapping)
    _write_json(output_dir / "merge_log.json", state.merge_log)
    _write_json(output_dir / "refinement_log.json", state.refinement_log)
    if cegar_log is not None:
        _write_json(output_dir / "cegar_log.json", cegar_log)


def get_last_hidden_layer_indices_from_state(weights: list[np.ndarray], count: int = 2) -> list[int]:
    output_layer_index = len(weights)
    hidden_indices = list(range(1, output_layer_index))
    return list(reversed(hidden_indices[-count:]))


def compute_crown_bounds(network) -> tuple[HiddenLayerCrownBounds, Any]:
    return compute_crown_hidden_layer_bounds(network), compute_crown_output_bounds(network)


def _hidden_counter_for_layer(layer_index: int) -> int:
    return layer_index - 1


def _postactivation_bounds_for_layer(
        crown_bounds: HiddenLayerCrownBounds,
        layer_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    hidden_counter = _hidden_counter_for_layer(layer_index)
    lower = np.asarray(
        crown_bounds.postactivation_lower_bounds_by_hidden_layer[hidden_counter],
        dtype=float,
    )
    upper = np.asarray(
        crown_bounds.postactivation_upper_bounds_by_hidden_layer[hidden_counter],
        dtype=float,
    )
    return lower, upper


def _predecessor_bounds(
        state: RawCegarState,
        crown_bounds: HiddenLayerCrownBounds,
        layer_index: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    predecessor_layer_index = layer_index - 1
    if predecessor_layer_index == 0:
        lower, upper = state.input_bounds
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float), True
    lower, upper = _postactivation_bounds_for_layer(crown_bounds, predecessor_layer_index)
    return lower, upper, False


def classify_or_split_layer(
        state: RawCegarState,
        layer_index: int,
        tolerance: float = 1e-12,
) -> dict[str, Any]:
    output_layer_index = len(state.weights)
    if layer_index <= 0 or layer_index >= output_layer_index:
        raise ValueError("layer_index must identify a hidden layer")

    incoming_weights = state.weights[layer_index - 1]
    incoming_biases = state.biases[layer_index - 1]
    outgoing_weights = state.weights[layer_index]
    current_ids = state.ids_by_layer[layer_index]

    if layer_index == output_layer_index - 1:
        next_labels = ["inc"] * outgoing_weights.shape[0]
    else:
        next_labels = state.labels[layer_index + 1]
    next_signs = np.asarray([1.0 if label == "inc" else -1.0 for label in next_labels], dtype=float)

    new_rows = []
    new_biases = []
    new_outgoing_columns = []
    new_labels = []
    new_ids = []
    split_records = []

    for node_index, node_id in enumerate(current_ids):
        outgoing_column = outgoing_weights[:, node_index]
        nonzero_mask = np.abs(outgoing_column) > tolerance
        signed_direction = outgoing_column * next_signs
        positive_mask = nonzero_mask & (signed_direction > tolerance)
        negative_mask = nonzero_mask & (signed_direction < -tolerance)
        has_positive = bool(np.any(positive_mask))
        has_negative = bool(np.any(negative_mask))

        if not has_positive and not has_negative:
            label = "inc"
            new_rows.append(incoming_weights[node_index].copy())
            new_biases.append(float(incoming_biases[node_index]))
            new_outgoing_columns.append(outgoing_column.copy())
            new_labels.append(label)
            new_ids.append(node_id)
            continue

        if has_positive and not has_negative:
            label = "inc"
            new_rows.append(incoming_weights[node_index].copy())
            new_biases.append(float(incoming_biases[node_index]))
            new_outgoing_columns.append(outgoing_column.copy())
            new_labels.append(label)
            new_ids.append(node_id)
            continue

        if has_negative and not has_positive:
            label = "dec"
            new_rows.append(incoming_weights[node_index].copy())
            new_biases.append(float(incoming_biases[node_index]))
            new_outgoing_columns.append(outgoing_column.copy())
            new_labels.append(label)
            new_ids.append(node_id)
            continue

        created_ids = []
        for label, keep_mask, suffix in (
                ("inc", positive_mask, "inc"),
                ("dec", negative_mask, "dec"),
        ):
            if not bool(np.any(keep_mask)):
                continue
            split_id = "{}_{}_s{}".format(node_id, suffix, state.next_split_id)
            state.next_split_id += 1
            split_outgoing = np.zeros_like(outgoing_column, dtype=float)
            split_outgoing[keep_mask] = outgoing_column[keep_mask]
            new_rows.append(incoming_weights[node_index].copy())
            new_biases.append(float(incoming_biases[node_index]))
            new_outgoing_columns.append(split_outgoing)
            new_labels.append(label)
            new_ids.append(split_id)
            created_ids.append(split_id)

        split_records.append({
            "layer": layer_index,
            "original_neuron": node_id,
            "original_index": node_index,
            "created_neurons": created_ids,
        })

    state.weights[layer_index - 1] = np.vstack(new_rows)
    state.biases[layer_index - 1] = np.asarray(new_biases, dtype=float)
    state.weights[layer_index] = np.column_stack(new_outgoing_columns)
    state.labels[layer_index] = new_labels
    state.ids_by_layer[layer_index] = new_ids
    state.mapping["splits"].extend(split_records)
    _sync_mapping_ids(state)
    return {
        "layer": layer_index,
        "splits": split_records,
        "size_before": len(current_ids),
        "size_after": len(new_ids),
    }


def preprocess_last_two_hidden_layers_inc_dec(
        state: RawCegarState,
        tolerance: float = 1e-12,
) -> list[dict[str, Any]]:
    reports = []
    for layer_index in get_last_hidden_layer_indices_from_state(state.weights, count=2):
        reports.append(classify_or_split_layer(state, layer_index, tolerance=tolerance))
    return reports


def _random_equivalence_test(
        reference_network,
        state: RawCegarState,
        samples: int = 20,
        tolerance: float = 1e-6,
        seed: int = 0,
) -> dict[str, Any]:
    lower_bounds, upper_bounds = state.input_bounds
    rng = np.random.default_rng(seed)
    max_error = 0.0
    worst_case = None
    for _ in range(samples):
        input_vector = np.asarray([
            rng.uniform(low, high) if high > low else low
            for low, high in zip(lower_bounds, upper_bounds)
        ], dtype=float)
        input_dict = {index: float(value) for index, value in enumerate(input_vector)}
        reference_output = np.asarray(reference_network.speedy_evaluate(input_dict), dtype=float)
        preprocessed_output = forward_nnet(state, input_vector)
        error = float(np.max(np.abs(reference_output - preprocessed_output)))
        if error > max_error:
            max_error = error
            worst_case = {
                "input": input_vector.tolist(),
                "reference_output": reference_output.tolist(),
                "preprocessed_output": preprocessed_output.tolist(),
                "error": error,
            }
        if error > tolerance:
            return {
                "passed": False,
                "max_error": max_error,
                "worst_case": worst_case,
            }
    return {
        "passed": True,
        "max_error": max_error,
        "worst_case": worst_case,
    }


def _random_inputs_from_state(
        state: RawCegarState,
        samples: int = 10,
        seed: int | None = None,
) -> list[dict[int, float]]:
    lower_bounds, upper_bounds = state.input_bounds
    rng = np.random.default_rng(seed)
    return [
        {
            index: float(rng.uniform(low, high)) if high > low else float(low)
            for index, (low, high) in enumerate(zip(lower_bounds, upper_bounds))
        }
        for _ in range(samples)
    ]


def _random_property_violation_test(
        network,
        test_property: dict[str, Any],
        random_inputs: list[dict[int, float]],
) -> dict[str, Any]:
    _, variables2nodes = network.get_variables()
    max_output = None
    for sample_index, input_values in enumerate(random_inputs):
        output = np.asarray(network.speedy_evaluate(input_values), dtype=float)
        sample_max = float(np.max(output)) if output.size else None
        if sample_max is not None:
            max_output = sample_max if max_output is None else max(max_output, sample_max)
        if is_satisfying_assignment(network, test_property, output, variables2nodes):
            return {
                "sample_count": len(random_inputs),
                "has_violation": True,
                "violating_sample_index": sample_index,
                "violating_output": output.tolist(),
                "max_output": max_output,
            }
    return {
        "sample_count": len(random_inputs),
        "has_violation": False,
        "violating_sample_index": None,
        "violating_output": None,
        "max_output": max_output,
    }


def _has_inc_dec_labels(labels: list[str] | None) -> bool:
    return bool(labels) and all(label in {"inc", "dec"} for label in labels)


def _validate_lp_bounds(
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        predecessor_is_input: bool,
        tolerance: float = 1e-9,
) -> tuple[bool, str]:
    if lower_bounds.shape != upper_bounds.shape:
        return False, "predecessor bound shapes differ"
    if not np.all(np.isfinite(lower_bounds)) or not np.all(np.isfinite(upper_bounds)):
        return False, "predecessor bounds contain NaN or infinity"
    if np.any(lower_bounds > upper_bounds + tolerance):
        return False, "predecessor lower bound is greater than upper bound"
    if not predecessor_is_input and np.any(lower_bounds < -tolerance):
        return False, "post-ReLU predecessor lower bound is negative"
    return True, ""


def _fallback_merge_weights(
        weight_j: np.ndarray,
        bias_j: float,
        weight_k: np.ndarray,
        bias_k: float,
        neuron_type: str,
        reason: str,
) -> dict[str, Any]:
    if neuron_type == "inc":
        merged_weights = np.maximum(weight_j, weight_k)
        merged_bias = max(float(bias_j), float(bias_k))
    else:
        merged_weights = np.minimum(weight_j, weight_k)
        merged_bias = min(float(bias_j), float(bias_k))
    return {
        "merged_weights": merged_weights,
        "merged_bias": float(merged_bias),
        "solver_status": "fallback",
        "objective_value": None,
        "used_fallback": True,
        "fallback_reason": reason,
        "certificate_values": {},
    }


def solve_merged_incoming_weights_and_bias(
        neuron_j: dict[str, Any],
        neuron_k: dict[str, Any],
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        neuron_type: str,
        predecessor_is_input: bool = False,
) -> dict[str, Any]:
    weight_j = np.asarray(neuron_j["weights"], dtype=float)
    weight_k = np.asarray(neuron_k["weights"], dtype=float)
    bias_j = float(neuron_j["bias"])
    bias_k = float(neuron_k["bias"])
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)

    valid_bounds, invalid_reason = _validate_lp_bounds(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        predecessor_is_input=predecessor_is_input,
    )
    if not valid_bounds:
        return _fallback_merge_weights(
            weight_j, bias_j, weight_k, bias_k, neuron_type,
            "LP-Merge fallback: predecessor bounds invalid ({})".format(invalid_reason),
        )

    try:
        from scipy.optimize import linprog
    except Exception as exc:
        return _fallback_merge_weights(
            weight_j, bias_j, weight_k, bias_k, neuron_type,
            "LP-Merge fallback: scipy.optimize.linprog unavailable ({})".format(type(exc).__name__),
        )

    dimension = int(weight_j.size)
    if weight_k.size != dimension or lower_bounds.size != dimension or upper_bounds.size != dimension:
        return _fallback_merge_weights(
            weight_j, bias_j, weight_k, bias_k, neuron_type,
            "LP-Merge fallback: dimension mismatch",
        )

    delta = upper_bounds - lower_bounds
    midpoint = (lower_bounds + upper_bounds) / 2.0
    num_variables = dimension + 1 + 2 * dimension
    w_offset = 0
    b_index = dimension
    slack_j_offset = dimension + 1
    slack_k_offset = slack_j_offset + dimension

    c = np.zeros(num_variables, dtype=float)
    if neuron_type == "inc":
        c[w_offset:w_offset + dimension] = midpoint
        c[b_index] = 1.0
    elif neuron_type == "dec":
        c[w_offset:w_offset + dimension] = -midpoint
        c[b_index] = -1.0
    else:
        raise ValueError("neuron_type must be 'inc' or 'dec'")

    a_ub = []
    b_ub = []
    for source_weights, source_bias, slack_offset in (
            (weight_j, bias_j, slack_j_offset),
            (weight_k, bias_k, slack_k_offset),
    ):
        for coordinate in range(dimension):
            row = np.zeros(num_variables, dtype=float)
            if neuron_type == "inc":
                row[w_offset + coordinate] = -1.0
                row[slack_offset + coordinate] = -1.0
                a_ub.append(row)
                b_ub.append(-float(source_weights[coordinate]))
            else:
                row[w_offset + coordinate] = 1.0
                row[slack_offset + coordinate] = -1.0
                a_ub.append(row)
                b_ub.append(float(source_weights[coordinate]))

        row = np.zeros(num_variables, dtype=float)
        if neuron_type == "inc":
            row[w_offset:w_offset + dimension] = -lower_bounds
            row[b_index] = -1.0
            row[slack_offset:slack_offset + dimension] = delta
            a_ub.append(row)
            b_ub.append(-float(source_bias) - float(np.dot(lower_bounds, source_weights)))
        else:
            row[w_offset:w_offset + dimension] = lower_bounds
            row[b_index] = 1.0
            row[slack_offset:slack_offset + dimension] = delta
            a_ub.append(row)
            b_ub.append(float(source_bias) + float(np.dot(lower_bounds, source_weights)))

    bounds = [(None, None)] * (dimension + 1) + [(0.0, None)] * (2 * dimension)
    result = linprog(
        c=c,
        A_ub=np.asarray(a_ub, dtype=float),
        b_ub=np.asarray(b_ub, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return _fallback_merge_weights(
            weight_j, bias_j, weight_k, bias_k, neuron_type,
            "LP-Merge fallback: linprog status {} ({})".format(result.status, result.message),
        )

    solution = np.asarray(result.x, dtype=float)
    merged_weights = solution[w_offset:w_offset + dimension]
    merged_bias = float(solution[b_index])
    return {
        "merged_weights": merged_weights,
        "merged_bias": merged_bias,
        "solver_status": "optimal",
        "objective_value": float(result.fun),
        "used_fallback": False,
        "fallback_reason": "",
        "certificate_values": {
            "slack_j": solution[slack_j_offset:slack_j_offset + dimension].tolist(),
            "slack_k": solution[slack_k_offset:slack_k_offset + dimension].tolist(),
        },
    }


def select_merge_pair_by_crown_score(
        state: RawCegarState,
        layer_index: int,
        crown_bounds: HiddenLayerCrownBounds,
) -> dict[str, Any] | None:
    labels = state.labels.get(layer_index)
    if not labels:
        return None
    lower_bounds, upper_bounds = _postactivation_bounds_for_layer(crown_bounds, layer_index)
    scores = 0.5 * (upper_bounds - lower_bounds)
    label_to_indices: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(index)

    eligible_indices = [
        index
        for indices in label_to_indices.values()
        if len(indices) >= 2
        for index in indices
    ]
    if not eligible_indices:
        return None

    first_index = min(eligible_indices, key=lambda index: (float(scores[index]), index))
    first_label = labels[first_index]
    incoming_weights = state.weights[layer_index - 1]
    candidate_indices = [
        index for index, label in enumerate(labels)
        if label == first_label and index != first_index
    ]
    if not candidate_indices:
        return None
    second_index = min(
        candidate_indices,
        key=lambda index: (
            float(np.linalg.norm(incoming_weights[first_index] - incoming_weights[index])),
            float(scores[index]),
            index,
        ),
    )
    return {
        "layer": layer_index,
        "j": int(first_index),
        "k": int(second_index),
        "neuron_type": first_label,
        "score_j": float(scores[first_index]),
        "score_k": float(scores[second_index]),
        "incoming_weight_distance": float(
            np.linalg.norm(incoming_weights[first_index] - incoming_weights[second_index])
        ),
    }


def merge_two_neurons(
        state: RawCegarState,
        pair: dict[str, Any],
        crown_bounds: HiddenLayerCrownBounds,
) -> dict[str, Any]:
    layer_index = int(pair["layer"])
    j = int(pair["j"])
    k = int(pair["k"])
    if j == k:
        raise ValueError("Cannot merge a neuron with itself")
    if j > k:
        j, k = k, j

    incoming_weights = state.weights[layer_index - 1]
    incoming_biases = state.biases[layer_index - 1]
    outgoing_weights = state.weights[layer_index]
    labels = state.labels[layer_index]
    ids = state.ids_by_layer[layer_index]
    if labels[j] != labels[k]:
        raise ValueError("Merge candidates must have the same inc/dec label")

    predecessor_lower, predecessor_upper, predecessor_is_input = _predecessor_bounds(
        state=state,
        crown_bounds=crown_bounds,
        layer_index=layer_index,
    )
    lp_result = solve_merged_incoming_weights_and_bias(
        neuron_j={"weights": incoming_weights[j], "bias": incoming_biases[j]},
        neuron_k={"weights": incoming_weights[k], "bias": incoming_biases[k]},
        lower_bounds=predecessor_lower,
        upper_bounds=predecessor_upper,
        neuron_type=labels[j],
        predecessor_is_input=predecessor_is_input,
    )

    merge_id = state.next_merge_id
    state.next_merge_id += 1
    merged_id = "merge_{}_L{}_{}_{}".format(merge_id, layer_index, ids[j], ids[k])
    keep_indices = [index for index in range(len(ids)) if index not in (j, k)]

    old_layer_weights = incoming_weights.copy()
    old_layer_biases = incoming_biases.copy()
    old_next_weights = outgoing_weights.copy()
    old_ids = list(ids)
    old_labels = list(labels)

    new_incoming_weights = np.vstack(
        [incoming_weights[index] for index in keep_indices] + [np.asarray(lp_result["merged_weights"], dtype=float)]
    )
    new_incoming_biases = np.asarray(
        [incoming_biases[index] for index in keep_indices] + [float(lp_result["merged_bias"])],
        dtype=float,
    )
    new_outgoing_column = outgoing_weights[:, j] + outgoing_weights[:, k]
    new_outgoing_weights = np.column_stack(
        [outgoing_weights[:, index] for index in keep_indices] + [new_outgoing_column]
    )
    new_ids = [ids[index] for index in keep_indices] + [merged_id]
    new_labels = [labels[index] for index in keep_indices] + [labels[j]]

    state.weights[layer_index - 1] = new_incoming_weights
    state.biases[layer_index - 1] = new_incoming_biases
    state.weights[layer_index] = new_outgoing_weights
    state.ids_by_layer[layer_index] = new_ids
    state.labels[layer_index] = new_labels

    log_record = {
        "merge_id": merge_id,
        "layer": layer_index,
        "merged_neuron": merged_id,
        "old_neurons": [old_ids[j], old_ids[k]],
        "neuron_type": labels[j],
        "before_merge_index_j": j,
        "before_merge_index_k": k,
        "old_incoming_weights_j": incoming_weights[j].tolist(),
        "old_incoming_weights_k": incoming_weights[k].tolist(),
        "old_bias_j": float(incoming_biases[j]),
        "old_bias_k": float(incoming_biases[k]),
        "old_outgoing_weights_j": outgoing_weights[:, j].tolist(),
        "old_outgoing_weights_k": outgoing_weights[:, k].tolist(),
        "new_incoming_weights_t": np.asarray(lp_result["merged_weights"], dtype=float).tolist(),
        "new_bias_t": float(lp_result["merged_bias"]),
        "new_outgoing_weights_t": new_outgoing_column.tolist(),
        "score_j": float(pair["score_j"]),
        "score_k": float(pair["score_k"]),
        "incoming_weight_distance": float(pair["incoming_weight_distance"]),
        "solver_status": lp_result["solver_status"],
        "objective_value": lp_result["objective_value"],
        "used_fallback": bool(lp_result["used_fallback"]),
        "fallback_reason": lp_result.get("fallback_reason", ""),
        "certificate_values": lp_result["certificate_values"],
        "layer_size_before_merge": len(old_ids),
        "layer_size_after_merge": len(new_ids),
        "layer_ids_before": old_ids,
        "layer_labels_before": old_labels,
        "layer_weights_before": old_layer_weights.tolist(),
        "layer_biases_before": old_layer_biases.tolist(),
        "next_layer_weights_before": old_next_weights.tolist(),
    }
    state.merge_log_stack.append(log_record)
    state.merge_log.append(log_record)
    state.mapping["merges"].append({
        "merge_id": merge_id,
        "layer": layer_index,
        "merged_neuron": merged_id,
        "old_neurons": [old_ids[j], old_ids[k]],
    })
    _sync_mapping_ids(state)
    return log_record


def merge_last_two_hidden_layers(
        state: RawCegarState,
        tolerance: float = 1e-12,
        test_property: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    merge_reports = []
    reference_sizes = {
        int(layer_index): int(size)
        for layer_index, size in state.mapping.get("merge_reference_hidden_sizes", {}).items()
    }
    for layer_index in get_last_hidden_layer_indices_from_state(state.weights, count=2):
        initial_size = reference_sizes.get(layer_index, len(state.ids_by_layer[layer_index]))
        target_size = max(int(math.ceil(0.5 * initial_size)), 1)
        random_inputs = _random_inputs_from_state(state, samples=10, seed=layer_index)
        if len(state.ids_by_layer[layer_index]) <= target_size:
            merge_reports.append({
                "layer": layer_index,
                "reason": "target size already reached; inc/dec preprocessing skipped",
                "initial_size_before_inc_dec": initial_size,
                "current_size": len(state.ids_by_layer[layer_index]),
                "target_size": target_size,
            })
            continue
        if layer_index < len(state.weights) - 1:
            next_labels = state.labels.get(layer_index + 1)
            if not _has_inc_dec_labels(next_labels):
                merge_reports.append({
                    "layer": layer_index,
                    "reason": "downstream hidden layer was not inc/dec preprocessed; merge skipped",
                    "initial_size_before_inc_dec": initial_size,
                    "current_size": len(state.ids_by_layer[layer_index]),
                    "target_size": target_size,
                })
                continue
        preprocess_report = classify_or_split_layer(state, layer_index, tolerance=tolerance)
        merge_reports.append({
            "layer": layer_index,
            "reason": "inc/dec preprocessing before layer merge",
            "initial_size_before_inc_dec": initial_size,
            "target_size": target_size,
            "inc_dec_preprocess": preprocess_report,
        })
        while True:
            current_size = len(state.ids_by_layer[layer_index])
            if current_size <= target_size:
                if test_property is None:
                    merge_reports.append({
                        "layer": layer_index,
                        "reason": "target size reached",
                        "initial_size_before_inc_dec": initial_size,
                        "current_size": current_size,
                        "target_size": target_size,
                    })
                    break
                current_network = build_network_from_state(state)
                random_test = _random_property_violation_test(
                    network=current_network,
                    test_property=test_property,
                    random_inputs=random_inputs,
                )
                merge_reports.append({
                    "layer": layer_index,
                    "reason": (
                        "target size reached with random violation"
                        if random_test["has_violation"]
                        else "target size reached with safe random samples"
                    ),
                    "initial_size_before_inc_dec": initial_size,
                    "current_size": current_size,
                    "target_size": target_size,
                    "random_property_test": random_test,
                })
                if random_test["has_violation"]:
                    return merge_reports
                break
            current_network = build_network_from_state(state)
            crown_hidden_bounds = compute_crown_hidden_layer_bounds(current_network)
            pair = select_merge_pair_by_crown_score(
                state=state,
                layer_index=layer_index,
                crown_bounds=crown_hidden_bounds,
            )
            if pair is None:
                merge_reports.append({
                    "layer": layer_index,
                    "reason": "no same-type merge pair",
                    "current_size": len(state.ids_by_layer[layer_index]),
                    "target_size": target_size,
                })
                break
            merge_record = merge_two_neurons(state, pair, crown_hidden_bounds)
            if test_property is not None:
                current_network = build_network_from_state(state)
                random_test = _random_property_violation_test(
                    network=current_network,
                    test_property=test_property,
                    random_inputs=random_inputs,
                )
                merge_record["random_property_test_after_merge"] = random_test
                merge_record["random_violation_ignored_until_target"] = (
                    len(state.ids_by_layer[layer_index]) > target_size
                    and random_test["has_violation"]
                )
            merge_reports.append(merge_record)
    return merge_reports


def verify_network_with_planet(
        network,
        test_property: dict[str, Any],
        timeout: int | None = None,
        save_query_path: str | os.PathLike[str] | None = None,
        planet_bin: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    from core.utils.planet_query_utils import verify_network_with_planet as _verify

    result = _verify(
        network=network,
        test_property=copy.deepcopy(test_property),
        timeout=1200 if timeout is None else int(timeout),
        save_query_path=save_query_path,
        planet_bin=planet_bin,
    )
    return {
        "status": result["status"],
        "counterexample": result["counterexample"],
        "planet_runtime": result["planet_runtime"],
        "raw_solver_output": result["raw_solver_output"],
    }


def verify_with_planet(
        nnet_file: str | os.PathLike[str],
        property_file: str | os.PathLike[str],
        timeout: int | None = None,
        planet_bin: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    test_property = read_test_property(str(property_file))
    if "type" not in test_property:
        if len(test_property) != 1:
            raise ValueError(
                "Property file contains multiple properties; use the batch driver or pass a single-property file."
            )
        test_property = next(iter(test_property.values()))
    network = network_from_nnet_file(str(nnet_file))
    from core.utils.marabou_query_utils import reduce_property_to_basic_form

    network, test_property = reduce_property_to_basic_form(network, copy.deepcopy(test_property))
    apply_input_bounds_from_property(network, test_property)
    return verify_network_with_planet(
        network,
        test_property,
        timeout=timeout,
        planet_bin=planet_bin,
    )


def verify_network_with_marabou(
        network,
        test_property: dict[str, Any],
        timeout: int | None = None,
        save_query_path: str | os.PathLike[str] | None = None,
        planet_bin: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    return verify_network_with_planet(
        network=network,
        test_property=test_property,
        timeout=timeout,
        save_query_path=save_query_path,
        planet_bin=planet_bin,
    )


def verify_with_marabou(
        nnet_file: str | os.PathLike[str],
        property_file: str | os.PathLike[str],
        timeout: int | None = None,
        planet_bin: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    return verify_with_planet(
        nnet_file=nnet_file,
        property_file=property_file,
        timeout=timeout,
        planet_bin=planet_bin,
    )


def _normalize_solver_status(status: Any) -> str:
    text = str(status).strip().lower()
    if "timeout" in text:
        return "timeout"
    if "unsat" in text:
        return "unsat"
    if text == "sat" or "sat" in text:
        return "sat"
    if "error" in text:
        return "error"
    return "unknown"


def _counterexample_input(counterexample: Any, input_size: int) -> dict[int, float]:
    if not isinstance(counterexample, dict):
        return {}
    normalized = {}
    for key, value in counterexample.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if index < input_size:
            normalized[index] = float(value)
    return {index: normalized[index] for index in sorted(normalized)[:input_size]}


def _counterexample_violates_network(
        network,
        counterexample: dict[int, float],
        property_spec: dict[str, Any],
        tolerance: float = 1e-7,
) -> tuple[bool, list[float]]:
    del tolerance
    expected = set(range(len(network.layers[0].nodes)))
    if set(counterexample) != expected:
        raise ValueError("Counterexample input is incomplete")
    for i, bounds in property_spec['input']:
        value = counterexample[int(i)]
        if not np.isfinite(value) or not bounds['Lower'] <= value <= bounds['Upper']:
            raise ValueError("Counterexample lies outside the property input box")
    output = np.asarray(network.speedy_evaluate(counterexample), dtype=float)
    if not np.all(np.isfinite(output)):
        raise ValueError("Nonfinite counterexample output")
    _, variables2nodes = network.get_variables(property_type=property_spec.get("type", "basic"))
    violates = is_satisfying_assignment(
        network=network,
        test_property=property_spec,
        output=output,
        variables2nodes=variables2nodes,
    )
    return bool(violates), output.tolist()


def is_counterexample_on_original_network(
        original_network,
        counterexample: dict[int, float],
        property_spec: dict[str, Any],
        tolerance: float = 1e-7,
) -> bool:
    violates, _ = _counterexample_violates_network(
        network=original_network,
        counterexample=counterexample,
        property_spec=property_spec,
        tolerance=tolerance,
    )
    return violates


def is_counterexample_on_current_network(
        current_network,
        counterexample: dict[int, float],
        property_spec: dict[str, Any],
        tolerance: float = 1e-7,
) -> bool:
    violates, _ = _counterexample_violates_network(
        network=current_network,
        counterexample=counterexample,
        property_spec=property_spec,
        tolerance=tolerance,
    )
    return violates


def undo_last_merge(
        current_network,
        merge_log_stack: list[dict[str, Any]],
        labels: dict[int, list[str]],
        mapping: dict[str, Any],
        state: RawCegarState | None = None,
) -> dict[str, Any]:
    del current_network
    if state is None:
        raise ValueError("undo_last_merge requires the RawCegarState used by this flow")
    if not merge_log_stack:
        raise ValueError("No merge operation is available to undo")
    record = merge_log_stack.pop()
    layer_index = int(record["layer"])
    state.weights[layer_index - 1] = np.asarray(record["layer_weights_before"], dtype=float)
    state.biases[layer_index - 1] = np.asarray(record["layer_biases_before"], dtype=float)
    state.weights[layer_index] = np.asarray(record["next_layer_weights_before"], dtype=float)
    state.ids_by_layer[layer_index] = list(record["layer_ids_before"])
    state.labels[layer_index] = list(record["layer_labels_before"])
    labels[layer_index] = state.labels[layer_index]
    mapping["current_ids_by_layer"] = copy.deepcopy(state.ids_by_layer)
    refinement_record = {
        "undo_merge_id": record["merge_id"],
        "layer": layer_index,
        "restored_neurons": record["old_neurons"],
        "removed_merged_neuron": record["merged_neuron"],
    }
    state.refinement_log.append(refinement_record)
    return refinement_record


def refine_by_undo_merge(
        current_network,
        counterexample: dict[int, float],
        property_spec: dict[str, Any],
        merge_log_stack: list[dict[str, Any]],
        labels: dict[int, list[str]],
        mapping: dict[str, Any],
        tolerance: float = 1e-7,
        max_steps: int | None = None,
        state: RawCegarState | None = None,
        extra_refinement_merges: int = DEFAULT_EXTRA_REFINEMENT_MERGES,
) -> dict[str, Any]:
    if state is None:
        raise ValueError("refine_by_undo_merge requires the RawCegarState used by this flow")
    if extra_refinement_merges < 0 or (max_steps is not None and max_steps < 0):
        raise ValueError("Refinement budgets must be non-negative")
    del current_network
    refined_steps = []
    eliminated = False
    extra_steps = 0
    is_still_counterexample = True
    while merge_log_stack and (max_steps is None or len(refined_steps) < max_steps):
        is_extra = eliminated
        refinement_record = undo_last_merge(
            current_network=None, merge_log_stack=merge_log_stack,
            labels=labels, mapping=mapping, state=state,
        )
        current_network_after_undo = build_network_from_state(state)
        is_still_counterexample = is_counterexample_on_current_network(
            current_network=current_network_after_undo, counterexample=counterexample,
            property_spec=property_spec, tolerance=tolerance,
        )
        refinement_record["is_still_counterexample"] = bool(is_still_counterexample)
        refinement_record["extra_refinement"] = is_extra
        refined_steps.append(refinement_record)
        if is_extra:
            extra_steps += 1
        if not is_still_counterexample:
            eliminated = True
        if eliminated and extra_steps >= extra_refinement_merges:
            break
    success = eliminated and not is_still_counterexample
    budget_exhausted = max_steps is not None and len(refined_steps) >= max_steps
    return {
        "success": success,
        "partial": not success and bool(merge_log_stack) and budget_exhausted,
        "refinement_exhausted": not success and not merge_log_stack,
        "budget_exhausted": budget_exhausted,
        "extra_steps": extra_steps,
        "steps": refined_steps,
    }


def _center_input_from_property(test_property: dict[str, Any]) -> dict[int, float]:
    center = {}
    for variable, bounds in test_property.get("input", []):
        lower = float(bounds["Lower"])
        upper = float(bounds["Upper"])
        center[int(variable)] = (lower + upper) / 2.0
    return center


def _finalize(
        output_dir: Path,
        state: RawCegarState | None,
        final_result: dict[str, Any],
        cegar_log: dict[str, Any],
        final_network: Any,
        counterexample: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_result.update(cegar_log.get("search_statistics", {
        "pgd_calls": 0, "pgd_time_seconds": 0.0, "pgd_candidates_found": 0,
        "pgd_genuine_counterexamples": 0, "pgd_spurious_counterexamples": 0,
        "batch_refinement_rounds": 0, "extra_refinement_steps": 0,
    }))
    final_result["cegar_iterations"] = final_result.get("iterations", 0)
    final_result["formal_verifier_calls"] = final_result.get("iterations", 0)
    final_result["planet_calls"] = final_result.get("iterations", 0)
    final_result["marabou_calls"] = 0
    write_nnet(final_network if state is None else state, output_dir / "refined_network_final.nnet")
    if state is not None:
        _write_stage_logs(state, output_dir, cegar_log=cegar_log)
    else:
        _write_json(output_dir / "labels.json", {})
        _write_json(output_dir / "mapping.json", {})
        _write_json(output_dir / "merge_log.json", [])
        _write_json(output_dir / "refinement_log.json", [])
        _write_json(output_dir / "cegar_log.json", cegar_log)
    _write_json(output_dir / "final_result.json", final_result)
    _write_json(output_dir / "counterexample.json", counterexample or {})
    return final_result


def _precheck_final_result(
        status: str,
        reason: str,
        original_network,
        working_network,
        test_property: dict[str, Any],
        output_dir: Path,
        cegar_log: dict[str, Any],
        tolerance: float,
) -> dict[str, Any]:
    write_nnet(working_network, output_dir / "preprocessed_equivalent.nnet")
    write_nnet(working_network, output_dir / "merged_abstract.nnet")
    if status == "UNSAT":
        final_result = {
            "result": "VERIFIED",
            "query_result": "UNSAT",
            "reason": reason,
            "iterations": 0,
            "total_refinement_steps": 0,
            "final_network": "refined_network_final.nnet",
            "crown_precheck_status": status,
        }
        cegar_log["final_result"] = "VERIFIED"
        return _finalize(output_dir, None, final_result, cegar_log, working_network)

    center_input = _center_input_from_property(test_property)
    genuine = is_counterexample_on_original_network(
        original_network=original_network,
        counterexample=center_input,
        property_spec=test_property,
        tolerance=tolerance,
    )
    if genuine:
        _, original_output = _counterexample_violates_network(
            network=original_network,
            counterexample=center_input,
            property_spec=test_property,
            tolerance=tolerance,
        )
        counterexample = {
            "input": center_input,
            "original_network_output": original_output,
            "counterexample_type": "genuine",
            "source": "CROWN precheck SAT",
        }
        final_result = {
            "result": "UNSAFE",
            "query_result": "SAT",
            "reason": reason,
            "counterexample_type": "genuine",
            "counterexample_file": "counterexample.json",
            "iterations": 0,
            "total_refinement_steps": 0,
            "crown_precheck_status": status,
        }
        cegar_log["final_result"] = "UNSAFE"
        cegar_log["final_counterexample"] = counterexample
        return _finalize(output_dir, None, final_result, cegar_log, working_network, counterexample)

    final_result = {
        "result": "UNKNOWN",
        "query_result": "UNKNOWN",
        "reason": "CROWN precheck returned SAT on the transformed network, but the center input did not violate the original network.",
        "iterations": 0,
        "total_refinement_steps": 0,
        "crown_precheck_status": status,
    }
    cegar_log["final_result"] = "UNKNOWN"
    return _finalize(output_dir, None, final_result, cegar_log, working_network)


def cegar_verify_with_planet(
        original_nnet_file: str | os.PathLike[str],
        property_file: str | os.PathLike[str] | dict[str, Any],
        output_dir: str | os.PathLike[str],
        max_refinement_steps: int | None = None,
        planet_timeout: int | None = None,
        planet_bin: str | os.PathLike[str] | None = None,
        tolerance: float = 1e-7,
        property_id: str | None = None,
        marabou_timeout: int | None = None,
        extra_refinement_merges: int = DEFAULT_EXTRA_REFINEMENT_MERGES,
        pgd_config: PGDConfig | None = None,
) -> dict[str, Any]:
    pgd_config = pgd_config or PGDConfig()
    if extra_refinement_merges < 0 or (max_refinement_steps is not None and max_refinement_steps < 0):
        raise ValueError("Refinement budgets must be non-negative")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if planet_timeout is None:
        planet_timeout = marabou_timeout

    if isinstance(property_file, dict):
        raw_property = copy.deepcopy(property_file)
    else:
        raw_property = read_test_property(str(property_file), property_id=property_id)
    if "type" not in raw_property:
        if property_id is not None and property_id in raw_property:
            raw_property = raw_property[property_id]
        else:
            raise ValueError("A single property with a 'type' field is required.")

    original_network = network_from_nnet_file(str(original_nnet_file))
    working_network = network_from_nnet_file(str(original_nnet_file))
    from core.utils.marabou_query_utils import reduce_property_to_basic_form

    working_network, test_property = reduce_property_to_basic_form(
        network=working_network,
        test_property=copy.deepcopy(raw_property),
    )
    apply_input_bounds_from_property(working_network, test_property)

    cegar_log: dict[str, Any] = {
        "original_nnet_file": str(original_nnet_file),
        "property_id": property_id,
        "iterations": [],
        "final_result": None,
        "final_counterexample": None,
    }
    search_stats = {
        "pgd_calls": 0, "pgd_time_seconds": 0.0,
        "pgd_candidates_found": 0, "pgd_genuine_counterexamples": 0,
        "pgd_spurious_counterexamples": 0, "batch_refinement_rounds": 0,
        "extra_refinement_steps": 0,
    }
    cegar_log["search_statistics"] = search_stats
    cegar_log["pgd_searches"] = []
    cegar_log["refinement_policy"] = {
        "extra_refinement_merges": extra_refinement_merges, "pgd": pgd_config.to_dict(),
    }

    crown_start = time.perf_counter()
    crown_hidden_bounds, crown_output_bounds = compute_crown_bounds(working_network)
    crown_time = time.perf_counter() - crown_start
    crown_precheck_status, crown_precheck_reason = classify_crown_output_bounds(
        test_property=test_property,
        output_bounds=crown_output_bounds,
    )
    cegar_log["crown_precheck"] = {
        "status": crown_precheck_status,
        "reason": crown_precheck_reason,
        "time_seconds": crown_time,
        "output_lower_bounds": crown_output_bounds.lower_bounds,
        "output_upper_bounds": crown_output_bounds.upper_bounds,
    }
    if crown_precheck_status in {"SAT", "UNSAT"}:
        return _precheck_final_result(
            status=crown_precheck_status,
            reason=crown_precheck_reason,
            original_network=original_network,
            working_network=working_network,
            test_property=test_property,
            output_dir=output_path,
            cegar_log=cegar_log,
            tolerance=tolerance,
        )

    pruning_report = prune_dead_relu_neurons(
        network=working_network,
        crown_bounds=crown_hidden_bounds,
        preactivation_upper_threshold=0.0,
    )
    cegar_log["dead_relu_pruning_report"] = pruning_report

    state = _state_from_network(working_network, test_property)
    merge_report = merge_last_two_hidden_layers(state, test_property=test_property)
    cegar_log["inc_dec_preprocess_report"] = [
        report["inc_dec_preprocess"]
        for report in merge_report
        if "inc_dec_preprocess" in report
    ]
    cegar_log["inc_dec_equivalence_test"] = {
        "passed": True,
        "skipped": True,
        "reason": "inc/dec preprocessing is applied lazily only to layers that enter Merge",
    }
    cegar_log["merge_report"] = merge_report
    write_nnet(state, output_path / "merged_abstract.nnet")
    _write_stage_logs(state, output_path, cegar_log=cegar_log)

    iteration = 0
    total_refinement_steps = 0
    run_pgd = False
    while True:
        current_network = build_network_from_state(state)
        counterexample_source = "verifier"
        if run_pgd:
            search_stats["pgd_calls"] += 1
            pgd_result = search_counterexample(
                current_network, test_property, pgd_config,
                seed_offset=search_stats["pgd_calls"]-1,
            )
            search_stats["pgd_time_seconds"] += pgd_result["time_seconds"]
            search_record = {
                **pgd_result, "search_index": search_stats["pgd_calls"],
                "after_verifier_iteration": iteration, "source": "pgd", "refinement_steps": [],
            }
            cegar_log["pgd_searches"].append(search_record)
            print("[CEGAR][PGD] search={} found={} elapsed={:.4f}s".format(
                search_stats["pgd_calls"], pgd_result["found"], pgd_result["time_seconds"]), flush=True)
            if pgd_result["found"]:
                search_stats["pgd_candidates_found"] += 1
                counterexample_source = "pgd"
                verify_result = {"status": "sat", "counterexample": pgd_result["counterexample"]}
            # PGD can only be re-enabled after an actual undo: no zero-progress cycle.
            run_pgd = False
        if counterexample_source == "verifier":
            iteration += 1
            current_nnet_file = output_path / "abstract_iter_{}.nnet".format(iteration)
            write_nnet(state, current_nnet_file)
            query_path = output_path / "abstract_iter_{}.rlv".format(iteration)
            _print_to_console(
                "[CEGAR][Planet] property={} iteration={} network={} query={} structure={}".format(
                    property_id,
                    iteration,
                    current_nnet_file,
                    query_path,
                    _network_structure_for_log(current_network),
                )
            )
            verify_result = verify_network_with_planet(
                network=current_network,
                test_property=test_property,
                timeout=planet_timeout,
                save_query_path=query_path,
                planet_bin=planet_bin,
            )
            iteration_log: dict[str, Any] = {
                "iteration": iteration,
                "current_nnet_file": current_nnet_file.name,
                "planet_status": verify_result["status"],
                "planet_runtime": verify_result["planet_runtime"],
                "raw_solver_output": verify_result["raw_solver_output"],
                "refinement_steps": [],
            }
            cegar_log["iterations"].append(iteration_log)
        else:
            iteration_log = search_record

        if verify_result["status"] == "unsat":
            final_result = {
                "result": "VERIFIED",
                "query_result": "UNSAT",
                "reason": "Planet returned unsat on current abstract/refined network",
                "iterations": iteration,
                "total_refinement_steps": total_refinement_steps,
                "final_network": "refined_network_final.nnet",
                "dead_relu_pruned": pruning_report.get("total_pruned", 0),
                "crown_precheck_status": crown_precheck_status,
                "elapsed_seconds": time.perf_counter() - started,
            }
            cegar_log["final_result"] = "VERIFIED"
            return _finalize(output_path, state, final_result, cegar_log, state)

        if verify_result["status"] == "sat":
            counterexample_input = _counterexample_input(
                verify_result["counterexample"],
                input_size=len(state.ids_by_layer[0]),
            )
            iteration_log["counterexample"] = counterexample_input
            if len(counterexample_input) != len(state.ids_by_layer[0]):
                final_result = {
                    "result": "UNKNOWN",
                    "query_result": "SAT",
                    "reason": "Planet returned SAT without a complete input valuation",
                    "iterations": iteration,
                    "total_refinement_steps": total_refinement_steps,
                    "dead_relu_pruned": pruning_report.get("total_pruned", 0),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                cegar_log["final_result"] = "UNKNOWN"
                return _finalize(output_path, state, final_result, cegar_log, state)
            original_violates, original_output = _counterexample_violates_network(
                network=original_network,
                counterexample=counterexample_input,
                property_spec=test_property,
                tolerance=tolerance,
            )
            current_violates, current_output = _counterexample_violates_network(
                network=current_network,
                counterexample=counterexample_input,
                property_spec=test_property,
                tolerance=tolerance,
            )
            iteration_log["original_network_output"] = original_output
            iteration_log["current_network_output"] = current_output
            if original_violates:
                if counterexample_source == "pgd":
                    search_stats["pgd_genuine_counterexamples"] += 1
                counterexample = {
                    "source": counterexample_source,
                    "input": counterexample_input,
                    "original_network_output": original_output,
                    "current_network_output": current_output,
                    "counterexample_type": "genuine",
                }
                final_result = {
                    "result": "UNSAFE",
                    "query_result": "SAT",
                    "counterexample_type": "genuine",
                    "counterexample_file": "counterexample.json",
                    "iterations": iteration,
                    "total_refinement_steps": total_refinement_steps,
                    "dead_relu_pruned": pruning_report.get("total_pruned", 0),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                iteration_log["counterexample_type"] = "genuine"
                cegar_log["final_result"] = "UNSAFE"
                cegar_log["final_counterexample"] = counterexample
                return _finalize(output_path, state, final_result, cegar_log, state, counterexample)

            iteration_log["counterexample_type"] = "spurious"
            if counterexample_source == "pgd":
                search_stats["pgd_spurious_counterexamples"] += 1
            remaining_refinement_budget = (
                None if max_refinement_steps is None
                else max(int(max_refinement_steps) - total_refinement_steps, 0)
            )
            if not state.merge_log_stack or remaining_refinement_budget == 0:
                if counterexample_source == "pgd":
                    iteration_log["fallback_reason"] = "no refinement capacity; call formal verifier"
                    continue
                final_result = {
                    "result": "UNKNOWN",
                    "query_result": "UNKNOWN",
                    "reason": "no remaining Merge or refinement budget after formal SAT",
                    "iterations": iteration,
                    "total_refinement_steps": total_refinement_steps,
                    "dead_relu_pruned": pruning_report.get("total_pruned", 0),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                cegar_log["final_result"] = "UNKNOWN"
                return _finalize(output_path, state, final_result, cegar_log, state)
            refinement_result = refine_by_undo_merge(
                current_network=current_network,
                counterexample=counterexample_input,
                property_spec=test_property,
                merge_log_stack=state.merge_log_stack,
                labels=state.labels,
                mapping=state.mapping,
                tolerance=tolerance,
                max_steps=remaining_refinement_budget,
                extra_refinement_merges=extra_refinement_merges,
                state=state,
            )
            total_refinement_steps += len(refinement_result["steps"])
            search_stats["batch_refinement_rounds"] += int(bool(refinement_result["steps"]))
            search_stats["extra_refinement_steps"] += refinement_result["extra_steps"]
            iteration_log["refinement_steps"] = refinement_result["steps"]
            _write_stage_logs(state, output_path, cegar_log=cegar_log)
            if refinement_result["steps"]:
                run_pgd = pgd_config.enabled
                continue
            final_result = {
                "result": "UNKNOWN",
                "query_result": "UNKNOWN",
                "reason": (
                    "refinement_exhausted"
                    if refinement_result["refinement_exhausted"]
                    else "partial refinement did not eliminate spurious counterexample"
                ),
                "iterations": iteration,
                "total_refinement_steps": total_refinement_steps,
                "dead_relu_pruned": pruning_report.get("total_pruned", 0),
                "elapsed_seconds": time.perf_counter() - started,
            }
            cegar_log["final_result"] = "UNKNOWN"
            return _finalize(output_path, state, final_result, cegar_log, state)

        final_result = {
            "result": "UNKNOWN",
            "query_result": verify_result["status"].upper(),
            "reason": "Planet returned {}".format(verify_result["status"]),
            "iterations": iteration,
            "total_refinement_steps": total_refinement_steps,
            "dead_relu_pruned": pruning_report.get("total_pruned", 0),
            "elapsed_seconds": time.perf_counter() - started,
        }
        cegar_log["final_result"] = "UNKNOWN"
        return _finalize(output_path, state, final_result, cegar_log, state)


def cegar_verify_with_marabou(
        original_nnet_file: str | os.PathLike[str],
        property_file: str | os.PathLike[str] | dict[str, Any],
        output_dir: str | os.PathLike[str],
        max_refinement_steps: int | None = None,
        marabou_timeout: int | None = None,
        tolerance: float = 1e-7,
        property_id: str | None = None,
        planet_bin: str | os.PathLike[str] | None = None,
        extra_refinement_merges: int = DEFAULT_EXTRA_REFINEMENT_MERGES,
        pgd_config: PGDConfig | None = None,
) -> dict[str, Any]:
    return cegar_verify_with_planet(
        original_nnet_file=original_nnet_file,
        property_file=property_file,
        output_dir=output_dir,
        max_refinement_steps=max_refinement_steps,
        planet_timeout=marabou_timeout,
        planet_bin=planet_bin,
        tolerance=tolerance,
        property_id=property_id,
        extra_refinement_merges=extra_refinement_merges,
        pgd_config=pgd_config,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Merge CEGAR with Planet on one .nnet/property pair.")
    parser.add_argument("original_nnet_file", type=Path)
    parser.add_argument("property_file", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--property-id", default=None)
    parser.add_argument("--max-refinement-steps", type=int, default=None)
    parser.add_argument("--planet-timeout", "--marabou-timeout", dest="planet_timeout", type=int, default=None)
    parser.add_argument("--planet-bin", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    add_search_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = cegar_verify_with_planet(
        original_nnet_file=args.original_nnet_file,
        property_file=args.property_file,
        output_dir=args.output_dir,
        max_refinement_steps=args.max_refinement_steps,
        planet_timeout=args.planet_timeout,
        planet_bin=args.planet_bin,
        tolerance=args.tolerance,
        property_id=args.property_id,
        **search_options(args),
    )
    print(json.dumps(_to_jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
