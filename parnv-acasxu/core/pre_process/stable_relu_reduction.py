from __future__ import annotations

import copy
from typing import Any

import numpy as np

from core.data_structures.Network import Network
from core.nnet.read_nnet import _build_network_from_weights_and_biases
from core.pre_process.crown_bounds import HiddenLayerCrownBounds


def _as_float_matrices(network: Network) -> tuple[list[np.ndarray], list[np.ndarray]]:
    weights = [np.asarray(layer, dtype=float) for layer in network.generate_weights()]
    biases = [np.asarray(layer, dtype=float) for layer in network.generate_biases()]
    return weights, biases


def _copy_input_bounds(source: Network, destination: Network) -> None:
    if len(source.layers[0].nodes) != len(destination.layers[0].nodes):
        raise ValueError("Input dimension changed during stable-ReLU reduction.")
    for src_node, dst_node in zip(source.layers[0].nodes, destination.layers[0].nodes):
        dst_node.lower_bound = float(src_node.lower_bound)
        dst_node.upper_bound = float(src_node.upper_bound)


def _rebuild_network(reference: Network, weights: list[np.ndarray], biases: list[np.ndarray]) -> Network:
    reduced = _build_network_from_weights_and_biases(
        weights=[layer.tolist() for layer in weights],
        biases=[layer.tolist() for layer in biases],
        acasxu_net=None,
    )
    reduced.nnet_metadata = copy.deepcopy(getattr(reference, "nnet_metadata", {}))
    _copy_input_bounds(reference, reduced)
    return reduced


def _interval_affine_lower_bound(
        matrix: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must be rank-2")
    if lower.shape != upper.shape or matrix.shape[1] != lower.size:
        raise ValueError("Affine interval-bound dimensions do not match.")
    positive = np.maximum(matrix, 0.0)
    negative = np.minimum(matrix, 0.0)
    return positive @ lower + negative @ upper


def _predecessor_bounds(
        network: Network,
        crown_bounds: HiddenLayerCrownBounds,
        hidden_counter: int,
) -> tuple[np.ndarray, np.ndarray]:
    if hidden_counter == 0:
        lower = np.asarray([float(node.lower_bound) for node in network.layers[0].nodes], dtype=float)
        upper = np.asarray([float(node.upper_bound) for node in network.layers[0].nodes], dtype=float)
        return lower, upper
    lower = np.asarray(
        crown_bounds.postactivation_lower_bounds_by_hidden_layer[hidden_counter - 1],
        dtype=float,
    )
    upper = np.asarray(
        crown_bounds.postactivation_upper_bounds_by_hidden_layer[hidden_counter - 1],
        dtype=float,
    )
    return lower, upper


def _hidden_sizes_from_weights(weights: list[np.ndarray]) -> list[int]:
    return [int(weight.shape[0]) for weight in weights[:-1]]


def reduce_stable_relu_neurons(
        network: Network,
        crown_bounds: HiddenLayerCrownBounds,
        stability_tolerance: float = 1e-9,
        activation_margin: float = 1e-9,
        reconstruct_stably_active: bool = True,
) -> tuple[Network, dict[str, Any]]:
    """Construct an exact REDNet-style reduction for a sequential fully-connected ReLU network.

    Stable inactive ReLUs are removed. If a hidden layer contains more stably-active
    ReLUs than the width of its already-reduced successor, their affine contribution
    is folded into successor_width new ReLUs. A non-negative shift makes those new
    ReLUs provably active, so the rewrite is input-output equivalent over the
    verification input box.

    Hidden layers are processed backwards, matching REDNet. CROWN bounds are used
    only to certify stability; this rewrite itself introduces no over-approximation.
    """
    if stability_tolerance < 0.0:
        raise ValueError("stability_tolerance must be non-negative")
    if activation_margin < 0.0:
        raise ValueError("activation_margin must be non-negative")

    weights, biases = _as_float_matrices(network)
    hidden_count = len(weights) - 1
    if hidden_count <= 0:
        return network, {
            "mode": "full",
            "original_hidden_sizes": [],
            "reduced_hidden_sizes": [],
            "original_relu_count": 0,
            "reduced_relu_count": 0,
            "reduction_ratio": 1.0,
            "total_inactive_removed": 0,
            "total_active_eliminated": 0,
            "total_reconstructed_units": 0,
            "per_layer": [],
        }

    bound_groups = (
        crown_bounds.preactivation_lower_bounds_by_hidden_layer,
        crown_bounds.preactivation_upper_bounds_by_hidden_layer,
        crown_bounds.postactivation_lower_bounds_by_hidden_layer,
        crown_bounds.postactivation_upper_bounds_by_hidden_layer,
    )
    if any(len(group) != hidden_count for group in bound_groups):
        raise ValueError(
            "CROWN hidden-layer bound count does not match the network: {} hidden layers.".format(hidden_count)
        )

    original_hidden_sizes = _hidden_sizes_from_weights(weights)
    per_layer: list[dict[str, Any]] = []
    total_inactive_removed = 0
    total_active_eliminated = 0
    total_reconstructed_units = 0

    # Backward order is essential: after reducing layer l+1, its smaller width is
    # immediately available as the reconstruction width for layer l.
    for hidden_counter in reversed(range(hidden_count)):
        current_weight_index = hidden_counter
        next_weight_index = hidden_counter + 1
        incoming = np.asarray(weights[current_weight_index], dtype=float)
        current_bias = np.asarray(biases[current_weight_index], dtype=float)
        outgoing = np.asarray(weights[next_weight_index], dtype=float)
        next_bias = np.asarray(biases[next_weight_index], dtype=float)

        lower = np.asarray(
            crown_bounds.preactivation_lower_bounds_by_hidden_layer[hidden_counter],
            dtype=float,
        )
        upper = np.asarray(
            crown_bounds.preactivation_upper_bounds_by_hidden_layer[hidden_counter],
            dtype=float,
        )
        width_before = int(current_bias.size)
        if lower.size != width_before or upper.size != width_before:
            raise ValueError(
                "CROWN bounds for hidden layer {} have widths ({}, {}), expected {}.".format(
                    hidden_counter + 1,
                    lower.size,
                    upper.size,
                    width_before,
                )
            )
        if incoming.shape[0] != width_before or outgoing.shape[1] != width_before:
            raise ValueError("Dense network dimensions are inconsistent at hidden layer {}.".format(hidden_counter + 1))

        inactive_mask = upper <= -float(stability_tolerance)
        active_mask = lower >= float(stability_tolerance)
        # With tolerance == 0, a neuron exactly at zero could satisfy both masks.
        # It is a zero ReLU; classify it as inactive to avoid overlap.
        active_mask = active_mask & ~inactive_mask
        inactive_indices = np.flatnonzero(inactive_mask)
        active_indices = np.flatnonzero(active_mask)
        unstable_indices = np.flatnonzero(~inactive_mask & ~active_mask)

        total_inactive_removed += int(inactive_indices.size)
        next_width = int(outgoing.shape[0])
        reconstruct_active = bool(
            reconstruct_stably_active
            and active_indices.size > next_width
            and next_width > 0
        )

        layer_report: dict[str, Any] = {
            "layer_index": hidden_counter + 1,
            "width_before": width_before,
            "successor_width_after_later_reduction": next_width,
            "stable_inactive": int(inactive_indices.size),
            "stable_active": int(active_indices.size),
            "unstable": int(unstable_indices.size),
            "reconstructed_stable_active": reconstruct_active,
            "inactive_indices": inactive_indices.tolist(),
            "active_indices": active_indices.tolist(),
        }

        if reconstruct_active:
            # For h_A = W_A x + b_A (all active) and z_next = V_A h_A + ...,
            # fold V_A h_A into M x + d. Introduce next_width active ReLUs
            # ReLU(M x + shift), then compensate d-shift in the next bias.
            active_incoming = incoming[active_indices, :]
            active_bias = current_bias[active_indices]
            active_outgoing = outgoing[:, active_indices]
            folded_weight = active_outgoing @ active_incoming
            folded_bias = active_outgoing @ active_bias

            predecessor_lower, predecessor_upper = _predecessor_bounds(
                network=network,
                crown_bounds=crown_bounds,
                hidden_counter=hidden_counter,
            )
            folded_lower = _interval_affine_lower_bound(
                matrix=folded_weight,
                lower=predecessor_lower,
                upper=predecessor_upper,
            )
            shift = np.maximum(0.0, float(activation_margin) - folded_lower)

            if unstable_indices.size:
                new_incoming = np.vstack((incoming[unstable_indices, :], folded_weight))
                new_current_bias = np.concatenate((current_bias[unstable_indices], shift))
                new_outgoing = np.column_stack((
                    outgoing[:, unstable_indices],
                    np.eye(next_width, dtype=float),
                ))
            else:
                new_incoming = folded_weight.copy()
                new_current_bias = shift.copy()
                new_outgoing = np.eye(next_width, dtype=float)
            new_next_bias = next_bias + folded_bias - shift

            total_active_eliminated += int(active_indices.size)
            total_reconstructed_units += next_width
            layer_report.update({
                "active_eliminated": int(active_indices.size),
                "reconstructed_units": next_width,
                "reconstruction_interval_lower": folded_lower.tolist(),
                "reconstruction_shift": shift.tolist(),
            })
        else:
            keep_indices = np.flatnonzero(~inactive_mask)
            if keep_indices.size:
                new_incoming = incoming[keep_indices, :]
                new_current_bias = current_bias[keep_indices]
                new_outgoing = outgoing[:, keep_indices]
                new_next_bias = next_bias
            else:
                # Sequential .nnet representations do not support an empty hidden layer.
                # A single always-zero dummy ReLU is exact and keeps the graph well formed.
                new_incoming = np.zeros((1, incoming.shape[1]), dtype=float)
                new_current_bias = np.asarray([-max(1.0, float(activation_margin))], dtype=float)
                new_outgoing = np.zeros((next_width, 1), dtype=float)
                new_next_bias = next_bias
                layer_report["inserted_zero_dummy"] = True
            layer_report.update({
                "active_eliminated": 0,
                "reconstructed_units": 0,
            })

        weights[current_weight_index] = np.asarray(new_incoming, dtype=float)
        biases[current_weight_index] = np.asarray(new_current_bias, dtype=float)
        weights[next_weight_index] = np.asarray(new_outgoing, dtype=float)
        biases[next_weight_index] = np.asarray(new_next_bias, dtype=float)
        layer_report["width_after"] = int(new_current_bias.size)
        layer_report["neurons_removed"] = int(width_before - new_current_bias.size)
        per_layer.append(layer_report)

    reduced = _rebuild_network(network, weights=weights, biases=biases)
    reduced_hidden_sizes = _hidden_sizes_from_weights(weights)
    original_relu_count = int(sum(original_hidden_sizes))
    reduced_relu_count = int(sum(reduced_hidden_sizes))
    reduction_ratio = (
        float(original_relu_count) / float(reduced_relu_count)
        if reduced_relu_count > 0
        else float("inf")
    )
    per_layer.sort(key=lambda item: int(item["layer_index"]))
    report = {
        "mode": "full",
        "stability_tolerance": float(stability_tolerance),
        "activation_margin": float(activation_margin),
        "original_hidden_sizes": original_hidden_sizes,
        "reduced_hidden_sizes": reduced_hidden_sizes,
        "original_relu_count": original_relu_count,
        "reduced_relu_count": reduced_relu_count,
        "relu_removed": int(original_relu_count - reduced_relu_count),
        "reduction_ratio": reduction_ratio,
        "total_inactive_removed": int(total_inactive_removed),
        "total_active_eliminated": int(total_active_eliminated),
        "total_reconstructed_units": int(total_reconstructed_units),
        "per_layer": per_layer,
    }
    return reduced, report
