from __future__ import annotations

from typing import Any

from core.data_structures.Network import Network
from core.pre_process.crown_bounds import HiddenLayerCrownBounds, OutputCrownBounds


def apply_input_bounds_from_property(network: Network, test_property: dict[str, Any]) -> None:
    input_bounds = {
        int(var_index): bounds
        for var_index, bounds in test_property.get("input", [])
    }
    missing_inputs = []
    for node_index, node in enumerate(network.layers[0].nodes):
        bounds = input_bounds.get(node_index)
        if bounds is None:
            missing_inputs.append(node_index)
            continue
        if "Lower" not in bounds or "Upper" not in bounds:
            raise ValueError(
                "Input variable {} must have both Lower and Upper bounds for CROWN.".format(node_index)
            )
        node.lower_bound = float(bounds["Lower"])
        node.upper_bound = float(bounds["Upper"])
    if missing_inputs:
        raise ValueError(
            "Input bounds are missing for variables required by CROWN: {}".format(missing_inputs)
        )


def classify_crown_output_bounds(
        test_property: dict[str, Any],
        output_bounds: OutputCrownBounds,
) -> tuple[str, str]:
    property_type = test_property.get("type")
    if property_type != "adversarial":
        return "UNKNOWN", "CROWN early conclusion is currently implemented for adversarial properties only."

    output_indices = [int(var_index) for var_index, _ in test_property.get("output", [])]
    if not output_indices:
        return "UNKNOWN", "Adversarial property does not contain output disjuncts."

    if min(output_indices) < 0 or max(output_indices) >= len(output_bounds.lower_bounds):
        return (
            "UNKNOWN",
            "Adversarial output index is outside CROWN output bounds: {}".format(output_indices),
        )

    query_mode = test_property.get("_adversarial_query_mode", "disjunction")
    if query_mode == "bounds":
        output_specs = [
            (int(var_index), bounds)
            for var_index, bounds in test_property.get("output", [])
        ]

        def spec_is_always_satisfied(output_index: int, bounds: dict[str, Any]) -> bool:
            if "Lower" in bounds and output_bounds.lower_bounds[output_index] < float(bounds["Lower"]):
                return False
            if "Upper" in bounds and output_bounds.upper_bounds[output_index] > float(bounds["Upper"]):
                return False
            return True

        def spec_is_impossible(output_index: int, bounds: dict[str, Any]) -> bool:
            if "Lower" in bounds and output_bounds.upper_bounds[output_index] < float(bounds["Lower"]):
                return True
            if "Upper" in bounds and output_bounds.lower_bounds[output_index] > float(bounds["Upper"]):
                return True
            return False

        if all(spec_is_always_satisfied(output_index, bounds) for output_index, bounds in output_specs):
            return (
                "SAT",
                "CROWN proved every adversarial output bound is always satisfied.",
            )

        for output_index, bounds in output_specs:
            if spec_is_impossible(output_index, bounds):
                return (
                    "UNSAT",
                    "CROWN proved adversarial output {} cannot satisfy its bounds.".format(output_index),
                )

        return "UNKNOWN", "CROWN output bounds do not prove SAT or UNSAT."

    violation_operator = test_property.get("_adversarial_violation_operator", "ge")
    if violation_operator == "ge":
        output_specs = [
            (int(var_index), bounds)
            for var_index, bounds in test_property.get("output", [])
        ]
        for output_index, bounds in output_specs:
            threshold = float(bounds.get("Lower", 0.0))
            if output_bounds.lower_bounds[output_index] >= threshold:
                return (
                    "SAT",
                    "CROWN proved adversarial output {} is always >= {}.".format(
                        output_index,
                        threshold,
                    ),
                )

        if all(
                output_bounds.upper_bounds[output_index] < float(bounds.get("Lower", 0.0))
                for output_index, bounds in output_specs
        ):
            return (
                "UNSAT",
                "CROWN proved every adversarial output upper bound is below its threshold.",
            )
    elif violation_operator == "le":
        output_specs = [
            (int(var_index), bounds)
            for var_index, bounds in test_property.get("output", [])
        ]
        for output_index, bounds in output_specs:
            threshold = float(bounds.get("Upper", 0.0))
            if output_bounds.upper_bounds[output_index] <= threshold:
                return (
                    "SAT",
                    "CROWN proved adversarial output {} is always <= {}.".format(
                        output_index,
                        threshold,
                    ),
                )

        if all(
                output_bounds.lower_bounds[output_index] > float(bounds.get("Upper", 0.0))
                for output_index, bounds in output_specs
        ):
            return (
                "UNSAT",
                "CROWN proved every adversarial output lower bound is above its threshold.",
            )
    else:
        raise ValueError(
            "Unsupported adversarial violation operator '{}'.".format(violation_operator)
        )

    return "UNKNOWN", "CROWN output bounds do not prove SAT or UNSAT."


def _remove_edge_from_node(edge_list: list[Any], edge: Any) -> None:
    for index, current_edge in enumerate(edge_list):
        if current_edge is edge:
            del edge_list[index]
            return
        if (
            getattr(current_edge, "src", None) == getattr(edge, "src", None)
            and getattr(current_edge, "dest", None) == getattr(edge, "dest", None)
            and getattr(current_edge, "weight", None) == getattr(edge, "weight", None)
        ):
            del edge_list[index]
            return


def prune_dead_relu_neurons(
        network: Network,
        crown_bounds: HiddenLayerCrownBounds,
        preactivation_upper_threshold: float = 0.0,
) -> dict[str, Any]:
    pruned_by_layer = []
    skipped_layers = []
    total_pruned = 0
    hidden_layer_counter = 0

    for layer_index, layer in enumerate(network.layers[1:-1], start=1):
        if layer.type_name != "hidden":
            continue
        preactivation_uppers = crown_bounds.preactivation_upper_bounds_by_hidden_layer[hidden_layer_counter]
        if len(preactivation_uppers) != len(layer.nodes):
            raise ValueError(
                "Hidden layer {} has {} nodes but {} CROWN upper bounds.".format(
                    layer_index,
                    len(layer.nodes),
                    len(preactivation_uppers),
                )
            )

        dead_node_indices = [
            node_index
            for node_index, upper_bound in enumerate(preactivation_uppers)
            if float(upper_bound) < float(preactivation_upper_threshold)
        ]
        if dead_node_indices and len(dead_node_indices) >= len(layer.nodes):
            skipped_layers.append({
                "layer_index": layer_index,
                "reason": "all neurons were classified as dead; keeping the layer to avoid an empty hidden layer",
                "candidate_node_indices": dead_node_indices,
            })
            hidden_layer_counter += 1
            continue

        pruned_nodes = []
        for node_index in sorted(dead_node_indices, reverse=True):
            node = layer.nodes[node_index]
            for in_edge in list(node.in_edges):
                src_node = network.name2node_map[in_edge.src]
                _remove_edge_from_node(src_node.out_edges, in_edge)
            for out_edge in list(node.out_edges):
                dest_node = network.name2node_map[out_edge.dest]
                _remove_edge_from_node(dest_node.in_edges, out_edge)
            pruned_nodes.append({
                "node_index": node_index,
                "node_name": node.name,
                "preactivation_upper_bound": float(preactivation_uppers[node_index]),
            })
            del layer.nodes[node_index]

        if pruned_nodes:
            pruned_nodes.reverse()
            total_pruned += len(pruned_nodes)
            pruned_by_layer.append({
                "layer_index": layer_index,
                "pruned_nodes": pruned_nodes,
            })
        hidden_layer_counter += 1

    network.generate_name2node_map()
    network.weights = network.generate_weights()
    network._biases = network.generate_biases()
    network.biases = network.generate_biases()

    return {
        "total_pruned": total_pruned,
        "pruned_by_layer": pruned_by_layer,
        "skipped_layers": skipped_layers,
    }
