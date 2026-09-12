import os

import numpy as np

import core.import_marabou
try:
    from maraboupy import MarabouNetworkNNet
except (ImportError, ModuleNotFoundError):
    MarabouNetworkNNet = None

from core.configuration.consts import (
    PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES,
)
from core.data_structures.Edge import Edge
from core.data_structures.ARNode import ARNode
from core.data_structures.Layer import Layer
from core.data_structures.Network import Network
from core.utils.activation_functions import relu


def _parse_nnet_csv_line(line, value_type=float):
    return [value_type(value) for value in line.strip().split(",") if value.strip() != ""]


def read_nnet_parameters(nnet_filename: str) -> dict:
    with open(nnet_filename, "r", encoding="utf-8") as nnet_file:
        line = nnet_file.readline()
        while line and line.strip().startswith("//"):
            line = nnet_file.readline()
        if not line:
            raise ValueError("Empty .nnet file: {}".format(nnet_filename))

        num_layers, input_size, output_size, max_layer_size = _parse_nnet_csv_line(line, int)
        layer_sizes = _parse_nnet_csv_line(nnet_file.readline(), int)
        symmetric = _parse_nnet_csv_line(nnet_file.readline(), int)[0]
        input_minimums = _parse_nnet_csv_line(nnet_file.readline(), float)
        input_maximums = _parse_nnet_csv_line(nnet_file.readline(), float)
        means = _parse_nnet_csv_line(nnet_file.readline(), float)
        ranges = _parse_nnet_csv_line(nnet_file.readline(), float)

        weights = []
        biases = []
        for layer_index in range(num_layers):
            previous_layer_size = layer_sizes[layer_index]
            current_layer_size = layer_sizes[layer_index + 1]
            layer_weights = []
            for _ in range(current_layer_size):
                row = _parse_nnet_csv_line(nnet_file.readline(), float)
                if len(row) != previous_layer_size:
                    raise ValueError(
                        "Layer {} weight row has width {}, expected {}.".format(
                            layer_index, len(row), previous_layer_size
                        )
                    )
                layer_weights.append(row)
            layer_biases = []
            for _ in range(current_layer_size):
                values = _parse_nnet_csv_line(nnet_file.readline(), float)
                if not values:
                    raise ValueError("Missing bias value in layer {}.".format(layer_index))
                layer_biases.append(values[0])
            weights.append(layer_weights)
            biases.append(layer_biases)

    if len(layer_sizes) != num_layers + 1:
        raise ValueError(
            ".nnet layerSizes length {} does not match numLayers {}.".format(
                len(layer_sizes), num_layers
            )
        )
    if len(input_minimums) != input_size or len(input_maximums) != input_size:
        raise ValueError("Input min/max metadata width does not match input size.")
    if len(means) != input_size + 1 or len(ranges) != input_size + 1:
        raise ValueError("Normalization metadata width does not match input/output sizes.")

    return {
        "numLayers": num_layers,
        "inputSize": input_size,
        "outputSize": output_size,
        "maxLayersize": max_layer_size,
        "layerSizes": layer_sizes,
        "symmetric": symmetric,
        "inputMinimums": input_minimums,
        "inputMaximums": input_maximums,
        "inputMeans": means[:-1],
        "inputRanges": ranges[:-1],
        "outputMean": means[-1],
        "outputRange": ranges[-1],
        "weights": weights,
        "biases": biases,
    }


def write_nnet_file(
        nnet_filename: str,
        weights,
        biases,
        metadata: dict | None = None,
) -> None:
    if len(weights) != len(biases):
        raise ValueError("weights and biases must describe the same number of layers")
    if not weights:
        raise ValueError("Cannot write an empty .nnet network")

    normalized_weights = [
        [list(map(float, row)) for row in layer_weights]
        for layer_weights in weights
    ]
    normalized_biases = [list(map(float, layer_biases)) for layer_biases in biases]
    input_size = len(normalized_weights[0][0])
    layer_sizes = [input_size] + [len(layer_biases) for layer_biases in normalized_biases]
    output_size = layer_sizes[-1]
    num_layers = len(normalized_weights)
    max_layer_size = max(layer_sizes)
    metadata = metadata or {}

    input_minimums = list(map(float, metadata.get("inputMinimums", [-1.0e20] * input_size)))
    input_maximums = list(map(float, metadata.get("inputMaximums", [1.0e20] * input_size)))
    input_means = list(map(float, metadata.get("inputMeans", [0.0] * input_size)))
    input_ranges = list(map(float, metadata.get("inputRanges", [1.0] * input_size)))
    output_mean = float(metadata.get("outputMean", 0.0))
    output_range = float(metadata.get("outputRange", 1.0))
    symmetric = int(metadata.get("symmetric", 0))

    if not (
            len(input_minimums)
            == len(input_maximums)
            == len(input_means)
            == len(input_ranges)
            == input_size
    ):
        raise ValueError("Input normalization metadata width does not match input size.")

    with open(nnet_filename, "w", encoding="utf-8") as nnet_file:
        nnet_file.write("// Neural Network File Format by Kyle Julian, Stanford 2016\n")
        nnet_file.write("// Network written by parnv-acasxu raw CEGAR flow\n")
        nnet_file.write("{},{},{},{},\n".format(num_layers, input_size, output_size, max_layer_size))
        nnet_file.write(",".join(str(size) for size in layer_sizes) + ",\n")
        nnet_file.write("{},\n".format(symmetric))
        nnet_file.write(",".join(str(value) for value in input_minimums) + ",\n")
        nnet_file.write(",".join(str(value) for value in input_maximums) + ",\n")
        nnet_file.write(",".join(str(value) for value in input_means + [output_mean]) + ",\n")
        nnet_file.write(",".join(str(value) for value in input_ranges + [output_range]) + ",\n")

        for layer_weights, layer_biases in zip(normalized_weights, normalized_biases):
            for row in layer_weights:
                nnet_file.write(",".join("{:.12e}".format(value) for value in row) + ",\n")
            for bias in layer_biases:
                nnet_file.write("{:.12e},\n".format(bias))


def _build_network_from_weights_and_biases(weights, biases, acasxu_net=None) -> Network:
    """
    Build NARv's internal graph from dense layer matrices.
    weights[layer][dest][src] is the edge weight from src in layer `layer`
    to dest in layer `layer + 1`.
    """
    if len(weights) != len(biases):
        raise ValueError("weights and biases must describe the same number of layers")
    if not weights:
        raise ValueError("network must contain at least one linear layer")

    edges = []
    for layer_index, layer_weights in enumerate(weights):
        edges.append([])
        for dest_index, node_weights in enumerate(layer_weights):
            edges[layer_index].append([])
            for src_index, weight in enumerate(node_weights):
                edge = Edge(
                    src="x_{}_{}".format(layer_index, src_index),
                    dest="x_{}_{}".format(layer_index + 1, dest_index),
                    weight=float(weight),
                )
                edges[layer_index][dest_index].append(edge)

    nodes = []
    name2node_map = {}
    for layer_index, layer in enumerate(edges):
        nodes.append([])
        for node_edges in layer:
            for edge in node_edges:
                if edge.src in name2node_map:
                    continue
                src_node = ARNode(
                    name=edge.src,
                    ar_type=None,
                    in_edges=[],
                    out_edges=[],
                    activation_func=relu,
                    bias=0.0,
                    upper_bound=0.0,
                    lower_bound=0.0,
                )
                nodes[layer_index].append(src_node)
                name2node_map[edge.src] = src_node

    nodes.append([])
    output_layer_index = len(edges)
    for node_edges in edges[-1]:
        for edge in node_edges:
            if edge.dest in name2node_map:
                continue
            dest_node = ARNode(
                name=edge.dest,
                ar_type=None,
                in_edges=[],
                out_edges=[],
                activation_func=relu,
                bias=0.0,
                upper_bound=0.0,
                lower_bound=0.0,
            )
            nodes[output_layer_index].append(dest_node)
            name2node_map[edge.dest] = dest_node

    for layer in edges:
        for node_edges in layer:
            for edge in node_edges:
                src_node = name2node_map[edge.src]
                dest_node = name2node_map[edge.dest]
                src_node.out_edges.append(edge)
                dest_node.in_edges.append(edge)

    layers = []
    for layer_index, layer_nodes in enumerate(nodes):
        if layer_index == 0:
            type_name = "input"
        elif layer_index == len(nodes) - 1:
            type_name = "output"
        else:
            type_name = "hidden"
        layers.append(Layer(type_name=type_name, nodes=layer_nodes))

    for layer_index, layer_biases in enumerate(biases):
        layer = layers[layer_index + 1]
        for node_index, node in enumerate(layer.nodes):
            node.bias = float(layer_biases[node_index])

    normalized_weights = [
        [list(map(float, node_weights)) for node_weights in layer_weights]
        for layer_weights in weights
    ]
    normalized_biases = [list(map(float, layer_biases)) for layer_biases in biases]
    return Network(
        layers=layers,
        weights=normalized_weights,
        biases=normalized_biases,
        acasxu_net=acasxu_net,
    )


def network_from_nnet_file(nnet_filename: str) -> Network:
    """
    generate Net instance which is equivalent to given nnet formatted network
    @nnet_filename fullpath of nnet file, see maraboupy.MarabouNetworkNNet
    under the root dir of git marabou project
    :return: Network object
    """
    nnet_parameters = read_nnet_parameters(nnet_filename)
    acasxu_net = None
    weights = nnet_parameters["weights"]
    biases = nnet_parameters["biases"]
    if MarabouNetworkNNet is not None:
        acasxu_net = MarabouNetworkNNet.MarabouNetworkNNet(filename=nnet_filename)
        weights = acasxu_net.weights
        biases = acasxu_net.biases
    network = _build_network_from_weights_and_biases(
        weights=weights,
        biases=biases,
        acasxu_net=acasxu_net,
    )
    network.nnet_metadata = nnet_parameters
    return network


def get_all_acas_nets(indices=None):
    """
    :param indices: list of indices of nnet files, if None return all
    :return: list of Net objects of acas networks in the relevant indices
    """
    nnet_dir = PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES
    l = []
    for i, filename in enumerate(os.listdir(nnet_dir)):
        if i not in indices:
            continue
        nnet_filename = os.path.join(nnet_dir, filename)
        l.append(network_from_nnet_file(nnet_filename))
    return l
