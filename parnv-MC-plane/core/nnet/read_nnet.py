import os

import numpy as np

import core.import_marabou
try:
    from maraboupy import MarabouNetworkNNet
    from maraboupy import MarabouNetworkONNX
except (ImportError, ModuleNotFoundError):
    MarabouNetworkNNet = None
    MarabouNetworkONNX = None

from core.configuration.consts import (
    PATH_TO_MARABOU_APPLICATIONS_ACAS_EXAMPLES,
)
from core.data_structures.Edge import Edge
from core.data_structures.ARNode import ARNode
from core.data_structures.Layer import Layer
from core.data_structures.Network import Network
from core.utils.activation_functions import relu


SUPPORTED_ONNX_CHAIN_OPS = {"Identity", "Flatten", "Gemm", "MatMul", "Add", "Relu"}


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


def _require_onnx():
    try:
        import onnx
        from onnx import helper, numpy_helper
    except Exception as exc:
        raise RuntimeError(
            "The onnx package is required for loading .onnx models. "
            "Please install it in the verification environment."
        ) from exc
    return onnx, helper, numpy_helper


def _tensor_shape_from_value_info(value_info):
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None

    dims = []
    for dim in tensor_type.shape.dim[1:]:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            return None
    return tuple(dims) if dims else None


def _get_initializer_array(initializers, name):
    if name not in initializers:
        raise ValueError("missing ONNX initializer '{}'".format(name))
    return np.asarray(initializers[name], dtype=np.float32)


def _consume_activation_if_present(nodes, index, current_tensor):
    if index >= len(nodes):
        return "linear", current_tensor, index

    node = nodes[index]
    if len(node.input) != 1 or node.input[0] != current_tensor:
        return "linear", current_tensor, index

    if node.op_type == "Relu":
        return "relu", node.output[0], index + 1
    return "linear", current_tensor, index


def _parse_gemm_node(node, current_tensor, initializers, helper):
    if len(node.input) < 2:
        raise ValueError("Gemm node must have at least two inputs")
    if node.input[0] != current_tensor:
        raise ValueError("Only simple chain-structured Gemm nodes are supported")

    attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
    alpha = float(attributes.get("alpha", 1.0))
    beta = float(attributes.get("beta", 1.0))
    trans_a = int(attributes.get("transA", 0))
    trans_b = int(attributes.get("transB", 0))

    if trans_a != 0:
        raise ValueError("Unsupported Gemm attribute transA != 0")

    weight_raw = _get_initializer_array(initializers, node.input[1])
    if weight_raw.ndim != 2:
        raise ValueError("Gemm weight must be rank-2")
    weight = alpha * weight_raw if trans_b == 1 else alpha * weight_raw.T

    if len(node.input) >= 3 and node.input[2]:
        bias = beta * _get_initializer_array(initializers, node.input[2]).reshape(-1)
    else:
        bias = np.zeros((weight.shape[0],), dtype=np.float32)

    return weight.astype(np.float32), bias.astype(np.float32), node.output[0]


def _parse_matmul_add_chain(nodes, index, current_tensor, initializers):
    node = nodes[index]
    if node.op_type != "MatMul":
        raise ValueError("Expected a MatMul node")
    if len(node.input) != 2:
        raise ValueError("MatMul must have exactly two inputs")

    if node.input[0] == current_tensor and node.input[1] in initializers:
        weight_raw = _get_initializer_array(initializers, node.input[1])
        weight = weight_raw.T
    elif node.input[1] == current_tensor and node.input[0] in initializers:
        weight_raw = _get_initializer_array(initializers, node.input[0])
        weight = weight_raw
    else:
        raise ValueError("MatMul must combine the current tensor with one constant weight matrix")

    if weight.ndim != 2:
        raise ValueError("MatMul weight must be rank-2")

    current_output = node.output[0]
    next_index = index + 1
    bias = np.zeros((weight.shape[0],), dtype=np.float32)

    if next_index < len(nodes):
        add_node = nodes[next_index]
        if add_node.op_type == "Add" and current_output in add_node.input:
            bias_name = add_node.input[0] if add_node.input[1] == current_output else add_node.input[1]
            if bias_name in initializers:
                bias = _get_initializer_array(initializers, bias_name).reshape(-1)
                current_output = add_node.output[0]
                next_index += 1

    return weight.astype(np.float32), bias.astype(np.float32), current_output, next_index


def _load_onnx_dense_layers(nnet_filename):
    onnx_lib, helper, numpy_helper = _require_onnx()
    model = onnx_lib.load(nnet_filename)
    graph = model.graph
    initializers = {
        initializer.name: numpy_helper.to_array(initializer).astype(np.float32)
        for initializer in graph.initializer
    }

    graph_inputs = [value for value in graph.input if value.name not in initializers]
    graph_outputs = list(graph.output)
    if len(graph_inputs) != 1:
        raise ValueError("Only single-input ONNX networks are supported")
    if len(graph_outputs) != 1:
        raise ValueError("Only single-output ONNX networks are supported")

    nodes = list(graph.node)
    for node in nodes:
        if node.op_type not in SUPPORTED_ONNX_CHAIN_OPS:
            raise ValueError(
                "Unsupported ONNX operator '{}'. Only sequential {} chains are supported".format(
                    node.op_type,
                    sorted(SUPPORTED_ONNX_CHAIN_OPS),
                )
            )

    current_tensor = graph_inputs[0].name
    input_shape = _tensor_shape_from_value_info(graph_inputs[0])
    flatten_input = False
    seen_linear = False
    weights = []
    biases = []
    activations = []
    index = 0

    while index < len(nodes):
        node = nodes[index]

        if node.op_type == "Identity":
            if len(node.input) != 1 or node.input[0] != current_tensor:
                raise ValueError("Identity nodes must preserve the current tensor in a simple chain")
            current_tensor = node.output[0]
            index += 1
            continue

        if node.op_type == "Flatten":
            if seen_linear:
                raise ValueError("Flatten is only supported before the first linear layer")
            if flatten_input:
                raise ValueError("Only one Flatten node is supported")
            if len(node.input) != 1 or node.input[0] != current_tensor:
                raise ValueError("Flatten must consume the current tensor in a simple chain")
            attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
            axis = int(attributes.get("axis", 1))
            if axis != 1:
                raise ValueError("Only Flatten(axis=1) is supported")
            flatten_input = True
            current_tensor = node.output[0]
            index += 1
            continue

        if node.op_type == "Gemm":
            weight, bias, current_tensor = _parse_gemm_node(
                node=node,
                current_tensor=current_tensor,
                initializers=initializers,
                helper=helper,
            )
            index += 1
        elif node.op_type == "MatMul":
            weight, bias, current_tensor, index = _parse_matmul_add_chain(
                nodes=nodes,
                index=index,
                current_tensor=current_tensor,
                initializers=initializers,
            )
        else:
            raise ValueError(
                "Unexpected operator '{}'. Only sequential Flatten -> Linear -> ReLU chains are supported".format(
                    node.op_type
                )
            )

        seen_linear = True
        activation, current_tensor, index = _consume_activation_if_present(
            nodes=nodes,
            index=index,
            current_tensor=current_tensor,
        )
        weights.append(weight)
        biases.append(bias)
        activations.append(activation)

    if current_tensor != graph_outputs[0].name:
        raise ValueError("The ONNX graph is not a simple single-chain network")
    if not weights:
        raise ValueError("No linear layers were found in the ONNX graph")

    if any(activation != "relu" for activation in activations[:-1]):
        raise ValueError(
            "This verifier currently supports ReLU hidden layers only; found hidden activations {}".format(
                activations[:-1]
            )
        )
    if activations[-1] != "linear":
        raise ValueError(
            "This verifier currently supports a linear output layer only; found '{}'".format(
                activations[-1]
            )
        )

    if input_shape is not None:
        first_input_dim = int(weights[0].shape[1])
        if flatten_input and int(np.prod(input_shape)) != first_input_dim:
            raise ValueError(
                "Flattened input shape {} does not match the first linear layer input dimension {}".format(
                    input_shape,
                    first_input_dim,
                )
            )
        if (not flatten_input) and len(input_shape) == 1 and input_shape[0] != first_input_dim:
            raise ValueError(
                "Input shape {} does not match the first linear layer input dimension {}".format(
                    input_shape,
                    first_input_dim,
                )
            )

    return weights, biases


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


def network_from_onnx_file(nnet_filename: str) -> Network:
    """
    generate Net instance which is equivalent to given onnx formatted network
    @nnet_filename fullpath of onnx file
    :return: Network object
    """
    acasxu_net = None
    if MarabouNetworkONNX is not None:
        try:
            acasxu_net = MarabouNetworkONNX.MarabouNetworkONNX(filename=nnet_filename)
        except Exception:
            acasxu_net = None

    if acasxu_net is not None and \
            hasattr(acasxu_net, "weights") and \
            hasattr(acasxu_net, "biases") and \
            hasattr(acasxu_net, "layerSizes"):
        return _build_network_from_weights_and_biases(
            weights=acasxu_net.weights,
            biases=acasxu_net.biases,
            acasxu_net=acasxu_net,
        )

    weights, biases = _load_onnx_dense_layers(nnet_filename)
    return _build_network_from_weights_and_biases(
        weights=weights,
        biases=biases,
        acasxu_net=acasxu_net,
    )


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
