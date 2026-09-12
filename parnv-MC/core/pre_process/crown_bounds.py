from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import types

from core.data_structures.Network import Network


@dataclass(frozen=True)
class HiddenLayerCrownBounds:
    preactivation_lower_bounds_by_hidden_layer: list[list[float]]
    preactivation_upper_bounds_by_hidden_layer: list[list[float]]
    postactivation_lower_bounds_by_hidden_layer: list[list[float]]
    postactivation_upper_bounds_by_hidden_layer: list[list[float]]


@dataclass(frozen=True)
class OutputCrownBounds:
    lower_bounds: list[float]
    upper_bounds: list[float]


_AUTO_LIRPA_IMPORT_ERROR: Exception | None = None
_AUTO_LIRPA_BOUNDED_MODULE = None
_AUTO_LIRPA_BOUNDED_TENSOR = None
_AUTO_LIRPA_PERTURBATION_LP_NORM = None


def _ensure_appdirs_fallback() -> None:
    import os
    import sys

    if "appdirs" in sys.modules:
        return

    try:
        import appdirs  # type: ignore  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    fallback_module = types.ModuleType("appdirs")

    def user_data_dir(appname: str | None = None, appauthor: str | None = None, version: str | None = None) -> str:
        app_name = str(appname or "auto_LiRPA")
        base_dir = os.environ.get("XDG_DATA_HOME")
        if not base_dir:
            base_dir = os.path.join(os.path.expanduser("~"), ".local", "share")
        path = os.path.join(base_dir, app_name)
        if version:
            path = os.path.join(path, str(version))
        return path

    fallback_module.user_data_dir = user_data_dir  # type: ignore[attr-defined]
    sys.modules["appdirs"] = fallback_module


def _load_auto_lirpa() -> tuple[Any, Any, Any]:
    global _AUTO_LIRPA_IMPORT_ERROR
    global _AUTO_LIRPA_BOUNDED_MODULE
    global _AUTO_LIRPA_BOUNDED_TENSOR
    global _AUTO_LIRPA_PERTURBATION_LP_NORM

    if (
        _AUTO_LIRPA_BOUNDED_MODULE is not None
        and _AUTO_LIRPA_BOUNDED_TENSOR is not None
        and _AUTO_LIRPA_PERTURBATION_LP_NORM is not None
    ):
        return (
            _AUTO_LIRPA_BOUNDED_MODULE,
            _AUTO_LIRPA_BOUNDED_TENSOR,
            _AUTO_LIRPA_PERTURBATION_LP_NORM,
        )
    if _AUTO_LIRPA_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Failed to import the local auto_LiRPA package required for CROWN hidden-layer bounds."
        ) from _AUTO_LIRPA_IMPORT_ERROR

    import sys

    auto_lirpa_root = Path(__file__).resolve().parents[2] / "auto_LiRPA-master"
    if not auto_lirpa_root.exists():
        raise FileNotFoundError(
            "Local auto_LiRPA checkout was not found at {}.".format(auto_lirpa_root)
        )
    if str(auto_lirpa_root) not in sys.path:
        sys.path.insert(0, str(auto_lirpa_root))
    _ensure_appdirs_fallback()

    try:
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    except Exception as exc:
        _AUTO_LIRPA_IMPORT_ERROR = exc
        raise RuntimeError(
            "Failed to import the local auto_LiRPA package required for CROWN hidden-layer bounds."
        ) from exc

    _AUTO_LIRPA_BOUNDED_MODULE = BoundedModule
    _AUTO_LIRPA_BOUNDED_TENSOR = BoundedTensor
    _AUTO_LIRPA_PERTURBATION_LP_NORM = PerturbationLpNorm
    return BoundedModule, BoundedTensor, PerturbationLpNorm


def _flatten_single_batch_bound(bound: Any) -> list[float]:
    bound_tensor = bound.detach().float().cpu()
    if int(bound_tensor.shape[0]) != 1:
        raise ValueError(
            "Expected a single-batch bound tensor, but received shape {}.".format(tuple(bound_tensor.shape))
        )
    return [float(value) for value in bound_tensor.reshape(bound_tensor.shape[0], -1)[0].tolist()]


def _network_to_torch_sequential(network: Network) -> Any:
    import torch
    from torch import nn

    modules = []
    weights = network.generate_weights()
    biases = network.generate_biases()
    for layer_index, (layer_weights, layer_biases) in enumerate(zip(weights, biases)):
        weight_tensor = torch.as_tensor(layer_weights, dtype=torch.float32)
        bias_tensor = torch.as_tensor(layer_biases, dtype=torch.float32)
        if weight_tensor.ndim != 2:
            raise ValueError("Layer {} weight matrix must be rank-2.".format(layer_index + 1))
        if bias_tensor.ndim != 1:
            raise ValueError("Layer {} bias vector must be rank-1.".format(layer_index + 1))
        if int(weight_tensor.shape[0]) != int(bias_tensor.numel()):
            raise ValueError(
                "Layer {} weight/bias output sizes differ: {} vs {}.".format(
                    layer_index + 1, int(weight_tensor.shape[0]), int(bias_tensor.numel())
                )
            )

        linear = nn.Linear(
            in_features=int(weight_tensor.shape[1]),
            out_features=int(weight_tensor.shape[0]),
            bias=True,
        )
        with torch.no_grad():
            linear.weight.copy_(weight_tensor)
            linear.bias.copy_(bias_tensor)
        modules.append(linear)
        if network.layers[layer_index + 1].type_name == "hidden":
            modules.append(nn.ReLU())

    model = nn.Sequential(*modules)
    model.eval()
    return model


def _input_bounds_from_network(network: Network) -> tuple[list[float], list[float]]:
    input_layer = network.layers[0]
    lower_bounds = [float(node.lower_bound) for node in input_layer.nodes]
    upper_bounds = [float(node.upper_bound) for node in input_layer.nodes]
    return lower_bounds, upper_bounds


def _build_prefix_model(model: Any, module_count: int) -> Any:
    from torch import nn

    if module_count <= 0:
        raise ValueError("module_count must be positive.")
    modules = list(model.children())
    if module_count > len(modules):
        raise ValueError(
            "Cannot build prefix model with {} modules from a {}-module network.".format(
                module_count, len(modules)
            )
        )
    prefix_model = nn.Sequential(*(copy.deepcopy(module) for module in modules[:module_count]))
    prefix_model.eval()
    return prefix_model


def _compute_crown_output_bounds(
    model: Any,
    input_lower_flat: list[float],
    input_upper_flat: list[float],
) -> tuple[list[float], list[float]]:
    import torch

    BoundedModule, BoundedTensor, PerturbationLpNorm = _load_auto_lirpa()
    input_lower = torch.as_tensor(input_lower_flat, dtype=torch.float32).reshape(1, -1)
    input_upper = torch.as_tensor(input_upper_flat, dtype=torch.float32).reshape(1, -1)
    if input_lower.shape != input_upper.shape:
        raise ValueError(
            "Input lower/upper bound shapes differ: {} vs {}.".format(
                tuple(input_lower.shape), tuple(input_upper.shape)
            )
        )

    input_center = ((input_lower + input_upper) / 2.0).detach()
    bounded_model = BoundedModule(copy.deepcopy(model), input_center)
    perturbation = PerturbationLpNorm(norm=float("inf"), x_L=input_lower, x_U=input_upper)
    bounded_input = BoundedTensor(input_center, perturbation)
    lower_bound, upper_bound = bounded_model.compute_bounds(x=(bounded_input,), method="CROWN")
    return _flatten_single_batch_bound(lower_bound), _flatten_single_batch_bound(upper_bound)


def compute_crown_hidden_layer_bounds(network: Network) -> HiddenLayerCrownBounds:
    model = _network_to_torch_sequential(network)
    input_lower, input_upper = _input_bounds_from_network(network)

    hidden_layers = network.layers[1:-1]
    preactivation_lower_bounds_by_hidden_layer: list[list[float]] = []
    preactivation_upper_bounds_by_hidden_layer: list[list[float]] = []
    postactivation_lower_bounds_by_hidden_layer: list[list[float]] = []
    postactivation_upper_bounds_by_hidden_layer: list[list[float]] = []

    for hidden_layer_index, layer in enumerate(hidden_layers):
        if layer.type_name != "hidden":
            continue
        preactivation_module_count = hidden_layer_index * 2 + 1
        postactivation_module_count = preactivation_module_count + 1

        preactivation_model = _build_prefix_model(model, preactivation_module_count)
        preactivation_lower, preactivation_upper = _compute_crown_output_bounds(
            model=preactivation_model,
            input_lower_flat=input_lower,
            input_upper_flat=input_upper,
        )
        postactivation_model = _build_prefix_model(model, postactivation_module_count)
        postactivation_lower, postactivation_upper = _compute_crown_output_bounds(
            model=postactivation_model,
            input_lower_flat=input_lower,
            input_upper_flat=input_upper,
        )

        expected_width = len(layer.nodes)
        for label, values in (
            ("preactivation lower", preactivation_lower),
            ("preactivation upper", preactivation_upper),
            ("postactivation lower", postactivation_lower),
            ("postactivation upper", postactivation_upper),
        ):
            if len(values) != expected_width:
                raise ValueError(
                    "CROWN {} bound width for hidden layer {} is {}, expected {}.".format(
                        label, hidden_layer_index, len(values), expected_width
                    )
                )

        preactivation_lower_bounds_by_hidden_layer.append(preactivation_lower)
        preactivation_upper_bounds_by_hidden_layer.append(preactivation_upper)
        postactivation_lower_bounds_by_hidden_layer.append(postactivation_lower)
        postactivation_upper_bounds_by_hidden_layer.append(postactivation_upper)

    return HiddenLayerCrownBounds(
        preactivation_lower_bounds_by_hidden_layer=preactivation_lower_bounds_by_hidden_layer,
        preactivation_upper_bounds_by_hidden_layer=preactivation_upper_bounds_by_hidden_layer,
        postactivation_lower_bounds_by_hidden_layer=postactivation_lower_bounds_by_hidden_layer,
        postactivation_upper_bounds_by_hidden_layer=postactivation_upper_bounds_by_hidden_layer,
    )


def compute_crown_output_bounds(network: Network) -> OutputCrownBounds:
    model = _network_to_torch_sequential(network)
    input_lower, input_upper = _input_bounds_from_network(network)
    lower_bounds, upper_bounds = _compute_crown_output_bounds(
        model=model,
        input_lower_flat=input_lower,
        input_upper_flat=input_upper,
    )
    expected_width = len(network.layers[-1].nodes)
    if len(lower_bounds) != expected_width or len(upper_bounds) != expected_width:
        raise ValueError(
            "CROWN output bound width is ({}, {}), expected {}.".format(
                len(lower_bounds),
                len(upper_bounds),
                expected_width,
            )
        )
    return OutputCrownBounds(lower_bounds=lower_bounds, upper_bounds=upper_bounds)


def apply_crown_hidden_layer_bounds(network: Network) -> HiddenLayerCrownBounds:
    bounds = compute_crown_hidden_layer_bounds(network)
    hidden_layer_counter = 0
    for layer in network.layers[1:-1]:
        if layer.type_name != "hidden":
            continue
        pre_lowers = bounds.preactivation_lower_bounds_by_hidden_layer[hidden_layer_counter]
        pre_uppers = bounds.preactivation_upper_bounds_by_hidden_layer[hidden_layer_counter]
        post_lowers = bounds.postactivation_lower_bounds_by_hidden_layer[hidden_layer_counter]
        post_uppers = bounds.postactivation_upper_bounds_by_hidden_layer[hidden_layer_counter]
        for node_index, node in enumerate(layer.nodes):
            node.preactivation_lower_bound = float(pre_lowers[node_index])
            node.preactivation_upper_bound = float(pre_uppers[node_index])
            node.lower_bound = float(post_lowers[node_index])
            node.upper_bound = float(post_uppers[node_index])
        hidden_layer_counter += 1
    return bounds
