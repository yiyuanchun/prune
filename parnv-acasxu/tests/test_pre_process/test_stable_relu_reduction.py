import numpy as np

from core.nnet.read_nnet import _build_network_from_weights_and_biases
from core.pre_process.crown_bounds import HiddenLayerCrownBounds
from core.pre_process.stable_relu_reduction import reduce_stable_relu_neurons


def _build_test_network():
    # x in [-1, 1]^2. The first four ReLUs are always active; the last is always inactive.
    weights = [
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
        ],
        [
            [1.0, 2.0, 3.0, 4.0, 7.0],
            [-2.0, 1.0, 0.5, 1.0, -5.0],
        ],
    ]
    biases = [
        [3.0, 3.0, 3.0, 3.0, -3.0],
        [0.25, -0.75],
    ]
    network = _build_network_from_weights_and_biases(weights, biases, acasxu_net=None)
    for node in network.layers[0].nodes:
        node.lower_bound = -1.0
        node.upper_bound = 1.0
    bounds = HiddenLayerCrownBounds(
        preactivation_lower_bounds_by_hidden_layer=[[2.0, 2.0, 1.0, 1.0, -5.0]],
        preactivation_upper_bounds_by_hidden_layer=[[4.0, 4.0, 5.0, 5.0, -1.0]],
        postactivation_lower_bounds_by_hidden_layer=[[2.0, 2.0, 1.0, 1.0, 0.0]],
        postactivation_upper_bounds_by_hidden_layer=[[4.0, 4.0, 5.0, 5.0, 0.0]],
    )
    return network, bounds


def _max_random_error(reference, candidate, samples=256):
    rng = np.random.default_rng(7)
    max_error = 0.0
    for _ in range(samples):
        values = rng.uniform(-1.0, 1.0, size=2)
        input_assignment = {index: float(value) for index, value in enumerate(values)}
        reference_output = np.asarray(reference.speedy_evaluate(input_assignment), dtype=float)
        candidate_output = np.asarray(candidate.speedy_evaluate(input_assignment), dtype=float)
        max_error = max(max_error, float(np.max(np.abs(reference_output - candidate_output))))
    return max_error


def test_full_rednet_reduction_removes_inactive_and_reconstructs_active_relu_neurons():
    network, bounds = _build_test_network()
    reduced, report = reduce_stable_relu_neurons(
        network,
        bounds,
        stability_tolerance=1e-9,
        activation_margin=1e-9,
        reconstruct_stably_active=True,
    )

    assert report["original_relu_count"] == 5
    assert report["reduced_relu_count"] == 2
    assert report["total_inactive_removed"] == 1
    assert report["total_active_eliminated"] == 4
    assert report["total_reconstructed_units"] == 2
    assert report["reduction_ratio"] == 2.5
    assert len(reduced.layers[1].nodes) == 2
    assert _max_random_error(network, reduced) < 1e-10


def test_inactive_only_mode_is_exact_and_keeps_active_relu_neurons():
    network, bounds = _build_test_network()
    reduced, report = reduce_stable_relu_neurons(
        network,
        bounds,
        reconstruct_stably_active=False,
    )

    assert report["original_relu_count"] == 5
    assert report["reduced_relu_count"] == 4
    assert report["total_inactive_removed"] == 1
    assert report["total_active_eliminated"] == 0
    assert len(reduced.layers[1].nodes) == 4
    assert _max_random_error(network, reduced) < 1e-10
