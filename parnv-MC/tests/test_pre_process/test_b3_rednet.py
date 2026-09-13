import itertools
import copy
import numpy as np
import pytest

from core.nnet.read_nnet import _build_network_from_weights_and_biases, network_from_nnet_file, write_nnet_file
from core.pre_process.crown_bounds import HiddenLayerCrownBounds
from core.pre_process.stable_relu_reduction import reduce_stable_relu_neurons
from core.pre_process.rednet_pipeline import check_input_output_equivalence


def fixture_network(widths=(2, 8, 7, 6, 2)):
    rng = np.random.default_rng(21)
    weights = [rng.uniform(-0.1, 0.1, (b, a)) for a, b in zip(widths[:-1], widths[1:])]
    biases = [np.full(n, 3.) for n in widths[1:-1]] + [np.array([0.4, -0.5])]
    # One dead neuron in each layer, and signed outgoing connections.
    for bias in biases[:-1]:
        bias[-1] = -3.
    net = _build_network_from_weights_and_biases(weights, biases)
    for node in net.layers[0].nodes:
        node.lower_bound, node.upper_bound = -1., 1.
    lower, upper = np.full(2, -1.), np.ones(2)
    pl, pu, al, au = [], [], [], []
    for w, b in zip(weights[:-1], biases[:-1]):
        lo = np.maximum(w, 0)@lower + np.minimum(w, 0)@upper + b
        hi = np.maximum(w, 0)@upper + np.minimum(w, 0)@lower + b
        lower, upper = np.maximum(lo, 0), np.maximum(hi, 0)
        pl.append(lo.tolist()); pu.append(hi.tolist())
        al.append(lower.tolist()); au.append(upper.tolist())
    return net, HiddenLayerCrownBounds(pl, pu, al, au)


def test_backward_reconstruction_and_serialization_at_box_corners(tmp_path):
    net, bounds = fixture_network()
    reduced, report = reduce_stable_relu_neurons(net, bounds)
    assert all(p['reconstructed_stable_active'] for p in report['per_layer'])
    assert report['total_inactive_removed'] == 3
    assert report['reduced_relu_count'] == 6
    assert report['relu_removed'] == report['total_inactive_removed'] + report['total_active_eliminated'] - report['total_reconstructed_units']
    path = tmp_path/'reduced.nnet'
    write_nnet_file(str(path), reduced.generate_weights(), reduced.generate_biases())
    reloaded = network_from_nnet_file(str(path))
    for point in itertools.product([-1., 0., 1.], repeat=2):
        x = dict(enumerate(point))
        np.testing.assert_allclose(net.speedy_evaluate(x), reloaded.speedy_evaluate(x), atol=1e-10, rtol=0)
    assert check_input_output_equivalence(net, reloaded, samples=128, tolerance=1e-10)['passed']


def test_inactive_only_keeps_active_units():
    net, bounds = fixture_network()
    reduced, report = reduce_stable_relu_neurons(net, bounds, reconstruct_stably_active=False)
    assert report['total_active_eliminated'] == report['total_reconstructed_units'] == 0
    assert report['relu_removed'] == 3
    assert check_input_output_equivalence(net, reduced, tolerance=1e-12)['passed']


def test_error_statistics_are_pointwise_absolute_errors():
    net, _ = fixture_network()
    candidate = copy.deepcopy(net)
    candidate.biases[-1][0] += 0.125
    candidate.biases[-1][1] -= 0.125
    result = check_input_output_equivalence(net, candidate, samples=128, tolerance=0.01)
    assert not result['passed']
    assert result['mean_abs_error'] == pytest.approx(0.125)
    assert result['max_abs_error'] == pytest.approx(0.125)
    assert result['variance_abs_error'] < 1e-25


def test_nonfinite_output_never_passes_equivalence():
    net, _ = fixture_network()
    candidate = copy.deepcopy(net)
    candidate.biases[-1][0] = float('nan')
    with pytest.raises(ValueError, match='Non-finite'):
        check_input_output_equivalence(net, candidate)


def test_zero_samples_and_invalid_bounds_rejected():
    net, bounds = fixture_network()
    with pytest.raises(ValueError):
        check_input_output_equivalence(net, net, samples=0)
    bounds.preactivation_lower_bounds_by_hidden_layer[0][0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        reduce_stable_relu_neurons(net, bounds)
