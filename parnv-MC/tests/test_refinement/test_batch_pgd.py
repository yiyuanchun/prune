"""Run this suite with each variant on PYTHONPATH; formal calls are scripted.

Controller tests use the real undo/refine/finalize functions and inspect their
event ordering. Numerical PGD tests use real graph evaluation and gradients.
"""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.cegar import raw_cegar as c
from core.cegar.pgd_search import PGDConfig, search_counterexample, violation_objective
from core.nnet.read_nnet import _build_network_from_weights_and_biases as build


def make_state(count):
    records = [dict(layer=1, merge_id=i, layer_weights_before=[[i]],
                    layer_biases_before=[i], next_layer_weights_before=[[i + 1]],
                    layer_ids_before=[str(i)], layer_labels_before=['inc'],
                    old_neurons=[str(i)], merged_neuron='m' + str(i)) for i in range(count)]
    return c.RawCegarState(weights=[np.ones((1, 1)), np.ones((1, 1))],
        biases=[np.ones(1), np.zeros(1)], metadata={}, input_bounds=([0.], [1.]),
        ids_by_layer=[['in'], ['merged'], ['out']], labels={1: ['inc']},
        mapping={}, merge_log_stack=copy.deepcopy(records), merge_log=records)


@pytest.mark.parametrize('count,budget,extra,expected,extra_done', [
    (9, None, 3, 5, 3), (3, None, 3, 3, 1), (0, None, 3, 0, 0),
    (9, 0, 3, 0, 0), (9, 1, 3, 1, 0), (9, 3, 3, 3, 1), (9, None, 0, 2, 0),
])
def test_batch_restoration(monkeypatch, count, budget, extra, expected, extra_done):
    state = make_state(count)
    before = copy.deepcopy(state)
    monkeypatch.setattr(c, 'build_network_from_state', lambda s: s)
    monkeypatch.setattr(c, 'is_counterexample_on_current_network',
                        lambda **kw: len(state.refinement_log) < 2)
    result = c.refine_by_undo_merge(None, {0: .1}, {}, state.merge_log_stack,
        state.labels, state.mapping, state=state, max_steps=budget, extra_refinement_merges=extra)
    assert len(result['steps']) == expected
    assert result['extra_steps'] == extra_done
    assert len(state.merge_log_stack) == count - expected
    assert state.merge_log == before.merge_log
    assert state.refinement_log == result['steps']
    if expected:
        last = before.merge_log_stack[-expected]
        np.testing.assert_array_equal(state.weights[0], last['layer_weights_before'])
        np.testing.assert_array_equal(state.weights[1], last['next_layer_weights_before'])
        np.testing.assert_array_equal(state.biases[0], last['layer_biases_before'])
        assert state.ids_by_layer[1] == last['layer_ids_before']
        assert state.labels[1] == last['layer_labels_before']
        assert state.mapping['current_ids_by_layer'] == state.ids_by_layer
        assert sum(r['extra_refinement'] for r in result['steps']) == extra_done
    else:
        np.testing.assert_array_equal(state.weights[0], before.weights[0])


@pytest.mark.parametrize('crown,formal,pgd,count,budget,wanted,events,refinements', [
    ('UNSAT', [], [], 8, None, 'VERIFIED', '', 0),
    ('UNKNOWN', ['unsat'], [], 8, None, 'VERIFIED', 'F', 0),
    ('UNKNOWN', [.9], [], 8, None, 'UNSAFE', 'F', 0),
    ('UNKNOWN', [.1, 'unsat'], [None], 8, None, 'VERIFIED', 'FPF', 4),
    ('UNKNOWN', [.1], [.9], 8, None, 'UNSAFE', 'FP', 4),
    ('UNKNOWN', [.1, 'unsat'], [.4, None], 8, None, 'VERIFIED', 'FPPF', 8),
    ('UNKNOWN', [.1, 'unsat'], [.4], 2, None, 'VERIFIED', 'FPF', 2),
    ('UNKNOWN', [.1], [], 0, None, 'UNKNOWN', 'F', 0),
    ('UNKNOWN', [.1], [], 8, 0, 'UNKNOWN', 'F', 0),
    ('UNKNOWN', [.1, .1], [.4], 8, 2, 'UNKNOWN', 'FPF', 2),
])
def test_controller(monkeypatch, tmp_path, crown, formal, pgd, count, budget, wanted, events, refinements,
                    initial_candidate=None):
    # Commit 2bba107 enables initial PGD in the MNIST/CIFAR variant only.
    if Path(c.__file__).resolve().parents[2].name == 'parnv-MC' and crown == 'UNKNOWN':
        pgd = [initial_candidate] + list(pgd)
        events = 'P' + events
    state = make_state(count)
    original, current = object(), object()
    networks = iter([original, current])
    prop = dict(type='basic', input=[(0, {'Lower': 0, 'Upper': 1})], output=[(0, {'Lower': 0})])
    monkeypatch.setattr(c, 'network_from_nnet_file', lambda *a: next(networks))
    from core.utils import marabou_query_utils as q
    monkeypatch.setattr(q, 'reduce_property_to_basic_form', lambda network, test_property: (network, test_property))
    monkeypatch.setattr(c, 'apply_input_bounds_from_property', lambda *a: None)
    bounds = SimpleNamespace(lower_bounds=[-1], upper_bounds=[1])
    monkeypatch.setattr(c, 'compute_crown_bounds', lambda *a: (None, bounds))
    monkeypatch.setattr(c, 'classify_crown_output_bounds', lambda **kw: (crown, 'scripted'))
    monkeypatch.setattr(c, 'prune_dead_relu_neurons', lambda **kw: {'total_pruned': 0})
    monkeypatch.setattr(c, '_state_from_network', lambda *a: state)
    monkeypatch.setattr(c, 'build_network_from_state', lambda *a: current)
    monkeypatch.setattr(c, 'merge_last_two_hidden_layers', lambda *a, **kw: [])
    monkeypatch.setattr(c, 'write_nnet', lambda *a: None)
    monkeypatch.setattr(c, '_write_stage_logs', lambda *a, **kw: None)
    monkeypatch.setattr(c, '_network_structure_for_log', lambda *a: 'mock', raising=False)
    monkeypatch.setattr(c, 'preprocess_last_two_hidden_layers_inc_dec', lambda *a: [])
    monkeypatch.setattr(c, '_random_equivalence_test', lambda **kw: {'passed': True})
    monkeypatch.setattr(c, '_counterexample_violates_network',
        lambda network, counterexample, **kw: (network is original and counterexample[0] == .9, [0.]))
    monkeypatch.setattr(c, 'is_counterexample_on_current_network', lambda **kw: False)
    actual, formal_values, pgd_values = [], iter(formal), iter(pgd)

    def verify(**kw):
        actual.append('F')
        value = next(formal_values)
        return dict(status=value if isinstance(value, str) else 'sat', counterexample={0: value},
                    marabou_runtime=0., planet_runtime=0., raw_solver_output='scripted')

    def search(*a, **kw):
        actual.append('P')
        value = next(pgd_values)
        return dict(found=value is not None, counterexample={0: value} if value is not None else {},
                    time_seconds=.01, reason='scripted')

    verifier = 'verify_network_with_planet' if hasattr(c, 'cegar_verify_with_planet') else 'verify_network_with_marabou'
    monkeypatch.setattr(c, verifier, verify)
    monkeypatch.setattr(c, 'search_counterexample', search)
    result = c.cegar_verify_with_marabou('mock.nnet', prop, tmp_path, max_refinement_steps=budget)
    assert result['result'] == wanted
    assert ''.join(actual) == events
    assert result['cegar_iterations'] == result['formal_verifier_calls'] == events.count('F')
    assert result['pgd_calls'] == events.count('P')
    assert result['marabou_calls'] == (0 if verifier.endswith('planet') else events.count('F'))
    assert result['total_refinement_steps'] == refinements
    assert len(state.refinement_log) == refinements


def test_initial_pgd_genuine_skips_formal(monkeypatch, tmp_path):
    if Path(c.__file__).resolve().parents[2].name != 'parnv-MC':
        pytest.skip('Initial PGD was changed only in the MC variant')
    test_controller(monkeypatch, tmp_path, 'UNKNOWN', [], [], 8, None,
                    'UNSAFE', '', 0, initial_candidate=.9)


@pytest.mark.parametrize('size', [784, 3072])
@pytest.mark.parametrize('operator', ['ge', 'le'])
def test_real_pgd_box_and_property(size, operator):
    weight = np.zeros((1, size))
    weight[0, 0] = 1
    net = build([weight], [np.zeros(1)])
    prop = dict(type='adversarial', input=[(i, {'Lower': .4, 'Upper': .6}) for i in range(size)],
                output=[(0, {'Lower': .59, 'Upper': .41})], _adversarial_violation_operator=operator)
    result = search_counterexample(net, prop, PGDConfig(steps=12, restarts=1, step_size=.5))
    assert result['found']
    x = result['counterexample']
    assert len(x) == size and all(.4 <= v <= .6 for v in x.values())
    assert x[0] >= .59 if operator == 'ge' else x[0] <= .41
    assert c.is_counterexample_on_current_network(net, x, prop)


def test_pgd_miss_is_not_proof_and_objective():
    net = build([np.ones((1, 1))], [np.zeros(1)])
    prop = dict(type='basic', input=[(0, {'Lower': 0, 'Upper': 1})], output=[(0, {'Lower': 2})])
    assert not search_counterexample(net, prop, PGDConfig(steps=3))['found']
    y = torch.tensor([[.3, -.2]], dtype=torch.float64)
    adv = dict(type='adversarial', output=[(0, {'Lower': .1}), (1, {'Lower': 0})])
    assert violation_objective(y, adv).item() == pytest.approx(.2)
    adv['type'] = 'basic'
    assert violation_objective(y, adv).item() == pytest.approx(-.2)


def test_invalid_candidates_and_budgets():
    net = build([np.ones((1, 1))], [np.zeros(1)])
    prop = dict(type='basic', input=[(0, {'Lower': 0, 'Upper': 1})], output=[(0, {'Lower': 0})])
    for x in [{}, {0: float('nan')}, {0: 1.01}]:
        with pytest.raises(ValueError):
            c.is_counterexample_on_current_network(net, x, prop)
    with pytest.raises(ValueError):
        PGDConfig(restarts=0)
    with pytest.raises(ValueError):
        PGDConfig(step_size=float('nan'))
