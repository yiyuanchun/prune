"""Progressive scheduling plus real LP/split/undo and formal-verifier regressions."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.cegar import raw_cegar as c
from core.cegar import progressive_merge as p
from core.nnet.read_nnet import _build_network_from_weights_and_biases as build


def network_and_property(widths, lower=0.):
    sizes = [2, *widths, 1]
    weights = [np.ones((b, a)) * .125 for a, b in zip(sizes, sizes[1:])]
    net = build(weights, [np.zeros(b) for b in sizes[1:]])
    prop = dict(type='basic', input=[(i, {'Lower': lower, 'Upper': 1.}) for i in range(2)],
                output=[(0, {'Lower': 100.})])
    return net, prop


def interval_bounds(network):
    """Certified interval bounds, to isolate scheduling from CROWN performance."""
    lo = np.array([n.lower_bound for n in network.layers[0].nodes])
    hi = np.array([n.upper_bound for n in network.layers[0].nodes])
    pre_l, pre_u, post_l, post_u = [], [], [], []
    for w, b in zip(network.generate_weights()[:-1], network.generate_biases()[:-1]):
        w, b = np.asarray(w), np.asarray(b)
        low = np.maximum(w, 0) @ lo + np.minimum(w, 0) @ hi + b
        high = np.maximum(w, 0) @ hi + np.minimum(w, 0) @ lo + b
        pre_l.append(low.tolist()); pre_u.append(high.tolist())
        lo, hi = np.maximum(low, 0), np.maximum(high, 0)
        post_l.append(lo.tolist()); post_u.append(hi.tolist())
    return SimpleNamespace(preactivation_lower_bounds_by_hidden_layer=pre_l,
        preactivation_upper_bounds_by_hidden_layer=pre_u,
        postactivation_lower_bounds_by_hidden_layer=post_l,
        postactivation_upper_bounds_by_hidden_layer=post_u)


def schedule(monkeypatch, widths, violations):
    net, prop = network_and_property(widths)
    baseline = c._state_from_network(net, prop)
    pristine = c._to_jsonable(baseline.__dict__)
    monkeypatch.setattr(c, 'compute_crown_hidden_layer_bounds', interval_bounds)
    answers = iter(violations)
    monkeypatch.setattr(c, '_random_property_violation_test',
                        lambda *a, **kw: dict(has_violation=next(answers)))
    visits, clones = [], []
    classify, clone = c.classify_or_split_layer, p.clone_baseline_state

    def classify_spy(state, layer, **kw):
        visits.append(layer)
        return classify(state, layer, **kw)

    def clone_spy(source):
        assert source is baseline and c._to_jsonable(source.__dict__) == pristine
        result = clone(source)
        assert not result.merge_log_stack and not result.refinement_log and not result.mapping['splits']
        clones.append(result)
        return result

    monkeypatch.setattr(c, 'classify_or_split_layer', classify_spy)
    monkeypatch.setattr(p, 'clone_baseline_state', clone_spy)
    result = p.run_progressive_equivalence_preprocessing(baseline, prop)
    assert c._to_jsonable(baseline.__dict__) == pristine
    assert len({id(s) for s in clones}) == len(clones)
    return result, baseline, visits


def test_single_layer_and_odd_final_merge(monkeypatch):
    result, _, visits = schedule(monkeypatch, [6], [False, False])
    layer = result.layer_reports[0]
    assert visits == [1] and result.statistics['progressive_rounds'] == 1
    assert layer['merge_count'] == 3 and layer['target_size'] == layer['final_layer_size'] == 3
    assert [t['after_merge_count'] for t in layer['random_tests']] == [2, 3]
    assert result.statistics['progressive_stop_reason'] == 'all_layers_processed'


def test_violation_after_two_merges_stops_before_target_and_front_layers(monkeypatch):
    result, baseline, visits = schedule(monkeypatch, [10, 10, 10], [True])
    assert visits == [3] and len(result.rounds) == 1
    assert result.layer_reports[0]['merge_count'] == 2
    assert result.layer_reports[0]['final_layer_size'] == 8 > result.layer_reports[0]['target_size']
    assert result.statistics['progressive_stop_reason'] == 'random_violation'
    for layer in [1, 2]:
        assert result.state.ids_by_layer[layer] == baseline.ids_by_layer[layer]
        assert result.state.labels[layer] == ['unknown'] * 10
    assert not any(r['layer'] in [1, 2] for r in result.state.mapping['splits'])


def test_confirmed_replay_no_random_tests_and_second_layer_stop(monkeypatch):
    result, _, visits = schedule(monkeypatch, [8, 8, 8], [False, False, True])
    assert visits == [3, 3, 2] and len(result.rounds) == 2
    replay, exploration = result.layer_reports
    assert replay['role'] == 'confirmed_replay' and replay['merge_count'] == 4
    assert replay['random_test_count'] == 0
    assert exploration['role'] == 'exploration' and exploration['merge_count'] == 2
    assert result.statistics['progressive_confirmed_layers'] == 1
    assert result.statistics['progressive_stop_layer'] == 2
    # Round 1 history is discarded; only Round 2 records enter refinement.
    assert len(result.state.merge_log_stack) == 6
    assert [r['merge_id'] for r in result.state.merge_log_stack] == list(range(6))
    assert all(r['round_index'] == 2 for r in result.state.merge_log)


def test_all_layers_clone_baseline_and_profile(monkeypatch):
    result, _, visits = schedule(monkeypatch, [4, 4, 4], [False] * 3)
    assert visits == [3, 3, 2, 3, 2, 1]
    assert result.statistics['progressive_confirmed_layers'] == 3
    assert result.statistics['progressive_rounds'] == 3
    assert [len(s) for s in result.state.ids_by_layer] == [2, 2, 2, 2, 1]
    for name in p.TIMING_FIELDS:
        assert result.statistics[name] >= 0
    total = result.statistics['progressive_preprocessing_time_seconds']
    assert sum(result.statistics[k] for k in ['merge_total_time_seconds',
        'inc_dec_preprocessing_time_seconds', 'progressive_round_rebuild_time_seconds']) <= total
    assert result.statistics['merge_lp_time_seconds'] > 0


@pytest.mark.parametrize('widths,rounds', [([], 0), ([1], 1)])
def test_no_merge_needed_never_classifies(monkeypatch, widths, rounds):
    result, _, visits = schedule(monkeypatch, widths, [])
    assert visits == [] and result.statistics['progressive_rounds'] == rounds
    assert not result.state.merge_log_stack


def test_no_same_label_pair_stops_expansion(monkeypatch):
    net, prop = network_and_property([2, 2])
    net = build([np.ones((2, 2)), np.ones((2, 2)), np.array([[1., -1.]])],
                [np.zeros(2), np.zeros(2), np.zeros(1)])
    baseline = c._state_from_network(net, prop)
    monkeypatch.setattr(c, 'compute_crown_hidden_layer_bounds', interval_bounds)
    monkeypatch.setattr(c, '_random_property_violation_test', lambda *a: pytest.fail('No merge, no test'))
    result = p.run_progressive_equivalence_preprocessing(baseline, prop)
    assert result.statistics['progressive_stop_reason'] == 'no_merge_pair'
    assert len(result.rounds) == 1 and result.state.labels[1] == ['unknown', 'unknown']
    assert result.state.labels[2] == ['inc', 'dec'] and not result.state.merge_log_stack


def test_cross_layer_undo_restores_intervening_split_dimensions(monkeypatch):
    _, prop = network_and_property([4, 4])
    net = build([np.ones((4, 2)), np.ones((4, 4)), np.array([[1.2, 1., -1., -.8]])],
                [np.zeros(4), np.zeros(4), np.zeros(1)])
    baseline = c._state_from_network(net, prop)
    monkeypatch.setattr(c, 'compute_crown_hidden_layer_bounds', interval_bounds)
    monkeypatch.setattr(c, '_random_property_violation_test', lambda *a: dict(has_violation=False))
    result = p.run_progressive_equivalence_preprocessing(baseline, prop)
    assert result.layer_reports[1]['size_after_inc_dec_split'] == 8
    state, records = result.state, []
    points = [{0: x, 1: y} for x in [0, .4, 1.] for y in [0, .7, 1.]]
    while state.merge_log_stack:
        current = c.build_network_from_state(state)
        for point in points:
            assert np.all(current.speedy_evaluate(point) >= net.speedy_evaluate(point) - 1e-8)
        records.append(c.undo_last_merge(None, state.merge_log_stack, state.labels, state.mapping, state))
    final = c.build_network_from_state(state)
    assert any(r['undone_inc_dec_layers'] == [1] for r in records)
    assert state.ids_by_layer[1] == baseline.ids_by_layer[1]
    assert state.mapping['current_ids_by_layer'] == state.ids_by_layer
    for point in points:
        np.testing.assert_allclose(final.speedy_evaluate(point), net.speedy_evaluate(point), atol=1e-10)


def test_random_violation_only_stops_merge_then_real_marabou_verifies(monkeypatch, tmp_path):
    net, prop = network_and_property([8])
    source = tmp_path / 'network.nnet'
    c.write_nnet(net, source)
    monkeypatch.setattr(c, 'compute_crown_bounds', lambda network: (interval_bounds(network),
                        SimpleNamespace(lower_bounds=[0.], upper_bounds=[1.])))
    monkeypatch.setattr(c, 'classify_crown_output_bounds', lambda **kw: ('UNKNOWN', 'exercise CEGAR'))
    monkeypatch.setattr(c, 'prune_dead_relu_neurons', lambda **kw: {'total_pruned': 0})
    monkeypatch.setattr(c, 'compute_crown_hidden_layer_bounds', interval_bounds)
    monkeypatch.setattr(c, '_random_property_violation_test', lambda *a: dict(has_violation=True))
    result = c.cegar_verify_with_marabou(source, prop, tmp_path / 'out', marabou_timeout=30)
    assert result['result'] == 'VERIFIED' and result['marabou_calls'] == 1
    assert result['progressive_stop_reason'] == 'random_violation'
    assert result['progressive_rounds'] == 1


def test_signed_input_fallback_fails_before_mutating_network(monkeypatch):
    net, prop = network_and_property([4], lower=-1.)
    state = c._state_from_network(net, prop)
    c.classify_or_split_layer(state, 1)
    pristine = c._to_jsonable(state.__dict__)
    monkeypatch.setattr(c, 'solve_merged_incoming_weights_and_bias', lambda **kw: {'used_fallback': True})
    bounds = interval_bounds(c.build_network_from_state(state))
    pair = c.select_merge_pair_by_crown_score(state, 1, bounds)
    with pytest.raises(ValueError, match='signed inputs'):
        c.merge_two_neurons(state, pair, bounds)
    assert c._to_jsonable(state.__dict__) == pristine


@pytest.mark.parametrize('widths', [[], [4]])
@pytest.mark.parametrize('threshold,expected', [(100., 'unsat'), (.015625, 'sat')])
def test_real_marabou_shallow_network_output_variables(widths, threshold, expected):
    net, prop = network_and_property(widths)
    prop['output'] = [(0, {'Lower': threshold})]
    c.apply_input_bounds_from_property(net, prop)
    result = c.verify_network_with_marabou(net, prop, timeout=30)
    assert result['status'] == expected, result['raw_solver_output']
    if expected == 'sat':
        candidate = c._counterexample_input(result['counterexample'], input_size=2)
        assert c.is_counterexample_on_current_network(net, candidate, prop)
