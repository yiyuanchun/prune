"""Progressively explore hidden layers; random testing only stops abstraction."""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any

from core.cegar import raw_cegar as c

TIMING_FIELDS = (
    'progressive_preprocessing_time_seconds', 'inc_dec_preprocessing_time_seconds',
    'merge_total_time_seconds', 'merge_crown_time_seconds',
    'merge_pair_selection_time_seconds', 'merge_lp_time_seconds',
    'merge_random_test_time_seconds', 'progressive_round_rebuild_time_seconds',
    'network_build_time_seconds',
)
STAT_FIELDS = ('progressive_rounds', 'progressive_confirmed_layers',
               'progressive_stop_layer', 'progressive_stop_reason') + TIMING_FIELDS


def empty_statistics():
    return dict(progressive_rounds=0, progressive_confirmed_layers=0,
                progressive_stop_layer=None, progressive_stop_reason='not_started',
                **{key: 0.0 for key in TIMING_FIELDS})


@dataclass
class ProgressiveResult:
    state: c.RawCegarState
    statistics: dict[str, Any]
    rounds: list[dict[str, Any]]

    @property
    def layer_reports(self):
        return self.rounds[-1]['layers'] if self.rounds else []


def _timed(profile, key, fn, *args, **kwargs):
    start = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        profile[key] += time.perf_counter() - start


def clone_baseline_state(baseline_state):
    """Never share mutable arrays, IDs, labels or undo history between rounds."""
    return copy.deepcopy(baseline_state)


def prepare_layer_for_merge(state, layer_index, target_size, profile, tolerance):
    if len(state.ids_by_layer[layer_index]) <= target_size:
        return None, 'target_reached'
    if layer_index < len(state.weights) - 1 and not c._has_inc_dec_labels(state.labels.get(layer_index + 1)):
        return None, 'downstream_labels_unavailable'
    # Splitting an upstream layer changes the column count of downstream Merge
    # snapshots. Remember this equivalent operation at its exact undo depth.
    def prepare():
        checkpoint = dict(
            layer=layer_index, merge_depth=len(state.merge_log_stack),
            incoming=state.weights[layer_index - 1].copy(),
            bias=state.biases[layer_index - 1].copy(),
            outgoing=state.weights[layer_index].copy(),
            ids=list(state.ids_by_layer[layer_index]),
            labels=copy.deepcopy(state.labels.get(layer_index)),
        )
        result = c.classify_or_split_layer(state, layer_index, tolerance=tolerance)
        state.preprocessing_undo_stack.append(checkpoint)
        return result
    return _timed(profile, 'inc_dec_preprocessing_time_seconds', prepare), None


def _merge_layer(state, layer_index, reference_size, test_property, profile,
                 round_index, exploration_layer, confirmed_layer_count, tolerance, explore):
    target = max(int(math.ceil(0.5 * reference_size)), 1)
    report = dict(layer=layer_index, layer_index=layer_index, round_index=round_index,
                  exploration_layer=exploration_layer, confirmed_layer_count=confirmed_layer_count,
                  role='exploration' if explore else 'confirmed_replay',
                  size_before_preprocessing=len(state.ids_by_layer[layer_index]),
                  initial_size_before_inc_dec=reference_size, target_size=target,
                  merge_count=0, random_test_count=0, random_violation_found=False,
                  random_tests=[], merge_ids=[])
    preprocessing, skip_reason = prepare_layer_for_merge(state, layer_index, target, profile, tolerance)
    report['size_after_inc_dec_split'] = len(state.ids_by_layer[layer_index])
    if preprocessing is not None:
        report['inc_dec_preprocess'] = preprocessing
    if skip_reason:
        report.update(reason='target_reached' if skip_reason == 'target_reached' else 'no_merge_pair',
                      detail=skip_reason, final_layer_size=len(state.ids_by_layer[layer_index]))
        return report
    random_inputs = c._random_inputs_from_state(state, samples=10, seed=layer_index) if explore else []
    merge_start = time.perf_counter()
    try:
        while len(state.ids_by_layer[layer_index]) > target:
            network = _timed(profile, 'network_build_time_seconds', c.build_network_from_state, state)
            bounds = _timed(profile, 'merge_crown_time_seconds', c.compute_crown_hidden_layer_bounds, network)
            pair = _timed(profile, 'merge_pair_selection_time_seconds',
                          c.select_merge_pair_by_crown_score, state, layer_index, bounds)
            if pair is None:
                report['reason'] = 'no_merge_pair'
                break
            record = c.merge_two_neurons(state, pair, bounds, profiling=profile)
            record.update(round_index=round_index, progressive_role=report['role'])
            report['merge_count'] += 1
            report['merge_ids'].append(record['merge_id'])
            at_target = len(state.ids_by_layer[layer_index]) <= target
            if explore and (report['merge_count'] % 2 == 0 or at_target):
                network = _timed(profile, 'network_build_time_seconds', c.build_network_from_state, state)
                test = _timed(profile, 'merge_random_test_time_seconds',
                              c._random_property_violation_test, network, test_property, random_inputs)
                report['random_test_count'] += 1
                report['random_tests'].append(dict(after_merge_count=report['merge_count'], **test))
                record['random_property_test_after_merge'] = test
                if test['has_violation']:
                    report['random_violation_found'] = True
                    report['reason'] = 'random_violation'
                    break
        else:
            report['reason'] = 'target_reached'
    finally:
        profile['merge_total_time_seconds'] += time.perf_counter() - merge_start
    report['final_layer_size'] = len(state.ids_by_layer[layer_index])
    return report


def merge_confirmed_layer(*args, **kwargs):
    return _merge_layer(*args, **kwargs, explore=False)


def explore_layer_with_random_tests(*args, **kwargs):
    return _merge_layer(*args, **kwargs, explore=True)


def run_progressive_equivalence_preprocessing(baseline_state, test_property,
                                              tolerance=1e-12, max_hidden_layers=None):
    """Return one coherent round, never a proof or a mixture of prior states.

    'Confirmed' only means that sampling did not stop the previous exploration;
    it is NOT a formal safety certificate. LPs and CROWN are recomputed on replay.
    """
    if test_property is None:
        raise ValueError('Progressive exploration requires a property')
    if (baseline_state.merge_log_stack or baseline_state.merge_log or baseline_state.refinement_log
            or baseline_state.preprocessing_undo_stack or baseline_state.mapping.get('splits')):
        raise ValueError('Progressive preprocessing requires a clean less_neuron baseline')
    if max_hidden_layers is not None and max_hidden_layers < 0:
        raise ValueError('max_hidden_layers must be non-negative')
    start = time.perf_counter()
    profile = empty_statistics()
    order = list(reversed(range(1, len(baseline_state.weights))))
    if max_hidden_layers is not None:
        order = order[:max_hidden_layers]
    confirmed, rounds = [], []
    state = None

    def finish(reason, layer):
        profile.update(progressive_rounds=len(rounds), progressive_confirmed_layers=len(confirmed),
                       progressive_stop_layer=layer, progressive_stop_reason=reason,
                       progressive_preprocessing_time_seconds=time.perf_counter() - start)
        return ProgressiveResult(state, profile, rounds)

    if not order:
        state = _timed(profile, 'progressive_round_rebuild_time_seconds', clone_baseline_state, baseline_state)
        return finish('all_layers_processed', None)
    for depth, exploration_layer in enumerate(order, 1):
        state = _timed(profile, 'progressive_round_rebuild_time_seconds', clone_baseline_state, baseline_state)
        round_report = dict(round_index=depth, exploration_layer=exploration_layer,
                            confirmed_layer_count=len(confirmed), layers=[])
        rounds.append(round_report)
        c._print_to_console('[CEGAR][Progressive] round={} exploration_layer={} replay={}'.format(
            depth, exploration_layer, confirmed))
        for layer_index in [*confirmed, exploration_layer]:
            fn = explore_layer_with_random_tests if layer_index == exploration_layer else merge_confirmed_layer
            layer_report = fn(state, layer_index, len(baseline_state.ids_by_layer[layer_index]),
                              test_property, profile, depth, exploration_layer, len(confirmed), tolerance)
            round_report['layers'].append(layer_report)
            c._print_to_console('[CEGAR][Progressive] layer={} role={} merges={} tests={} size={} stop={}'.format(
                layer_index, layer_report['role'], layer_report['merge_count'], layer_report['random_test_count'],
                layer_report['final_layer_size'], layer_report['reason']))
            if layer_report['reason'] != 'target_reached':
                return finish(layer_report['reason'], layer_index)
        confirmed.append(exploration_layer)
        round_report['confirmed_layer_count_after'] = len(confirmed)
    return finish('all_layers_processed', order[-1])
