"""Paired MNIST experiment using the current shared CEGAR implementation.

The observers below only measure calls and preserve tracebacks. In the REDNet
route the 10-logit concrete counterexample check explicitly uses the original
classifier; the existing 9-margin abstract/refinement checks remain unchanged.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch

from batch_verify_mnist_cifar10_inputs import _classifies_sample_correctly
from core.cegar import raw_cegar as cegar
from core.cegar.pgd_search import add_search_arguments, search_options
from core.cegar.progressive_merge import STAT_FIELDS as PROGRESSIVE_FIELDS
from core.nnet.read_nnet import network_from_nnet_file, _load_onnx_dense_layers, read_nnet_parameters
from core.pre_process import rednet_pipeline
from core.pre_process.dead_relu_pruning import apply_input_bounds_from_property
from core.utils import marabou_query_utils as queries
from core.utils.experiment_output import (
    artifacts_enabled, diagnostic_artifacts, sample_workspace, solver_console,
)
from core.utils.mnist_property_utils import (
    load_mnist_sample, build_mnist_adversarial_property, build_mnist_property_id,
    _resolve_idx_path,
)

ROOT = Path(__file__).resolve().parents[1]
COMMON = [
    'sample_index', 'property_id', 'epsilon', 'verification_result', 'query_result',
    'total_verification_time_seconds', 'crown_time_seconds', 'original_relu_count',
    'cegar_iterations', 'refinement_steps', 'cegar_time_seconds',
    'crown_initial_time_seconds', 'cegar_crown_time_seconds',
    'crown_precheck_status', 'marabou_calls', 'error_type', 'error_message',
    'pgd_calls', 'pgd_time_seconds', 'pgd_candidates_found',
    'pgd_genuine_counterexamples', 'pgd_spurious_counterexamples',
    'batch_refinement_rounds', 'extra_refinement_steps', 'formal_verifier_calls',
]
REDNET = [
    'rednet_reduction_time_seconds', 'rednet_preprocessing_time_seconds',
    'rednet_crown_time_seconds', 'stable_inactive_relu_count',
    'stable_inactive_removed_count', 'stable_active_relu_count',
    'stable_active_eliminated_count', 'reconstructed_stable_relu_count',
    'stable_active_net_reduction_count', 'relu_count_after_rednet',
    'total_relu_reduction_count', 'relu_reduction_ratio',
    'equivalence_sample_count', 'equivalence_output_abs_error_mean',
    'equivalence_output_abs_error_variance', 'equivalence_output_abs_error_max',
    'equivalence_passed', 'cegar_additional_dead_relu_removed_count',
]
BASELINE = ['crown_stable_inactive_relu_count', 'crown_removed_relu_count', 'relu_count_after_crown_pruning']


def dump(path, value):
    if not artifacts_enabled():
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def property_for(manifest, index):
    sample, label = load_mnist_sample(manifest['dataset_root'], manifest['dataset_split'], index)
    prop = build_mnist_adversarial_property(sample, label, manifest['epsilon'], manifest['output_threshold'])
    pid = build_mnist_property_id(manifest['dataset_split'], index, manifest['epsilon'])
    return sample, label, prop, pid


def forward_batch(weights, biases, inputs):
    values = np.asarray(inputs, dtype=np.float64)
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        values = values @ np.asarray(weight, dtype=np.float64).T + np.asarray(bias, dtype=np.float64)
        if i < len(weights) - 1:
            values = np.maximum(values, 0)
    return values


def prepare(args):
    import onnxruntime as ort
    if args.samples.exists():
        raise FileExistsError('Refusing to replace a frozen sample manifest: ' + str(args.samples))
    model = args.model.resolve()
    # Never assume a neighboring .nnet corresponds to the requested ONNX model.
    # Export a separate shared input using the project's existing converter.
    from convert_onnx_to_nnet import convert_onnx_to_nnet
    nnet = ROOT / 'results/b3_audit/mnist_fc_relu_64x3_from_onnx.nnet'
    convert_onnx_to_nnet(model, nnet)
    source_weights, source_biases = _load_onnx_dense_layers(str(model))
    exported = read_nnet_parameters(str(nnet))
    if exported['layerSizes'] != [784, 64, 64, 64, 10]:
        raise ValueError('Unexpected MNIST model architecture')
    for key in ['inputMeans', 'outputMean']:
        if np.any(np.asarray(exported[key]) != 0):
            raise ValueError('Non-identity NNet normalization')
    for key in ['inputRanges', 'outputRange']:
        if np.any(np.asarray(exported[key]) != 1):
            raise ValueError('Non-identity NNet normalization')
    from convert_onnx_to_nnet import _validate_export
    _validate_export(source_weights, source_biases, nnet)
    network = network_from_nnet_file(str(nnet))
    manifest = dict(dataset='mnist', dataset_split='train', epsilon=0.02,
                    output_threshold=2.220446049250313e-16, dataset_root=str(args.dataset_root.resolve()),
                    model=str(model), nnet=str(nnet), seed=20260912,
                    marabou_timeout_seconds=3600, max_refinement_steps=None, tolerance=1e-7,
                    property_timeout_seconds=None, equivalence_samples=128, equivalence_tolerance=1e-8,
                    crown_device='cpu', torch_threads=1, sample_indices=[], skipped_indices=[],
                    baseline_commit='9706f4c24522422540205d6f631f6b2764aefa29',
                    model_sha256=digest(model), nnet_sha256=digest(nnet))
    centers = []
    index = 0
    while len(manifest['sample_indices']) < 30:
        sample, label, prop, pid = property_for(manifest, index)
        metadata = {'sample': sample.tolist(), 'label': label, 'dataset': 'mnist'}
        if _classifies_sample_correctly(network, metadata):
            manifest['sample_indices'].append(index)
            centers.append(sample)
            dump(args.samples.parent / 'b3_properties' / (pid + '.json'), prop)
        else:
            manifest['skipped_indices'].append(index)
        index += 1
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=opts, providers=['CPUExecutionProvider'])
    shape = session.get_inputs()[0].shape
    tail = tuple(int(v) for v in shape[1:])
    errors, conversion_errors = [], []
    rng = np.random.default_rng(manifest['seed'])
    inputs = list(centers) + list(rng.uniform(0, 1, (128, 784)).astype(np.float32))
    for x in inputs:
        original = session.run(None, {session.get_inputs()[0].name: x.reshape((1,) + tail)})[0].reshape(-1)
        parsed = forward_batch(source_weights, source_biases, x)
        converted = network.speedy_evaluate(dict(enumerate(map(float, x))))
        errors.append(float(np.max(np.abs(original - converted))))
        conversion_errors.append(float(np.max(np.abs(parsed - converted))))
    if max(errors) > 1e-4 or max(conversion_errors) > 1e-8:
        raise ValueError('ONNX/NNet output validation failed: ' + str((max(errors), max(conversion_errors))))
    manifest['conversion_check'] = dict(samples=len(inputs), onnx_float32_vs_nnet_float64_max_error=max(errors),
                                        parsed_onnx_float64_vs_nnet_max_error=max(conversion_errors))
    manifest['dataset_sha256'] = {name: digest(_resolve_idx_path(manifest['dataset_root'], name))
                                  for name in ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte']}
    dump(args.samples, manifest)
    print(json.dumps(manifest), flush=True)


@contextlib.contextmanager
def observe(metrics, concrete_network=None):
    """Measure the main pipeline without altering its branching logic."""
    original_crown = cegar.compute_crown_bounds
    original_hidden = cegar.compute_crown_hidden_layer_bounds
    original_prune = cegar.prune_dead_relu_neurons
    original_query = queries.get_query
    original_check = cegar._counterexample_violates_network

    def initial_crown(network):
        start = time.perf_counter()
        before_hidden = metrics['all_hidden_crown_seconds']
        result = original_crown(network)
        metrics['initial_hidden_seconds'] = metrics['all_hidden_crown_seconds'] - before_hidden
        metrics['crown_initial_time_seconds'] = time.perf_counter() - start
        bounds = result[0]
        metrics['crown_stable_inactive_relu_count'] = sum(v < 0 for group in bounds.preactivation_upper_bounds_by_hidden_layer for v in group)
        return result

    def hidden_crown(network):
        start = time.perf_counter()
        try:
            return original_hidden(network)
        finally:
            metrics['all_hidden_crown_seconds'] += time.perf_counter() - start

    def prune(*args, **kwargs):
        result = original_prune(*args, **kwargs)
        metrics['crown_removed_relu_count'] += result['total_pruned']
        return result

    def query(*args, **kwargs):
        metrics['marabou_calls'] += 1
        try:
            return original_query(*args, **kwargs)
        except Exception:
            # main catches this exception; print it before it loses its traceback.
            traceback.print_exc()
            raise

    def check(network, counterexample, property_spec, tolerance=1e-7):
        if len(network.layers[-1].nodes) == 10:
            if set(counterexample) != set(range(len(network.layers[0].nodes))):
                raise ValueError('Concrete counterexample input is incomplete')
            for i, bound in property_spec['input']:
                value = counterexample[int(i)]
                if not np.isfinite(value) or not bound['Lower'] <= value <= bound['Upper']:
                    raise ValueError('Concrete counterexample lies outside the property box')
        if concrete_network is not None and len(network.layers[-1].nodes) == 10:
            network = concrete_network
            metrics['original_concrete_checks'] += 1
        return original_check(network, counterexample, property_spec, tolerance)

    with contextlib.ExitStack() as stack:
        for module, name, fn in [(cegar, 'compute_crown_bounds', initial_crown),
                                 (cegar, 'compute_crown_hidden_layer_bounds', hidden_crown),
                                 (cegar, 'prune_dead_relu_neurons', prune),
                                 (queries, 'get_query', query),
                                 (cegar, '_counterexample_violates_network', check)]:
            stack.enter_context(patch.object(module, name, fn))
        yield


def run(args):
    with diagnostic_artifacts(getattr(args, 'save_artifacts', False)):
        return _run(args)


def _run(args):
    manifest = json.loads(args.samples.read_text())
    for key in ['model', 'nnet']:
        if digest(manifest[key]) != manifest[key + '_sha256']:
            raise ValueError('Frozen model changed: ' + key)
    for name, expected in manifest['dataset_sha256'].items():
        if digest(_resolve_idx_path(manifest['dataset_root'], name)) != expected:
            raise ValueError('Frozen dataset changed: ' + name)
    expected_count = int(manifest.get('sample_count', 30))
    if expected_count <= 0 or len(manifest['sample_indices']) != expected_count or len(set(manifest['sample_indices'])) != expected_count:
        raise ValueError(f'Manifest must contain exactly {expected_count} distinct valid samples')
    epsilon = float(manifest['epsilon'])
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Epsilon must be finite and positive')
    selected = manifest['sample_indices'][:args.limit] if args.limit else manifest['sample_indices']
    enabled = args.route == 'rednet'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    epsilon_tag = format(epsilon, 'g').replace('.', '')
    csv_path = output / ('b3_{}_eps{}.csv'.format('rednet_narv' if enabled else 'baseline_par', epsilon_tag))
    concrete = network_from_nnet_file(manifest['nnet'])
    fields = COMMON + list(PROGRESSIVE_FIELDS) + (REDNET if enabled else BASELINE)
    dump(output / 'run_config.json', {**manifest, 'route': args.route, 'pid': os.getpid(),
         'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
         'sample_manifest_sha256': digest(args.samples),
         'selected_indices': selected, 'smoke_direct_solver': args.smoke_direct_solver,
         'cegar_policy': 'progressive-merge-initial-pgd-v3',
         'extra_refinement_merges': args.extra_refinement_merges,
         'pgd': search_options(args)['pgd_config'].to_dict(),
         'cegar_source_sha256': digest(Path(cegar.__file__)),
         'timing_definition': 'total: entry to final CEGAR return, including REDNet I/O and equivalence; excludes shared dataset/property preparation and supplementary smoke solver. CEGAR time includes its own CROWN calls; crown_time includes all CROWN calls and overlaps CEGAR time.'})
    print(f'B3 START route={args.route} pid={os.getpid()} eps={epsilon} split={manifest["dataset_split"]} indices={selected} CROWN=cpu', flush=True)
    with csv_path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        for index in selected:
            sample, label, prop, pid = property_for(manifest, index)
            if not _classifies_sample_correctly(concrete, {'sample': sample.tolist(), 'label': label}):
                raise ValueError('Frozen valid sample no longer classifies correctly')
            with sample_workspace(output / pid) as run_dir:
                dump(output / 'progress.json', {'sample_index': index, 'property_id': pid, 'stage': 'started', 'pid': os.getpid()})
                print(f'B3 SAMPLE START index={index} property={pid}', flush=True)
                row = dict.fromkeys(fields, '')
                row.update(sample_index=index, property_id=pid, epsilon=epsilon, original_relu_count=192)
                metrics = dict(all_hidden_crown_seconds=0., crown_initial_time_seconds=0.,
                               crown_removed_relu_count=0, marabou_calls=0, original_concrete_checks=0)
                start = time.perf_counter()
                cegar_start = None
                red = None
                failure = None
                try:
                    with solver_console():
                        working_file = manifest['nnet']
                        if enabled:
                            red = rednet_pipeline.prepare_rednet_nnet(
                                working_file, prop, run_dir / 'rednet.nnet',
                                equivalence_samples=manifest['equivalence_samples'],
                                equivalence_tolerance=manifest['equivalence_tolerance'],
                                seed=manifest['seed'] + index,
                            )
                            dump(run_dir / 'rednet_report.json', red)
                            working_file = red['output_nnet_file']
                            rr, eq = red['reduction'], red['equivalence']
                            row.update(rednet_reduction_time_seconds=red['reduction_time_seconds'],
                                rednet_preprocessing_time_seconds=red['preprocessing_time_seconds'],
                                rednet_crown_time_seconds=red['crown_time_seconds'],
                                stable_inactive_relu_count=sum(p['stable_inactive'] for p in rr['per_layer']),
                                stable_inactive_removed_count=rr['total_inactive_removed'],
                                stable_active_relu_count=sum(p['stable_active'] for p in rr['per_layer']),
                                stable_active_eliminated_count=rr['total_active_eliminated'],
                                reconstructed_stable_relu_count=rr['total_reconstructed_units'],
                                stable_active_net_reduction_count=rr['total_active_eliminated']-rr['total_reconstructed_units'],
                                relu_count_after_rednet=rr['reduced_relu_count'], total_relu_reduction_count=rr['relu_removed'],
                                relu_reduction_ratio=rr['relu_removed']/rr['original_relu_count'],
                                equivalence_sample_count=eq['samples'], equivalence_output_abs_error_mean=eq['mean_abs_error'],
                                equivalence_output_abs_error_variance=eq['variance_abs_error'],
                                equivalence_output_abs_error_max=eq['max_abs_error'], equivalence_passed=eq['passed'])
                            print(f"REDNET index={index} relus=192->{rr['reduced_relu_count']} inactive={rr['total_inactive_removed']} active={rr['total_active_eliminated']} reconstructed={rr['total_reconstructed_units']} max_error={eq['max_abs_error']:.3g}", flush=True)
                        cegar_start = time.perf_counter()
                        with observe(metrics, concrete if enabled else None):
                            result = cegar.cegar_verify_with_marabou(
                                working_file, prop, run_dir / 'cegar',
                                max_refinement_steps=manifest['max_refinement_steps'],
                                marabou_timeout=manifest['marabou_timeout_seconds'], tolerance=manifest['tolerance'], property_id=pid,
                                **search_options(args))
                        if result['cegar_iterations'] != metrics['marabou_calls']:
                            raise AssertionError('CEGAR iteration count differs from actual Marabou calls')
                        for name in ['pgd_calls', 'pgd_time_seconds', 'pgd_candidates_found',
                                     'pgd_genuine_counterexamples', 'pgd_spurious_counterexamples',
                                     'batch_refinement_rounds', 'extra_refinement_steps', 'formal_verifier_calls']:
                            row[name] = result[name]
                        row.update({name: result[name] for name in PROGRESSIVE_FIELDS})
                        row['cegar_time_seconds'] = time.perf_counter() - cegar_start
                        query_result = str(result.get('query_result', 'UNKNOWN')).upper()
                        row.update(verification_result={'SAT':'UNSAFE', 'UNSAT':'VERIFIED', 'TIMEOUT':'TIMEOUT', 'ERROR':'ERROR'}.get(query_result, 'UNKNOWN'),
                                   query_result=query_result, cegar_iterations=result.get('iterations', 0),
                                   refinement_steps=result.get('total_refinement_steps', 0), crown_precheck_status=result.get('crown_precheck_status', ''))
                        if query_result == 'ERROR':
                            raise RuntimeError('Marabou returned ERROR; see stderr traceback (use --save-artifacts for diagnostics)')
                except Exception as exc:
                    failure = exc
                    traceback.print_exc()
                    row.update(verification_result='ERROR', query_result='ERROR', error_type=type(exc).__name__, error_message=str(exc))
                    if cegar_start is not None:
                        row['cegar_time_seconds'] = time.perf_counter() - cegar_start
                row['total_verification_time_seconds'] = time.perf_counter() - start
                # initial_crown calls the wrapped hidden routine: subtract that nested
                # time once; output-bound time belongs to the initial stage as well.
                initial_time = metrics['crown_initial_time_seconds']
                # measured separately below by observe's first hidden call
                all_crown = metrics['all_hidden_crown_seconds'] + initial_time - metrics.get('initial_hidden_seconds', 0.)
                row.update(crown_initial_time_seconds=initial_time, cegar_crown_time_seconds=all_crown,
                           crown_time_seconds=all_crown + (red['crown_time_seconds'] if red else 0.),
                           marabou_calls=metrics['marabou_calls'])
                if enabled:
                    row['cegar_additional_dead_relu_removed_count'] = metrics['crown_removed_relu_count']
                else:
                    row.update(crown_stable_inactive_relu_count=metrics.get('crown_stable_inactive_relu_count', ''),
                               crown_removed_relu_count=metrics['crown_removed_relu_count'],
                               relu_count_after_crown_pruning=192-metrics['crown_removed_relu_count'])
                dump(run_dir / 'observations.json', metrics)
                dump(run_dir / 'row.json', row)
                writer.writerow(row)
                stream.flush()
                os.fsync(stream.fileno())
                print(f"B3 SAMPLE END index={index} result={row['verification_result']} total={row['total_verification_time_seconds']:.3f}s", flush=True)
                if failure:
                    raise RuntimeError(f'B3 stopped after ERROR on sample {index}') from failure
                if args.smoke_direct_solver:
                    # A sound CROWN precheck may finish the production chain early.
                    # Exercise actual Marabou separately without changing that chain.
                    net = network_from_nnet_file(working_file)
                    net, transformed = queries.reduce_property_to_basic_form(net, copy.deepcopy(prop))
                    direct = cegar.verify_network_with_marabou(net, transformed,
                        timeout=manifest['marabou_timeout_seconds'], save_query_path=run_dir/'smoke_direct.query')
                    dump(run_dir/'smoke_direct_result.json', direct)
                    if direct['status'] not in ['sat', 'unsat']:
                        raise RuntimeError('Supplementary smoke Marabou did not solve: '+str(direct))
                    if row['query_result'] in ['SAT','UNSAT'] and direct['status'].upper()!=row['query_result']:
                        raise AssertionError('Smoke CEGAR/direct solver result disagreement')
                    print(f"SMOKE MARABOU index={index} status={direct['status']}", flush=True)
                dump(output / 'progress.json', {'sample_index': index, 'property_id': pid, 'stage': 'completed', 'pid': os.getpid()})
    print('B3 COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--model', type=Path, default=ROOT/'data/models/mnist/mnist_fc_relu_64x3.onnx')
    parser.add_argument('--dataset-root', type=Path, default=ROOT/'data')
    parser.add_argument('--samples', type=Path, default=ROOT/'experiment_samples.json')
    parser.add_argument('--route', choices=['baseline','rednet'], default='baseline')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--smoke-direct-solver', action='store_true')
    parser.add_argument('--save-artifacts', action='store_true',
                        help='Keep detailed JSON, networks and queries; default: CSV results only')
    add_search_arguments(parser)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    np.random.seed(20260912)
    torch.manual_seed(20260912)
    if args.prepare:
        prepare(args)
    else:
        if args.output is None:
            parser.error('--output is required for verification')
        run(args)


if __name__ == '__main__':
    main()
