"""Prepare identical 50-image manifests and launch REDNet at eps .02 and .03."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from b3_mnist import ROOT, digest, dump, property_for, _classifies_sample_correctly
from core.nnet.read_nnet import network_from_nnet_file
from core.utils.mnist_property_utils import _resolve_idx_path

RADII = (.02, .03)


def code_hashes():
    return {str(p.relative_to(ROOT)): digest(p)
            for p in (ROOT / 'parnv-MC').rglob('*.py')
            if not {'__pycache__', 'results', '.git'}.intersection(p.relative_to(ROOT).parts)}


def prepare(run, commit, base_manifest=None):
    if run.exists():
        raise FileExistsError('Refusing to overwrite experiment directory: ' + str(run))
    base = json.loads((base_manifest or ROOT / 'experiment_samples.json').read_text())
    for key in ['model', 'nnet']:
        assert digest(base[key]) == base[key + '_sha256']
    for name, expected in base['dataset_sha256'].items():
        assert digest(_resolve_idx_path(base['dataset_root'], name)) == expected
    network = network_from_nnet_file(base['nnet'])
    selected, skipped, labels = [], [], {}
    index = 0
    while len(selected) < 50:
        sample, label, _, _ = property_for(base, index)
        if _classifies_sample_correctly(network, {'sample': sample.tolist(), 'label': label}):
            selected.append(index)
            labels[str(index)] = label
        else:
            skipped.append(index)
        index += 1
    run.mkdir(parents=True)
    hashes = code_hashes()
    for name in hashes:
        target = run / 'audit/source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    dump(run / 'audit/source_hashes.json', hashes)
    dump(run / 'sample_indices.json', dict(sample_count=50, dataset_split=base['dataset_split'],
         sample_indices=selected, skipped_indices=skipped, labels=labels))
    for epsilon in RADII:
        tag = format(epsilon, 'g').replace('.', '')
        manifest = dict(base, epsilon=epsilon, sample_count=50, sample_indices=selected,
                        skipped_indices=skipped, marabou_timeout_seconds=3600,
                        source_commit=commit, experiment='rednet_50_initial_pgd')
        dump(run / ('samples_eps' + tag + '.json'), manifest)
        for i in selected:
            _, _, prop, pid = property_for(manifest, i)
            dump(run / 'properties' / (pid + '.json'), prop)
    print(json.dumps(dict(run_root=str(run), sample_indices=selected, skipped_indices=skipped,
                         source_files=len(hashes), source_commit=commit)), flush=True)


def launch(run, smoke):
    hashes = json.loads((run / 'audit/source_hashes.json').read_text())
    assert code_hashes() == hashes, 'Source changed since preparation; use a fresh experiment directory'
    manifests = [run / ('samples_eps' + format(eps, 'g').replace('.', '') + '.json') for eps in RADII]
    configs = [json.loads(p.read_text()) for p in manifests]
    assert configs[0]['sample_indices'] == configs[1]['sample_indices']
    for epsilon, cfg in zip(RADII, configs):
        assert cfg['epsilon'] == epsilon and cfg['sample_count'] == 50
        assert len(set(cfg['sample_indices'])) == 50 and cfg['marabou_timeout_seconds'] == 3600
    if not smoke:
        for epsilon in RADII:
            tag = format(epsilon, 'g').replace('.', '')
            folder = run / ('smoke_eps' + tag)
            rows = list(csv.DictReader(next(folder.glob('*.csv')).open()))
            assert len(rows) == 1 and rows[0]['verification_result'] in ['VERIFIED', 'UNSAFE']
            assert rows[0]['equivalence_passed'] == 'True' and not rows[0]['error_message']
            assert int(rows[0]['marabou_calls']) == int(rows[0]['cegar_iterations'])
            assert float(rows[0]['epsilon']) == epsilon and int(rows[0]['pgd_calls']) > 0
            assert 'B3 COMPLETE' in (run / ('smoke_eps' + tag + '.log')).read_text()
            direct = json.loads(next(folder.glob('mnist_*/smoke_direct_result.json')).read_text())
            assert direct['status'].upper() == rows[0]['query_result']
        tests = (run / 'audit/tests.log').read_text()
        assert 'passed' in tests and 'failed' not in tests and 'ERROR' not in tests
    phase = 'smoke' if smoke else 'full'
    targets = []
    for epsilon, manifest, cpus in zip(RADII, manifests, ['8-11', '12-15']):
        name = phase + '_eps' + format(epsilon, 'g').replace('.', '')
        output, log = run / name, run / (name + '.log')
        if output.exists() or log.exists():
            raise FileExistsError('Refusing to overwrite or duplicate run ' + name)
        targets.append((epsilon, manifest, cpus, output, log))
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    info = dict(started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), processes=[])
    for epsilon, manifest, cpus, output, log in targets:
        command = ['taskset', '-c', cpus, sys.executable, '-u', str(ROOT / 'parnv-MC/b3_mnist.py'),
                   '--route', 'rednet', '--samples', str(manifest), '--output', str(output)]
        if smoke:
            command += ['--limit', '1', '--smoke-direct-solver']
        with log.open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT / 'parnv-MC', env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        info['processes'].append(dict(epsilon=epsilon, pid=process.pid, cpu_affinity=cpus,
             manifest=str(manifest), output=str(output), log=str(log), command=command))
        dump(run / (phase + '_launch.json'), info)
    print(json.dumps(info, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'smoke', 'full'])
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--source-commit', default='unspecified')
    parser.add_argument('--base-manifest', type=Path, default=None,
                        help='Base configuration with model/data paths for this server')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.run_root.resolve(), args.source_commit, args.base_manifest)
    else:
        launch(args.run_root.resolve(), args.phase == 'smoke')


if __name__ == '__main__':
    main()
