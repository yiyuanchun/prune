"""Launch isolated paired smoke/full MNIST runs with the new CEGAR policy."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/reduce_iteration_20260913'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'experiment_samples.json').read_text())
    assert manifest['epsilon'] == .02 and len(manifest['sample_indices']) == 30
    policy_hash = hashlib.sha256((ROOT / 'parnv-MC/core/cegar/raw_cegar.py').read_bytes()).hexdigest()
    smoke_rows = {}
    if not args.smoke:
        for route in ['rednet', 'baseline']:
            folder = RUN / ('smoke_' + route)
            config = json.loads((folder / 'run_config.json').read_text())
            assert config['cegar_source_sha256'] == policy_hash
            assert config['extra_refinement_merges'] == 3 and config['pgd']['enabled']
            assert config['pgd']['steps'] == 40 and config['pgd']['restarts'] == 5
            assert config['selected_indices'] == manifest['sample_indices'][:1]
            rows = list(csv.DictReader(next(folder.glob('*.csv')).open()))
            assert len(rows) == 1
            row = rows[0]
            assert row['verification_result'] in ['VERIFIED', 'UNSAFE'] and not row['error_message']
            assert int(row['marabou_calls']) == int(row['cegar_iterations'])
            assert int(row['pgd_calls']) > 0 and int(row['extra_refinement_steps']) > 0
            assert 'B3 COMPLETE' in (RUN / ('smoke_' + route + '.log')).read_text()
            direct = json.loads(next(folder.glob('mnist_*/smoke_direct_result.json')).read_text())
            assert direct['status'].upper() == row['query_result']
            smoke_rows[route] = row
        assert smoke_rows['rednet']['verification_result'] == smoke_rows['baseline']['verification_result']
        assert smoke_rows['rednet']['equivalence_passed'] == 'True'
        for name in ['tests_mc.log', 'tests_acasxu.log', 'tests_plane.log']:
            log = (RUN / 'audit' / name).read_text()
            assert 'passed' in log and 'failed' not in log and 'ERROR' not in log
    phase = 'smoke' if args.smoke else 'full'
    targets = [(route, cpus, RUN / (phase + '_' + route), RUN / (phase + '_' + route + '.log'))
               for route, cpus in [('rednet', '8-11'), ('baseline', '12-15')]]
    for _, _, output, log in targets:
        if output.exists() or log.exists():
            raise FileExistsError('Refusing to replace an existing run: ' + str(output))
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    info = dict(started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), source_sha256=policy_hash,
                smoke=smoke_rows, processes=[])
    for route, cpus, output, log in targets:
        command = ['taskset', '-c', cpus, sys.executable, '-u', str(ROOT / 'parnv-MC/b3_mnist.py'),
                   '--route', route, '--output', str(output)]
        if args.smoke:
            command += ['--limit', '1', '--smoke-direct-solver']
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT / 'parnv-MC', env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        info['processes'].append(dict(route=route, pid=child.pid, output=str(output), log=str(log), command=command))
        (RUN / (phase + '_launch.json')).write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
