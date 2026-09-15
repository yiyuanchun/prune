"""Launch both formal routes only after the paired smoke tests have succeeded."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT/'experiment_samples.json').read_text())
    assert manifest['epsilon'] == 0.02 and len(manifest['sample_indices']) == 30
    snapshot = json.loads((ROOT/'results/b3_audit/baseline_source.json').read_text())
    for item in snapshot['files']:
        data = (ROOT/'parnv-MC'/item['file']).read_bytes()
        if hashlib.sha256(data).hexdigest() != item['main_sha256']:
            raise RuntimeError('Main baseline file changed: '+item['file'])
    smoke = {}
    for route in ['rednet', 'baseline']:
        folder = ROOT/'results'/('b3_smoke_'+route)
        rows = list(csv.DictReader(next(folder.glob('*.csv')).open()))
        assert len(rows) == 1 and int(rows[0]['sample_index']) == manifest['sample_indices'][0]
        assert rows[0]['verification_result'] in ['VERIFIED', 'UNSAFE']
        assert rows[0]['query_result'] in ['SAT','UNSAT']
        assert all(v != '' for k,v in rows[0].items() if k not in ['error_type','error_message','crown_precheck_status'])
        log = (ROOT/'logs'/('b3_smoke_'+route+'.log')).read_text()
        assert 'B3 COMPLETE' in log and 'Traceback' not in log
        direct = json.loads(next(folder.glob('mnist_*/smoke_direct_result.json')).read_text())
        assert direct['status'].upper() == rows[0]['query_result']
        smoke[route] = rows[0]
    assert smoke['rednet']['verification_result'] == smoke['baseline']['verification_result']
    assert smoke['rednet']['equivalence_passed'] == 'True'
    assert float(smoke['rednet']['equivalence_output_abs_error_max']) <= manifest['equivalence_tolerance']
    assert int(smoke['rednet']['equivalence_sample_count']) == 128
    assert '5 passed' in (ROOT/'results/b3_audit/tests.log').read_text()
    equivalence = json.loads((ROOT/'results/b3_audit/all_property_equivalence/summary.json').read_text())
    assert equivalence['passed']
    assert [r['sample_index'] for r in equivalence['samples']] == manifest['sample_indices']
    targets = [('rednet','8-11','b3_rednet_narv'), ('baseline','12-15','b3_baseline_par')]
    for route, cpus, name in targets:
        if (ROOT/'results'/name).exists() or (ROOT/'logs'/(name+'.log')).exists():
            raise FileExistsError('Refusing to replace or duplicate formal run '+name)
    info = dict(started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), smoke=smoke,
                sample_manifest=str(ROOT/'experiment_samples.json'), processes=[])
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    for route, cpus, name in targets:
        log = ROOT/'logs'/(name+'.log')
        output = ROOT/'results'/name
        command = ['taskset','-c',cpus,sys.executable,'-u',str(ROOT/'parnv-MC/b3_mnist.py'),
                   '--route',route,'--output',str(output)]
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT/'parnv-MC', env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        info['processes'].append(dict(route=route, pid=child.pid, log=str(log), output=str(output),
            csv=str(output/(name+'_eps002.csv')), cpu_affinity=cpus, command=command))
        (ROOT/'results/b3_audit/formal_launch.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
