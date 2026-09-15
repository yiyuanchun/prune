"""I/O contracts without optional Torch/Marabou; solver calls are stubs.

Production functions are loaded by AST to isolate file handling from the ML
imports. These tests do not claim to validate numerical verification results.
"""
import ast
import contextlib
import csv
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import patch

MC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MC))
from core.utils.experiment_output import (
    artifacts_enabled, diagnostic_artifacts, sample_workspace, solver_console,
)


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
                      names=[ast.alias(name='annotations')], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return SimpleNamespace(**namespace)


class OutputTests(unittest.TestCase):
    def test_context_and_temp_cleanup_after_error(self):
        self.assertTrue(artifacts_enabled())
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            with diagnostic_artifacts(False), sample_workspace(Path('unused')) as path:
                (path / 'rednet.nnet').write_text('required transient network')
                with diagnostic_artifacts(True):
                    self.assertTrue(artifacts_enabled())
                self.assertFalse(artifacts_enabled())
                raise RuntimeError('failure')
        self.assertFalse(path.exists())
        self.assertTrue(artifacts_enabled())

    def test_disabled_writers_do_not_serialize_or_save_env_query(self):
        raw = load_functions(MC / 'core/cegar/raw_cegar.py',
                             ['write_nnet', '_write_json'], dict(artifacts_enabled=artifacts_enabled))
        query = load_functions(MC / 'core/utils/marabou_query_utils.py', ['_try_save_query'],
                               dict(artifacts_enabled=artifacts_enabled))
        with diagnostic_artifacts(False), patch.dict(os.environ, NARV_SAVE_QUERY_PATH='unwanted.query'):
            # Disabled writers must exit before inspecting/serializing arguments.
            raw.write_nnet(object(), None)
            raw._write_json(None, object())
            query._try_save_query(object(), None, True)

    def test_console_keeps_errors_and_restores_stdout(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with diagnostic_artifacts(False), solver_console():
                print('hidden progress')
                print('visible error', file=sys.stderr)
            print('summary')
        self.assertEqual(out.getvalue(), 'summary\n')
        self.assertEqual(err.getvalue(), 'visible error\n')

    def run_driver(self, root, detailed, fail=False):
        source = root / 'source.nnet'
        source.write_text('source')
        manifest_path = root / 'samples.json'
        manifest_path.write_text(json.dumps(dict(model=str(source), nnet=str(source),
            model_sha256='hash', nnet_sha256='hash', dataset_sha256={}, dataset_split='train',
            sample_count=2, sample_indices=[0, 1], epsilon=.02, seed=123,
            equivalence_samples=128, equivalence_tolerance=1e-8,
            max_refinement_steps=None, marabou_timeout_seconds=3600, tolerance=1e-7)))
        events, workspaces = [], []
        output = root / ('detailed' if detailed else 'light')
        stats = ['pgd_calls', 'pgd_time_seconds', 'pgd_candidates_found',
                 'pgd_genuine_counterexamples', 'pgd_spurious_counterexamples',
                 'batch_refinement_rounds', 'extra_refinement_steps', 'formal_verifier_calls']
        tree = ast.parse((MC / 'b3_mnist.py').read_text(encoding='utf-8-sig'))
        columns = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id in ['COMMON', 'REDNET', 'BASELINE']}

        def rednet(original, prop, target, **kwargs):
            events.append(('rednet', prop, kwargs))
            workspaces.append(target.parent)
            target.write_text('serialized and checked REDNet')
            return dict(output_nnet_file=str(target), reduction_time_seconds=.1,
                preprocessing_time_seconds=.2, crown_time_seconds=.1,
                reduction=dict(per_layer=[dict(stable_inactive=1, stable_active=2)],
                    total_inactive_removed=1, total_active_eliminated=2,
                    total_reconstructed_units=1, reduced_relu_count=190, relu_removed=2,
                    original_relu_count=192),
                equivalence=dict(samples=128, mean_abs_error=0., variance_abs_error=0.,
                                 max_abs_error=0., passed=True))

        active_metrics = []

        @contextlib.contextmanager
        def observe(metrics, concrete):
            active_metrics[:] = [metrics]
            yield

        def verify(working_file, prop, destination, **kwargs):
            self.assertEqual(Path(working_file).read_text(), 'serialized and checked REDNet')
            events.append(('cegar', prop, kwargs))
            if fail:
                raise ValueError('injected solver failure')
            active_metrics[0]['marabou_calls'] = 1
            return dict(**dict.fromkeys(stats, 0), query_result='UNSAT', iterations=1,
                        cegar_iterations=1, total_refinement_steps=2)

        namespace = dict(columns, Path=Path, json=json, csv=csv, os=os, time=time,
            traceback=traceback, np=SimpleNamespace(isfinite=math.isfinite),
            artifacts_enabled=artifacts_enabled, diagnostic_artifacts=diagnostic_artifacts,
            sample_workspace=sample_workspace, solver_console=solver_console,
            digest=lambda p: 'hash', network_from_nnet_file=lambda p: object(),
            property_for=lambda m, i: (SimpleNamespace(tolist=lambda: [0.]), 0, {'index': i}, f'sample{i}'),
            _classifies_sample_correctly=lambda *a: True, PROGRESSIVE_FIELDS=[],
            observe=observe, rednet_pipeline=SimpleNamespace(prepare_rednet_nnet=rednet),
            cegar=SimpleNamespace(__file__=str(source), cegar_verify_with_marabou=verify),
            search_options=lambda a: dict(pgd_config=SimpleNamespace(to_dict=lambda: {})))
        driver = load_functions(MC / 'b3_mnist.py', ['dump', 'run', '_run'], namespace)
        args = SimpleNamespace(samples=manifest_path, output=output, route='rednet', limit=None,
                               smoke_direct_solver=False, extra_refinement_merges=3, save_artifacts=detailed)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'stopped after ERROR'):
                    driver.run(args)
            else:
                driver.run(args)
        with (output / 'b3_rednet_narv_eps002.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        return events, workspaces, output, rows

    def test_driver_csv_only_matches_detailed_solver_calls_and_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            light = self.run_driver(root, False)
            detailed = self.run_driver(root, True)
            self.assertEqual([e[:2] for e in light[0]], [e[:2] for e in detailed[0]])
            self.assertEqual(len(light[0]), 4)
            for event in light[0]:
                if event[0] == 'cegar':
                    self.assertEqual(event[2]['marabou_timeout'], 3600)
                else:
                    self.assertEqual(event[2]['equivalence_samples'], 128)
            for a, b in zip(light[3], detailed[3]):
                for key in a:
                    if not key.endswith('_seconds'):
                        self.assertEqual(a[key], b[key], key)
                self.assertEqual(a['marabou_calls'], '1')
                self.assertEqual(a['verification_result'], 'VERIFIED')
            self.assertEqual([p.name for p in light[2].iterdir()], ['b3_rednet_narv_eps002.csv'])
            self.assertTrue(all(not p.exists() for p in light[1]))
            self.assertTrue((detailed[2] / 'run_config.json').exists())
            self.assertTrue((detailed[2] / 'sample1/row.json').exists())
            self.assertTrue(all(p.exists() for p in detailed[1]))

    def test_driver_error_flushes_csv_and_cleans_transient_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            events, paths, output, rows = self.run_driver(Path(tmp), False, fail=True)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['verification_result'], 'ERROR')
            self.assertEqual(rows[0]['error_message'], 'injected solver failure')
            self.assertTrue(all(not p.exists() for p in paths))
            self.assertEqual(len(list(output.iterdir())), 1)


if __name__ == '__main__':
    unittest.main()
