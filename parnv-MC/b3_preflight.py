"""Check REDNet equivalence on every frozen property before formal timing runs."""
import json
from pathlib import Path
import torch

from b3_mnist import ROOT, property_for, dump
from core.pre_process.rednet_pipeline import prepare_rednet_nnet


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest = json.loads((ROOT/'experiment_samples.json').read_text())
    root = ROOT/'results/b3_audit/all_property_equivalence'
    root.mkdir(exist_ok=True)
    records = []
    for index in manifest['sample_indices']:
        _, _, prop, pid = property_for(manifest, index)
        if index == manifest['sample_indices'][0]:
            # Reuse the successful identical first-property smoke check.
            report = json.loads((ROOT/'results/b3_smoke_rednet'/pid/'rednet_report.json').read_text())
        else:
            report = prepare_rednet_nnet(manifest['nnet'], prop, root/pid/'rednet.nnet',
                equivalence_samples=manifest['equivalence_samples'],
                equivalence_tolerance=manifest['equivalence_tolerance'], seed=manifest['seed']+index)
        assert report['equivalence']['passed']
        dump(root/pid/'report.json', report)
        r, e = report['reduction'], report['equivalence']
        records.append(dict(sample_index=index, relus=r['reduced_relu_count'],
                            inactive_removed=r['total_inactive_removed'],
                            active_removed=r['total_active_eliminated'],
                            reconstructed=r['total_reconstructed_units'], max_error=e['max_abs_error']))
        print('PREFLIGHT',records[-1],flush=True)
    dump(root/'summary.json',dict(passed=True, samples=records))
    print('ALL PROPERTY EQUIVALENCE PASSED', flush=True)


if __name__ == '__main__':
    main()
