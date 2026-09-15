# B3 MNIST experiment: REDNet + current NARv-like CEGAR

## Frozen protocol

Remote root: `/home/test/yiyuanchun/Prune-v2`.
Environment: `/home/test/.conda/envs/alpha-beta-crown`.
Source model: `data/models/mnist/mnist_fc_relu_64x3.onnx`.
Architecture: 784 -> 64 -> 64 -> 64 -> 10, with 192 hidden ReLUs.

The main baseline is pinned to GitHub commit
`9706f4c24522422540205d6f631f6b2764aefa29`. The remote deployment has no `.git`
directory. Eight relevant source files were downloaded at that commit and
compared byte-for-byte to the deployment. All matched. The copies and SHA-256
records are in `results/b3_audit/main_snapshot/` and `baseline_source.json`.
The original main batch, CEGAR, CROWN, pruning, property and Marabou modules
are left unchanged. The launcher checks their hashes again before starting.

Both routes share `experiment_samples.json`: training split, L-infinity radius
0.02, clipped input bounds `[max(0,p-0.02), min(1,p+0.02)]`, output disjunction
`logit[j]-logit[label] >= 2.220446049250313e-16` for any other label.
The existing MNIST loader, property builder and batch classification filter
are reused. All indices 0 through 29 classify correctly, so no sample is skipped.
The JSON freezes model/data hashes, seed, settings and indices. Property files
are also saved in `b3_properties/`.

Marabou timeout is 3600 seconds **per query**, maximum refinement steps is
unlimited (`None`), and the existing tolerance argument is 1e-7. Main's batch
parser exposes a property timeout argument, but its `par` dispatch does not
pass/enforce it. This experiment preserves that behavior: neither route adds
a property-wide time limit. Likewise, the existing concrete checker does not
use the tolerance argument to relax the output threshold.

The neighboring original `.nnet` rounds weights with maximum difference about
5e-11 from the ONNX parameters. The project's existing converter therefore
creates one independent shared input at
`results/b3_audit/mnist_fc_relu_64x3_from_onnx.nnet`, without overwriting the old
file. Identity normalization and all layer dimensions are checked. Across 158
inputs, parsed-ONNX float64 versus exported-NNet float64 maximum output error is
5.377e-12; ONNX Runtime float32 versus NNet float64 maximum error is 2.410e-5.
These are distinct measurements: the REDNet error columns compare the same
float64 mathematical classifier representation before and after reduction.

## Method and exactness

The provided paper, *Expediting Neural Network Verification via Network
Reduction*, arXiv:2308.03330v2, Section IV-B, equations (4)-(8), Figure 5 and
Lemma 1, defines the implemented reconstruction.

For active indices A with incoming W_A, bias b_A, outgoing V_A, and predecessor
activation z, their contribution is V_A W_A z + V_A b_A. Define F=V_A W_A and
d=V_A b_A. Choose a shift s such that Fz+s >= 0 throughout the predecessor
box. Replace those active units with ReLU(Fz+s), connect them to the successor
with the identity matrix, and change successor bias b to b+d-s. Then

    I ReLU(Fz+s) + b+d-s = Fz+b+d.

Inactive units contribute zero. Unstable units keep their original connections.
Layers are processed backwards, and reconstruction is performed only when the
number of active units exceeds the already reduced successor width. Predecessor
coordinates have not yet been rewritten at that point, so their original CROWN
bounds still apply. Earlier rewrites preserve the later preactivations as well.
This is the paper's exact affine rewrite, with no abstraction approximation.

The implementation uses existing property-specific CROWN bounds, with inactive
upper bounds <= -1e-9 and active lower bounds >= 1e-9. Main baseline retains its
existing inactive test `< 0`. Near-zero values are conservatively retained by
REDNet. Reconstructed shifts include a float64 rounding allowance for the
interval dot products. Nonfinite parameters/bounds/outputs are rejected.

As in the paper's Section VII, algebraic equivalence is over the real numbers;
floating-point CROWN, matrix multiplication and serialization have numerical
roundoff. Sampling is an implementation check, not a proof over the entire box.
Each property uses 128 uniform random inputs with seed 20260912+sample_index,
and computes `e(x)=max_i |N_i(x)-N_R_i(x)|`. Mean, population variance (`ddof=0`)
and maximum of these nonnegative errors are recorded. Both the in-memory and
reloaded exported networks must pass absolute tolerance 1e-8. Failed checks
write ERROR and stop the route before CEGAR or further samples.

## Execution chains

REDNet route:

1. Original 10-logit NNet and this sample's property input box.
2. CROWN hidden bounds; exact inactive deletion and active folding/reconstruction.
3. In-memory and serialized-network equivalence checks.
4. Pass the resulting `rednet.nnet` to **the unchanged main CEGAR function**.
5. Existing nine-margin property conversion, CROWN precheck, and any additional
   dead-ReLU pruning on the reduced network.
6. Existing lazy inc/dec preprocessing and LP Merge on the last two hidden layers.
7. Marabou; original 10-logit network counterexample check; spurious examples
   trigger existing LIFO Merge undo/refinement.

Baseline route:

1. Shared original NNet and identical property.
2. Existing nine-margin conversion and CROWN precheck.
3. Existing inactive deletion, inc/dec preprocessing, LP Merge, Marabou and
   original-network counterexample check / Merge undo refinement.
4. No call to active reduction or reconstruction.

Main can conclude SAT/UNSAT during the CROWN precheck, before pruning/Merge.
This early return is intentionally preserved. Detected inactive count and actual
removed count are separate; an early return can have zero actual removals.

`b3_mnist.py` instruments calls with scoped wrappers that restore the originals
after each property. Timing/count observers do not change main's branching
logic. The REDNet concrete checker substitutes the original 10-logit network;
abstract/refined networks have nine margin outputs and retain their normal
evaluation. In both routes, incomplete, nonfinite or out-of-box concrete
witnesses are rejected as errors. Solver exceptions print a full traceback
before main's existing error-catching wrapper can discard it. CSV maps solver
TIMEOUT and ERROR separately from UNKNOWN. An ERROR row is flushed and the
process stops with its chained traceback.

## Measurement and resources

`total_verification_time_seconds` measures from entry to the route until final
CEGAR return/error. It includes network loading, reduction, model export,
equivalence checks and CEGAR. Shared dataset loading/property construction and
the supplementary smoke-only direct Marabou check are outside this interval
for both routes. CEGAR's own artifact writing is included.

`cegar_time_seconds` includes all work within main CEGAR, including its CROWN
calls. `crown_time_seconds` measures **all CROWN calls**, including repeated
Merge-scoring bounds and REDNet's initial bounds. It overlaps CEGAR time and
must not be added to CEGAR time. Initial, CEGAR and REDNet CROWN components are
also separately recorded. REDNet reduction time measures the rewrite/rebuild;
REDNet preprocessing time additionally includes its CROWN, I/O and checks.

`stable_active_eliminated_count` counts old active units removed, and
`reconstructed_stable_relu_count` counts replacement units. Their difference
is `stable_active_net_reduction_count`. `relu_reduction_ratio` is the fraction
removed `(original-after)/original`, not the paper's original/after size factor.
All counts refer to physical hidden ReLUs in the relevant network stage.

Hardware: two Intel Xeon Gold 6148 CPUs, 40 physical cores / 80 logical CPUs,
125 GiB RAM; GPUs 0/1 are RTX 4090 24 GiB, GPU 2 reports RTX 4090 48 GiB,
GPU 3 is RTX 3090 24 GiB. Full `nvidia-smi`, `free -h`, `lscpu`, CPU topology
and process snapshots are saved under `results/b3_audit/`.

Current main CROWN creates CPU tensors and modules, so both routes use CPU.
CUDA is hidden from both processes to avoid accidental device use. Both have
one PyTorch / BLAS / OpenMP thread and four distinct physical cores on the same
socket: REDNet CPU 8-11, baseline CPU 12-15. Neither uses the sibling SMT cores
of the other. Shared socket caches, memory bandwidth and unrelated server
processes can still affect wall-clock measurements; these are concurrent-run
results, not uncontended serial timing claims.

## Launch and artifacts

`launch_b3_mnist.py` requires both same-sample smoke runs to complete, consistent
solved results, passing REDNet equivalence, complete CSV fields, no traceback,
and supplementary actual Marabou solutions. `b3_preflight.py` additionally checks
all 30 fixed property boxes before launch (reusing sample 0's smoke report).
These checks are outside formal timings; each formal property recomputes its
own CROWN bounds and reduction, so no preflight cache advantages either route.
It refuses existing formal output
directories/logs, then starts both detached sessions without waiting for either.
Commands, PIDs and destinations are saved to `results/b3_audit/formal_launch.json`.

Formal REDNet outputs: `results/b3_rednet_narv/b3_rednet_narv_eps002.csv` and
`logs/b3_rednet_narv.log`.
Formal baseline outputs: `results/b3_baseline_par/b3_baseline_par_eps002.csv` and
`logs/b3_baseline_par.log`.
Each directory also contains `run_config.json`, `progress.json`, and one
subdirectory per actual sample with CEGAR queries/networks/mappings/logs.

The regression suite `tests/test_pre_process/test_b3_rednet.py` checks a
three-hidden-layer backward reconstruction at box corners and random points,
NNet round-trip equivalence, inactive-only behavior, absolute error statistics,
and rejection of NaN and zero-sample checks.

Smoke results on training sample 0, epsilon 0.02:

| Route | Result | Total seconds | CEGAR queries | Refinement steps | ReLUs after initial reduction |
|---|---|---:|---:|---:|---:|
| REDNet + NARv-like | VERIFIED / UNSAT | 153.351 | 2 | 13 | 109 |
| Main baseline -par | VERIFIED / UNSAT | 367.673 | 9 | 62 | 150 |

Both supplemental direct Marabou checks also returned UNSAT; both logs completed
without a traceback. REDNet removed 42 inactive units and replaced 75 active
units with 34 new units, saving another 41. Its 128-point serialized-output
absolute error mean/variance/max were respectively 7.3941e-12, 5.3841e-29 and
7.4252e-12. The observer confirms the spurious counterexample was checked once
against the actual original 10-logit network before refinement. Smoke timing
is diagnostic only and is not a statistical performance conclusion.

Other server workloads include CPU-intensive `gzclient` processes. No exclusive
reservation of the CPU, memory bandwidth or cache has been made. Despite
disjoint affinities, external CPU load is a remaining measurement limitation;
see `cpu_load_during_smoke.json` and the process snapshots.

All 30 property-specific REDNet checks passed before formal launch. The largest
128-point maximum error across them was 9.73355e-12; reduced networks contain
78 through 159 ReLUs. Full per-property reports and serialized networks are in
`results/b3_audit/all_property_equivalence/`.

Formal jobs launched together on 2026-09-12:

- REDNet + NARv-like PID: **1456076**, CPU affinity 8-11.
- Main baseline -par PID: **1456077**, CPU affinity 12-15.

The handoff snapshot, including `ps` output, log sizes and current samples, is
saved in `results/b3_audit/formal_status_at_handoff.json`. The requested formal
experiments run independently after the interactive session ends; their full
30-sample completion is not awaited.
