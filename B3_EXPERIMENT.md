# B3: REDNet + NARv complementarity experiment

## Goal

B3 asks one question: does exact stable-ReLU reduction (REDNet) improve the verification efficiency of the repository's current Merge/CEGAR pipeline when the REDNet preprocessing cost is included?

The REDNet preprocessing in this branch is property-specific and exact over the input box. It removes stably inactive ReLUs and, when profitable, reconstructs groups of stably active ReLUs into a smaller set of provably active ReLUs. The resulting network is then passed to the existing CROWN + Merge/CEGAR + Marabou flow.

> Important scope note: `core/cegar/raw_cegar.py` is currently a NARv-inspired local implementation, not a complete reproduction of Liu et al. 2024. It preprocesses and merges only the last two hidden layers and refines by undoing Merge operations. It does not yet implement the paper's full Merge+Freeze / Split+Recover / dependency-graph refinement. Therefore B3 results from this branch should be described as **REDNet + current PARnv/NARv-like Merge-CEGAR**, unless the backend is completed later.

## Code added for B3

- `parnv-acasxu/core/pre_process/stable_relu_reduction.py`
- `parnv-acasxu/core/pre_process/rednet_pipeline.py`
- `parnv-acasxu/b3_compare_rednet_narv.py`
- `parnv-acasxu/analyze_b3_rednet_narv.py`
- `parnv-acasxu/tests/test_pre_process/test_stable_relu_reduction.py`
- Matching REDNet preprocessing modules under `parnv-MC/core/pre_process/` for later MNIST/CIFAR-10 experiments.

## Correctness checks before performance experiments

Run from `parnv-acasxu`:

```bash
pytest -q tests/test_pre_process/test_stable_relu_reduction.py
```

The synthetic test checks both:

1. inactive-only exact deletion;
2. full REDNet-style active-neuron reconstruction.

For every real property, `prepare_rednet_nnet` also samples inputs from the verification box and compares original vs reduced network outputs. This is only an implementation smoke test; soundness comes from the stable-neuron rewrite itself and the sound CROWN bounds used to certify stability.

## Phase 1: small pilot

Use 5-20 ACAS Xu network/property pairs to catch environment and numerical issues.

```bash
cd parnv-acasxu

python b3_compare_rednet_narv.py \
  --route narv \
  --limit 20 \
  --marabou-timeout-seconds 3600 \
  --output-csv b3_narv_pilot.csv

python b3_compare_rednet_narv.py \
  --route rednet-narv \
  --limit 20 \
  --marabou-timeout-seconds 3600 \
  --equivalence-samples 128 \
  --output-csv b3_rednet_narv_pilot.csv

python analyze_b3_rednet_narv.py \
  --narv-csv b3_narv_pilot.csv \
  --rednet-narv-csv b3_rednet_narv_pilot.csv \
  --timeout-seconds 3600 \
  --paired-output-csv b3_pilot_paired.csv \
  --summary-json b3_pilot_summary.json
```

Pilot acceptance criteria:

- `result_mismatches == 0` for instances solved by both routes;
- all REDNet equivalence smoke tests pass;
- no NaN/Inf in generated `.nnet` files;
- REDNet preprocessing time is recorded and included in `total_time_seconds`;
- at least some properties have `relu_reduction_ratio > 1` before proceeding to the full batch.

## Phase 2: ACAS Xu full paired experiment

Run the two routes in separate processes. Do not run one immediately inside the other in the same process; separate runs reduce warm-cache/import bias.

```bash
python b3_compare_rednet_narv.py \
  --route narv \
  --marabou-timeout-seconds 3600 \
  --output-csv b3_narv_acas.csv

python b3_compare_rednet_narv.py \
  --route rednet-narv \
  --marabou-timeout-seconds 3600 \
  --equivalence-samples 128 \
  --output-csv b3_rednet_narv_acas.csv

python analyze_b3_rednet_narv.py \
  --narv-csv b3_narv_acas.csv \
  --rednet-narv-csv b3_rednet_narv_acas.csv \
  --timeout-seconds 3600 \
  --paired-output-csv b3_acas_paired.csv \
  --summary-json b3_acas_summary.json
```

Run on an otherwise idle machine. Record CPU, RAM, GPU, Python, PyTorch, auto_LiRPA and Marabou versions. Use the same timeout, property set and model set for both routes.

## Primary metrics

The B3 runner records the following per property:

- final result and query result;
- wall-clock `total_time_seconds`;
- CEGAR-only time;
- REDNet preprocessing, CROWN and rewrite times;
- original and reduced ReLU counts;
- reduction ratio;
- stable inactive neurons removed;
- stable active neurons eliminated and reconstruction units inserted;
- implementation equivalence max absolute error;
- CEGAR iterations and refinement steps;
- dead ReLUs removed again inside the current NARv pipeline.

The paired analyzer reports:

- number of solved instances;
- mean/median solved runtime;
- mean PAR-2 score, counting an unsolved instance as `2 * timeout`;
- geometric-mean and median speedup on commonly solved instances;
- fraction of paired instances where REDNet+NARv is faster;
- average reduction ratio;
- average refinement-step count;
- result mismatches.

## What counts as complementarity

The primary comparison is:

- `NARv`: current CROWN + Merge/CEGAR + Marabou route on the original network;
- `REDNet+NARv`: exact REDNet reduction first, then the exact same CEGAR route.

Evidence of complementarity should include all of the following:

1. no sound-result disagreement on paired solved instances;
2. lower PAR-2 and/or more solved properties for REDNet+NARv;
3. positive geometric-mean speedup after including REDNet preprocessing time;
4. a measurable decrease in ReLU count;
5. preferably fewer CEGAR iterations/refinement steps, or at least lower per-iteration solver cost.

A useful mechanistic plot is `speedup vs. ReLU reduction ratio`. A second plot should compare `refinement_steps(NARv)` with `refinement_steps(REDNet+NARv)`. These reveal whether REDNet helps because the verifier receives a smaller network, because fewer spurious counterexamples are generated, or both.

## Recommended factorial ablation for the paper

B3's core implementation compares NARv and REDNet+NARv. For a publishable ablation, additionally run two control arms using the same backend and timeout:

| REDNet | CEGAR | Meaning |
|---|---|---|
| off | off | verifier-only baseline |
| on | off | REDNet-only contribution |
| off | on | current NARv/PARnv contribution |
| on | on | B3 combined method |

The combined method is complementary if it improves over both single-technique arms on a paired workload. Do not report only average runtime on successful cases; always include solved count and a timeout-aware score such as PAR-2.

## Phase 3: larger fully-connected MNIST/CIFAR-10 networks

ACAS Xu is useful for correctness and reproducibility but is small. B3 should ultimately be repeated on fully-connected MNIST/CIFAR-10 models, because REDNet's benefit is expected to become more visible as the number of stable ReLUs grows. The same reduction modules are mirrored under `parnv-MC/core/pre_process/` for this next experiment. Keep the scope to fully-connected ReLU models so that no convolution-to-dense transformation confounds the B3 result.

Recommended perturbation radii should include easy, medium and hard regions rather than a single epsilon. For each model, use the same correctly classified sample indices for both routes and report results separately per epsilon before aggregating.

## Interpretation rules

- If network size drops substantially but total time does not, inspect REDNet CROWN preprocessing overhead and Marabou's per-iteration time.
- If runtime improves but refinement steps increase, the benefit is mainly smaller solver queries rather than fewer spurious counterexamples.
- If refinement steps decrease, REDNet is also simplifying the abstraction/refinement search.
- If reduction ratio is near 1 on ACAS Xu, do not conclude that the methods are non-complementary until the larger fully-connected benchmarks are tested.
- Any `VERIFIED`/`UNSAFE` disagreement must be treated as a correctness bug before speedup results are used.
