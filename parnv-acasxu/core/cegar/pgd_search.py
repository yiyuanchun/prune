"""Property-directed, box-constrained search. A failed search is never a proof."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from core.utils.verification_properties_utils import is_satisfying_assignment

DEFAULT_EXTRA_REFINEMENT_MERGES = 3


@dataclass(frozen=True)
class PGDConfig:
    enabled: bool = True
    steps: int = 40
    restarts: int = 5
    step_size: float | None = None  # Absolute pixel step; None uses box width / 10.
    seed: int = 0

    def __post_init__(self):
        if self.steps < 0 or self.restarts < 1:
            raise ValueError('PGD requires steps >= 0 and restarts >= 1')
        if self.step_size is not None and (not math.isfinite(self.step_size) or self.step_size <= 0):
            raise ValueError('PGD step_size must be finite and positive')

    def to_dict(self):
        return asdict(self)


def add_search_arguments(parser):
    parser.add_argument('--extra-refinement-merges', type=int, default=DEFAULT_EXTRA_REFINEMENT_MERGES)
    parser.add_argument('--pgd-steps', type=int, default=40)
    parser.add_argument('--pgd-restarts', type=int, default=5)
    parser.add_argument('--pgd-step-size', type=float, default=None)
    parser.add_argument('--pgd-seed', type=int, default=0)
    parser.add_argument('--disable-pgd', action='store_true')


def search_options(args):
    return dict(extra_refinement_merges=args.extra_refinement_merges,
                pgd_config=PGDConfig(enabled=not args.disable_pgd, steps=args.pgd_steps,
                    restarts=args.pgd_restarts, step_size=args.pgd_step_size, seed=args.pgd_seed))


def violation_objective(outputs, property_spec):
    """Margin for the already transformed property; no cross-entropy objective.

    This describes the search direction only. Acceptance uses the existing
    property predicate on a fresh concrete forward evaluation below.
    """
    import torch
    terms = []
    adversarial = property_spec.get('type') == 'adversarial'
    for index, bounds in property_spec['output']:
        if adversarial:
            if property_spec.get('_adversarial_violation_operator', 'ge') == 'le':
                terms.append(float(bounds.get('Upper', 0.)) - outputs[:, int(index)])
            else:
                terms.append(outputs[:, int(index)] - float(bounds.get('Lower', 0.)))
        else:
            if 'Lower' in bounds:
                terms.append(outputs[:, int(index)] - float(bounds['Lower']))
            if 'Upper' in bounds:
                terms.append(float(bounds['Upper']) - outputs[:, int(index)])
    margins = torch.stack(terms, dim=1)
    return margins.max(dim=1).values if adversarial else margins.min(dim=1).values


def search_counterexample(network, property_spec, config: PGDConfig | None = None, seed_offset=0):
    config = config or PGDConfig()
    start = time.perf_counter()

    def finish(found=False, counterexample=None, reason='', steps=0):
        return dict(found=found, counterexample=counterexample or {}, reason=reason,
                    steps_executed=steps, time_seconds=time.perf_counter()-start)

    if not config.enabled:
        return finish(reason='disabled')
    kind = property_spec.get('type')
    if kind not in ['basic', 'adversarial']:
        return finish(reason='unsupported property; defer to formal verifier')
    if kind == 'adversarial' and property_spec.get('_adversarial_query_mode', 'disjunction') != 'disjunction':
        return finish(reason='non-disjunctive adversarial encoding; defer to formal verifier')
    if kind == 'adversarial' and property_spec.get('_adversarial_violation_operator', 'ge') not in ['ge', 'le']:
        return finish(reason='unsupported violation operator; defer to formal verifier')
    specs = property_spec.get('output', [])
    output_size = len(network.layers[-1].nodes)
    if not specs or any(int(i) < 0 or int(i) >= output_size for i, _ in specs):
        return finish(reason='unsupported output constraints; defer to formal verifier')
    if kind == 'adversarial' and property_spec.get('_adversarial_target_label') is not None and output_size > len(specs):
        return finish(reason='PGD requires transformed margin outputs')
    if kind == 'basic' and not any('Lower' in b or 'Upper' in b for _, b in specs):
        return finish(reason='empty objective; defer to formal verifier')
    for layer in network.layers[1:-1]:
        if any(getattr(n.activation_func, '__name__', '') != 'relu' for n in layer.nodes):
            return finish(reason='non-ReLU hidden activation; defer to formal verifier')

    import torch
    input_size = len(network.layers[0].nodes)
    bounds = {int(i): b for i, b in property_spec['input']}
    if set(bounds) != set(range(input_size)):
        raise ValueError('PGD needs a complete finite property input box')
    lower = torch.tensor([bounds[i]['Lower'] for i in range(input_size)], dtype=torch.float64)
    upper = torch.tensor([bounds[i]['Upper'] for i in range(input_size)], dtype=torch.float64)
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or (lower > upper).any():
        raise ValueError('PGD input bounds must be finite and ordered')
    weights = [torch.tensor(w, dtype=torch.float64) for w in network.generate_weights()]
    biases = [torch.tensor(b, dtype=torch.float64) for b in network.generate_biases()]
    if not all(torch.isfinite(v).all() for v in weights + biases):
        raise ValueError('PGD network contains nonfinite parameters')

    def forward(x):
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            x = x @ weight.T + bias
            if i < len(weights)-1:
                x = torch.relu(x)
        return x

    generator = torch.Generator(device='cpu').manual_seed(config.seed + int(seed_offset))
    x = lower + torch.rand((config.restarts, input_size), generator=generator, dtype=torch.float64)*(upper-lower)
    # One deterministic midpoint and the remaining independent random restarts.
    x[0] = (lower+upper)/2
    step_size = (upper-lower)/10 if config.step_size is None else torch.full_like(lower, config.step_size)
    _, variables2nodes = network.get_variables(property_type=kind)
    for step in range(config.steps+1):
        x = x.detach().requires_grad_(True)
        output = forward(x)
        margin = violation_objective(output, property_spec)
        if not torch.isfinite(margin).all():
            raise ValueError('Nonfinite PGD objective')
        for row in torch.nonzero(margin >= 0, as_tuple=False).flatten().tolist():
            candidate = {i: float(v) for i,v in enumerate(x[row].detach().tolist())}
            # Defend the search/verifier boundary against rounding or future changes.
            if not all(bounds[i]['Lower'] <= candidate[i] <= bounds[i]['Upper'] for i in bounds):
                raise ValueError('PGD candidate escaped its input box')
            values = np.asarray(network.speedy_evaluate(candidate), dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError('Nonfinite PGD candidate output')
            if is_satisfying_assignment(network, property_spec, values, variables2nodes):
                return finish(True, candidate, 'candidate validated on current network', step)
        if step == config.steps:
            break
        gradient, = torch.autograd.grad(margin.sum(), x)
        if not torch.isfinite(gradient).all():
            raise ValueError('Nonfinite PGD gradient')
        with torch.no_grad():
            x = torch.maximum(lower, torch.minimum(upper, x + step_size*gradient.sign()))
    return finish(reason='no counterexample found; formal verification required', steps=config.steps)
