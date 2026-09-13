from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.nnet.read_nnet import network_from_nnet_file, write_nnet_file
from core.pre_process.crown_bounds import compute_crown_hidden_layer_bounds
from core.pre_process.dead_relu_pruning import apply_input_bounds_from_property
from core.pre_process.stable_relu_reduction import reduce_stable_relu_neurons


def _sample_input(lower_bounds: list[float], upper_bounds: list[float], rng: np.random.Generator) -> dict[int, float]:
    return {
        index: float(rng.uniform(low, high)) if high > low else float(low)
        for index, (low, high) in enumerate(zip(lower_bounds, upper_bounds))
    }


def _input_bounds(network) -> tuple[list[float], list[float]]:
    return (
        [float(node.lower_bound) for node in network.layers[0].nodes],
        [float(node.upper_bound) for node in network.layers[0].nodes],
    )


def check_input_output_equivalence(
        original_network,
        reduced_network,
        samples: int = 64,
        tolerance: float = 1e-5,
        seed: int = 0,
) -> dict[str, Any]:
    if samples <= 0 or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Equivalence checks require positive samples and a finite non-negative tolerance.")
    lower_bounds, upper_bounds = _input_bounds(original_network)
    if not np.all(np.isfinite([lower_bounds, upper_bounds])) or np.any(np.asarray(lower_bounds) > upper_bounds):
        raise ValueError("Equivalence input box must be finite and ordered.")
    rng = np.random.default_rng(seed)
    max_error = 0.0
    errors = []
    worst_case: dict[str, Any] | None = None
    for sample_index in range(int(samples)):
        sample = _sample_input(lower_bounds, upper_bounds, rng)
        original_output = np.asarray(original_network.speedy_evaluate(sample), dtype=float)
        reduced_output = np.asarray(reduced_network.speedy_evaluate(sample), dtype=float)
        if original_output.shape != reduced_output.shape or not original_output.size:
            raise ValueError("Equivalence output dimensions differ or are empty.")
        if not np.all(np.isfinite(original_output)) or not np.all(np.isfinite(reduced_output)):
            raise ValueError("Non-finite network output in equivalence check.")
        error = float(np.max(np.abs(original_output - reduced_output))) if original_output.size else 0.0
        errors.append(error)
        if error > max_error:
            max_error = error
            worst_case = {
                "sample_index": sample_index,
                "input": [sample[index] for index in range(len(sample))],
                "original_output": original_output.tolist(),
                "reduced_output": reduced_output.tolist(),
                "max_abs_error": error,
            }
    return {
        "samples": int(samples),
        "tolerance": float(tolerance),
        "max_abs_error": float(max_error),
        "mean_abs_error": float(np.mean(errors)),
        "variance_abs_error": float(np.var(errors, ddof=0)),
        "seed": int(seed),
        "passed": bool(max_error <= tolerance),
        "worst_case": worst_case,
    }


def prepare_rednet_nnet(
        original_nnet_file: str | Path,
        test_property: dict[str, Any],
        output_nnet_file: str | Path,
        equivalence_samples: int = 64,
        equivalence_tolerance: float = 1e-5,
        seed: int = 0,
        stability_tolerance: float = 1e-9,
        activation_margin: float = 1e-9,
) -> dict[str, Any]:
    """Build an exact REDNet-style reduced .nnet for one verification property."""
    started = time.perf_counter()
    original_network = network_from_nnet_file(str(original_nnet_file))
    apply_input_bounds_from_property(original_network, copy.deepcopy(test_property))

    crown_started = time.perf_counter()
    crown_bounds = compute_crown_hidden_layer_bounds(original_network)
    crown_time = time.perf_counter() - crown_started

    reduction_started = time.perf_counter()
    reduced_network, reduction_report = reduce_stable_relu_neurons(
        network=original_network,
        crown_bounds=crown_bounds,
        stability_tolerance=stability_tolerance,
        activation_margin=activation_margin,
        reconstruct_stably_active=True,
    )
    reduction_time = time.perf_counter() - reduction_started

    equivalence_report = check_input_output_equivalence(
        original_network=original_network,
        reduced_network=reduced_network,
        samples=equivalence_samples,
        tolerance=equivalence_tolerance,
        seed=seed,
    )
    if not equivalence_report["passed"]:
        raise RuntimeError(
            "REDNet equivalence smoke test failed: max_abs_error={} > tolerance={}".format(
                equivalence_report["max_abs_error"],
                equivalence_report["tolerance"],
            )
        )

    output_path = Path(output_nnet_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_nnet_file(
        output_path.as_posix(),
        weights=reduced_network.generate_weights(),
        biases=reduced_network.generate_biases(),
        metadata=getattr(reduced_network, "nnet_metadata", {}),
    )
    # The verifier reads the serialized network, so check that exact artifact too.
    reloaded = network_from_nnet_file(output_path.as_posix())
    serialized_equivalence = check_input_output_equivalence(
        original_network, reloaded, samples=equivalence_samples,
        tolerance=equivalence_tolerance, seed=seed,
    )
    if not serialized_equivalence["passed"]:
        raise RuntimeError("Serialized REDNet equivalence check failed: {}".format(serialized_equivalence))

    return {
        "output_nnet_file": output_path.as_posix(),
        "crown_time_seconds": float(crown_time),
        "reduction_time_seconds": float(reduction_time),
        "preprocessing_time_seconds": float(time.perf_counter() - started),
        "reduction": reduction_report,
        "equivalence_in_memory": equivalence_report,
        "equivalence": serialized_equivalence,
    }
