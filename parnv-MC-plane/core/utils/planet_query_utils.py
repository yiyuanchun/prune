from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


DEFAULT_PLANET_TIMEOUT_SECONDS = 1200
QUERY_LOG_ENV_VAR = "NARV_SAVE_PLANET_QUERY_PATH"
PLANET_BINARY_ENV_VAR = "PLANET_BIN"
PLANET_HOME_ENV_VAR = "PLANET_HOME"
LARGE_BOUND = 1.0e20


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_with_platform_suffix(path: Path) -> list[Path]:
    candidates = [path]
    if path.suffix == "":
        candidates.append(path.with_suffix(".exe"))
    return candidates


def resolve_planet_binary(planet_bin: str | os.PathLike[str] | None = None) -> Path:
    candidates: list[str | os.PathLike[str]] = []
    if planet_bin is not None:
        candidates.append(planet_bin)
    env_bin = os.environ.get(PLANET_BINARY_ENV_VAR)
    if env_bin:
        candidates.append(env_bin)
    env_home = os.environ.get(PLANET_HOME_ENV_VAR)
    if env_home:
        candidates.append(Path(env_home) / "src" / "planet")
    candidates.extend([
        _project_root() / "planet" / "src" / "planet",
        _project_root().parent / "planet" / "src" / "planet",
        "planet",
    ])

    checked: list[str] = []
    for candidate in candidates:
        candidate_text = str(candidate).strip()
        if not candidate_text:
            continue
        candidate_path = Path(candidate_text).expanduser()
        has_path_part = candidate_path.parent != Path(".") or candidate_path.is_absolute()
        if has_path_part:
            for platform_candidate in _candidate_with_platform_suffix(candidate_path):
                checked.append(str(platform_candidate))
                if platform_candidate.is_file():
                    return platform_candidate.resolve()
        else:
            resolved = shutil.which(candidate_text)
            checked.append(candidate_text)
            if resolved:
                return Path(resolved).resolve()

    raise FileNotFoundError(
        "Planet executable was not found. Set PLANET_BIN to planet/src/planet "
        "or pass --planet-bin. Checked: {}".format(", ".join(checked))
    )


def prepare_planet_backend(planet_bin: str | os.PathLike[str] | None = None) -> Path:
    return resolve_planet_binary(planet_bin)


def _format_float(value: Any) -> str:
    return "{:.17g}".format(float(value))


def _node_line(kind: str, node: Any) -> str:
    edge_weights: dict[str, float] = {}
    for edge in node.in_edges:
        edge_weights[edge.src] = edge_weights.get(edge.src, 0.0) + float(edge.weight)

    parts = [kind, str(node.name), _format_float(getattr(node, "bias", 0.0))]
    for src, weight in edge_weights.items():
        parts.extend([_format_float(weight), str(src)])
    return " ".join(parts)


def _batch_aligned_output_specs(
        network: Any,
        test_property: dict[str, Any],
) -> list[dict[str, Any]]:
    output_nodes = network.layers[-1].nodes
    reference_index = int(test_property["_adversarial_reference_label"])
    if reference_index < 0 or reference_index >= len(output_nodes):
        raise ValueError(
            "Reference output index {} is invalid for a network with {} outputs.".format(
                reference_index,
                len(output_nodes),
            )
        )

    property_output_indices = {
        int(var_index)
        for var_index, _bounds in test_property.get("output", [])
    }
    non_reference_indices = sorted(
        output_index
        for output_index in property_output_indices
        if output_index != reference_index
    )
    if not non_reference_indices:
        raise ValueError(
            "adversarial property must include at least one non-reference output"
        )

    reference_node = output_nodes[reference_index]
    reference_edges = sorted(reference_node.in_edges, key=lambda edge: edge.src)
    output_layer_index = len(network.layers) - 1
    specs: list[dict[str, Any]] = []
    for comparison_index, wrong_index in enumerate(non_reference_indices):
        if wrong_index < 0 or wrong_index >= len(output_nodes):
            raise ValueError(
                "Wrong-class output index {} is invalid for a network with {} outputs.".format(
                    wrong_index,
                    len(output_nodes),
                )
            )
        wrong_node = output_nodes[wrong_index]
        wrong_edges = sorted(wrong_node.in_edges, key=lambda edge: edge.src)
        if len(reference_edges) != len(wrong_edges):
            raise ValueError(
                "Output nodes {} and {} have different input dimensions.".format(
                    reference_node.name,
                    wrong_node.name,
                )
            )

        weights: list[tuple[str, float]] = []
        for reference_edge, wrong_edge in zip(reference_edges, wrong_edges):
            if reference_edge.src != wrong_edge.src:
                raise ValueError(
                    "Output nodes {} and {} do not use the same input ordering.".format(
                        reference_node.name,
                        wrong_node.name,
                    )
                )
            weights.append(
                (
                    str(reference_edge.src),
                    float(wrong_edge.weight) - float(reference_edge.weight),
                )
            )
        specs.append(
            {
                "name": "x_{}_{}".format(output_layer_index, comparison_index),
                "bias": float(wrong_node.bias) - float(reference_node.bias),
                "weights": weights,
                "wrong_index": wrong_index,
            }
        )
    return specs


def _batch_aligned_output_line(spec: dict[str, Any]) -> str:
    parts = ["Linear", str(spec["name"]), _format_float(spec["bias"])]
    for src, weight in spec["weights"]:
        parts.extend([_format_float(weight), str(src)])
    return " ".join(parts)


def _network_lines(
        network: Any,
        test_property: dict[str, Any] | None = None,
) -> list[str]:
    lines: list[str] = []
    last_layer_index = len(network.layers) - 1
    for layer_index, layer in enumerate(network.layers):
        if layer_index == 0 or layer.type_name == "input":
            for node in layer.nodes:
                lines.append("Input {}".format(node.name))
        elif (
                layer_index == last_layer_index
                and test_property is not None
                and test_property.get("_planet_direct_output_comparison")
        ):
            lines.extend(
                _batch_aligned_output_line(spec)
                for spec in _batch_aligned_output_specs(network, test_property)
            )
        elif layer_index == last_layer_index or layer.type_name == "output":
            for node in layer.nodes:
                lines.append(_node_line("Linear", node))
        else:
            for node in layer.nodes:
                lines.append(_node_line("ReLU", node))
    return lines


def _assert_lower_bound(node_name: str, lower_bound: Any) -> str:
    return "Assert <= {} 1.0 {}".format(_format_float(lower_bound), node_name)


def _assert_upper_bound(node_name: str, upper_bound: Any) -> str:
    return "Assert >= {} 1.0 {}".format(_format_float(upper_bound), node_name)


def _bounds_lines(node_name: str, bounds: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if "Lower" in bounds:
        lines.append(_assert_lower_bound(node_name, bounds["Lower"]))
    if "Upper" in bounds:
        lines.append(_assert_upper_bound(node_name, bounds["Upper"]))
    return lines


def _input_constraint_lines(network: Any, test_property: dict[str, Any]) -> list[str]:
    bounds_by_index = {
        int(var_index): bounds
        for var_index, bounds in test_property.get("input", [])
    }
    lines: list[str] = []
    for input_index, node in enumerate(network.layers[0].nodes):
        bounds = bounds_by_index.get(
            input_index,
            {"Lower": -LARGE_BOUND, "Upper": LARGE_BOUND},
        )
        lines.extend(_bounds_lines(str(node.name), bounds))
    return lines


def _output_constraint_clauses(
        network: Any,
        test_property: dict[str, Any],
) -> list[list[tuple[int, dict[str, Any]]]]:
    if test_property.get("_planet_direct_output_comparison"):
        threshold = float(test_property.get("_adversarial_threshold", 0.0))
        return [
            [
                (
                    comparison_index,
                    {
                        "Lower": threshold,
                        "PlanetNodeName": spec["name"],
                    },
                )
            ]
            for comparison_index, spec in enumerate(
                _batch_aligned_output_specs(network, test_property)
            )
        ]

    output_bounds = [
        (int(var_index), dict(bounds))
        for var_index, bounds in test_property.get("output", [])
    ]
    if (
            test_property.get("type") == "adversarial"
            and test_property.get("_adversarial_query_mode", "disjunction") == "disjunction"
    ):
        violation_operator = test_property.get("_adversarial_violation_operator", "ge")
        clauses: list[list[tuple[int, dict[str, Any]]]] = []
        for var_index, bounds in output_bounds:
            if violation_operator == "ge":
                clauses.append([(var_index, {"Lower": bounds.get("Lower", 0.0)})])
            elif violation_operator == "le":
                clauses.append([(var_index, {"Upper": bounds.get("Upper", 0.0)})])
            else:
                raise ValueError(
                    "Unsupported adversarial violation operator '{}'.".format(
                        violation_operator
                    )
                )
        if not clauses:
            raise ValueError("adversarial property must include at least one output disjunct")
        return clauses
    return [output_bounds]


def _output_constraint_lines(
        network: Any,
        output_clause: list[tuple[int, dict[str, Any]]],
) -> list[str]:
    output_nodes = network.layers[-1].nodes
    lines: list[str] = []
    for output_index, bounds in output_clause:
        node_name = bounds.get("PlanetNodeName")
        if node_name is None:
            try:
                node_name = output_nodes[int(output_index)].name
            except IndexError as exc:
                raise ValueError(
                    "Output property references index {}, but the network has {} outputs.".format(
                        output_index,
                        len(output_nodes),
                    )
                ) from exc
        solver_bounds = {
            key: value
            for key, value in bounds.items()
            if key in {"Lower", "Upper"}
        }
        lines.extend(_bounds_lines(str(node_name), solver_bounds))
    return lines


def _as_rlv_path(path: Path) -> Path:
    if path.suffix.lower() != ".rlv":
        return path.with_suffix(".rlv")
    return path


def _resolve_query_save_path(save_query_path: str | os.PathLike[str] | None) -> Path | None:
    if save_query_path is not None:
        path_text = str(save_query_path).strip()
    else:
        path_text = os.environ.get(QUERY_LOG_ENV_VAR, "").strip()
    if not path_text:
        return None
    return _as_rlv_path(Path(path_text).expanduser())


def write_planet_query_files(
        network: Any,
        test_property: dict[str, Any],
        save_query_path: str | os.PathLike[str],
) -> list[Path]:
    base_path = _as_rlv_path(Path(save_query_path).expanduser())
    base_path.parent.mkdir(parents=True, exist_ok=True)

    network_lines = _network_lines(network, test_property)
    input_lines = _input_constraint_lines(network, test_property)
    output_clauses = _output_constraint_clauses(network, test_property)
    query_paths: list[Path] = []
    for clause_index, output_clause in enumerate(output_clauses):
        if len(output_clauses) == 1:
            query_path = base_path
        else:
            query_path = base_path.with_name(
                "{}_clause_{}{}".format(base_path.stem, clause_index, base_path.suffix)
            )
        output_lines = _output_constraint_lines(network, output_clause)
        query_path.write_text(
            "\n".join(network_lines + input_lines + output_lines) + "\n",
            encoding="utf-8",
        )
        query_paths.append(query_path)
    return query_paths


def _parse_planet_status(output: str) -> str:
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if stripped == "SAT":
            return "SAT"
        if stripped == "UNSAT":
            return "UNSAT"
    lowered = output.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return "TIMEOUT"
    if "error:" in lowered:
        return "ERROR"
    return "UNKNOWN"


_VALUATION_RE = re.compile(r"^\s*-\s+(\S+):\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


def _parse_input_valuation(output: str, network: Any) -> dict[int, float]:
    valuation_by_name: dict[str, float] = {}
    for line in output.splitlines():
        match = _VALUATION_RE.match(line)
        if match is None:
            continue
        valuation_by_name[match.group(1)] = float(match.group(2))

    counterexample: dict[int, float] = {}
    for input_index, node in enumerate(network.layers[0].nodes):
        if node.name in valuation_by_name:
            counterexample[input_index] = valuation_by_name[node.name]
    return counterexample


def _run_planet_file(
        planet_bin: Path,
        query_path: Path,
        timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(planet_bin), str(query_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(float(timeout_seconds), 1.0e-6),
        )
        output = "{}\n{}".format(completed.stdout, completed.stderr)
        status = _parse_planet_status(output)
        if status == "UNKNOWN" and completed.returncode != 0:
            status = "ERROR"
        return {
            "query_file": str(query_path),
            "status": status,
            "returncode": completed.returncode,
            "runtime": time.perf_counter() - started,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "query_file": str(query_path),
            "status": "TIMEOUT",
            "returncode": None,
            "runtime": time.perf_counter() - started,
            "stdout": stdout,
            "stderr": stderr,
        }


def _normalize_solver_status(status: Any) -> str:
    text = str(status).strip().lower()
    if "timeout" in text:
        return "timeout"
    if "unsat" in text:
        return "unsat"
    if text == "sat" or "sat" in text:
        return "sat"
    if "error" in text:
        return "error"
    return "unknown"


def get_query(
        network: Any,
        test_property: dict[str, Any],
        verbose: bool = False,
        planet_timeout_seconds: int | None = None,
        save_query_path: str | os.PathLike[str] | None = None,
        planet_bin: str | os.PathLike[str] | None = None,
        marabou_timeout_seconds: int | None = None,
        overall_timeout_seconds: float | None = None,
) -> tuple[dict[int, float], dict[str, Any], str]:
    overall_started = time.perf_counter()
    timeout_seconds = (
        DEFAULT_PLANET_TIMEOUT_SECONDS
        if planet_timeout_seconds is None and marabou_timeout_seconds is None
        else int(
            planet_timeout_seconds
            if planet_timeout_seconds is not None
            else marabou_timeout_seconds
        )
    )
    if timeout_seconds <= 0:
        raise ValueError("Planet query timeout must be positive")
    if overall_timeout_seconds is not None:
        overall_timeout_seconds = float(overall_timeout_seconds)
        if overall_timeout_seconds <= 0:
            raise ValueError("Planet overall timeout must be positive")
        overall_deadline = overall_started + overall_timeout_seconds
    else:
        overall_deadline = None
    resolved_planet_bin = resolve_planet_binary(planet_bin)

    resolved_save_path = _resolve_query_save_path(save_query_path)
    with tempfile.TemporaryDirectory(prefix="planet_query_") as temp_dir:
        query_base_path = resolved_save_path or (Path(temp_dir) / "query.rlv")
        query_paths = write_planet_query_files(
            network=network,
            test_property=test_property,
            save_query_path=query_base_path,
        )

        query_results: list[dict[str, Any]] = []
        fallback_status: str | None = None
        fallback_priority = {"UNKNOWN": 1, "ERROR": 2, "TIMEOUT": 3}
        for query_path in query_paths:
            effective_timeout_seconds = float(timeout_seconds)
            if overall_deadline is not None:
                remaining_seconds = overall_deadline - time.perf_counter()
                if remaining_seconds <= 0:
                    fallback_status = "TIMEOUT"
                    break
                effective_timeout_seconds = min(
                    effective_timeout_seconds,
                    remaining_seconds,
                )
            result = _run_planet_file(
                planet_bin=resolved_planet_bin,
                query_path=query_path,
                timeout_seconds=effective_timeout_seconds,
            )
            query_results.append(result)
            output = "{}\n{}".format(result.get("stdout", ""), result.get("stderr", ""))
            if verbose:
                print(output)
            if result["status"] == "SAT":
                stats = {
                    "backend": "planet",
                    "planet_bin": str(resolved_planet_bin),
                    "timeout_seconds": timeout_seconds,
                    "overall_timeout_seconds": overall_timeout_seconds,
                    "overall_runtime": time.perf_counter() - overall_started,
                    "queries": query_results,
                }
                return _parse_input_valuation(output, network), stats, "SAT"
            if result["status"] in {"TIMEOUT", "ERROR", "UNKNOWN"}:
                if (
                        fallback_status is None
                        or fallback_priority[result["status"]] > fallback_priority[fallback_status]
                ):
                    fallback_status = result["status"]
                if result["status"] == "TIMEOUT" and overall_deadline is not None:
                    break

        stats = {
            "backend": "planet",
            "planet_bin": str(resolved_planet_bin),
            "timeout_seconds": timeout_seconds,
            "overall_timeout_seconds": overall_timeout_seconds,
            "overall_runtime": time.perf_counter() - overall_started,
            "queries": query_results,
        }
        if fallback_status is not None:
            return {}, stats, fallback_status
        return {}, stats, "UNSAT"


def verify_network_with_planet(
        network: Any,
        test_property: dict[str, Any],
        timeout: int | None = None,
        save_query_path: str | os.PathLike[str] | None = None,
        planet_bin: str | os.PathLike[str] | None = None,
        overall_timeout: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        values, stats, status = get_query(
            network=network,
            test_property=test_property,
            verbose=False,
            planet_timeout_seconds=timeout,
            overall_timeout_seconds=overall_timeout,
            save_query_path=save_query_path,
            planet_bin=planet_bin,
        )
    except Exception as exc:
        lowered = str(exc).lower()
        status = "timeout" if "timeout" in lowered or "timed out" in lowered else "error"
        return {
            "status": status,
            "counterexample": {},
            "planet_runtime": time.perf_counter() - started,
            "raw_solver_output": "{}: {}".format(type(exc).__name__, exc),
        }
    normalized_status = _normalize_solver_status(status)
    return {
        "status": normalized_status,
        "counterexample": values if normalized_status == "sat" else {},
        "planet_runtime": time.perf_counter() - started,
        "raw_solver_output": str(stats),
    }
