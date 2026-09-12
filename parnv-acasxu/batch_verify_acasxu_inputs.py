from __future__ import annotations

import argparse
import contextlib
import csv
import io
import multiprocessing as mp
import os
import time
from datetime import datetime
from pathlib import Path
from queue import Empty
from typing import Any

from core.configuration import consts
from core.import_marabou import dynamically_import_marabou
from core.cegar.raw_cegar import cegar_verify_with_marabou, direct_verify_with_marabou
from core.utils.verification_properties_utils import read_test_properties


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Verify every ACAS Xu .nnet model in a directory against every "
            "property in an ACAS Xu input file."
        )
    )
    parser.add_argument(
        "--networks-dir",
        type=Path,
        default=script_dir / "experiments" / "ACASXu" / "networks",
        help="Directory containing ACAS Xu .nnet files.",
    )
    parser.add_argument(
        "--property-file",
        type=Path,
        default=script_dir / "experiments" / "ACASXu" / "inputs" / "input_001.txt",
        help="ACAS Xu property file, for example inputs/input_002.txt.",
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=Path(consts.results_directory) / "raw_cegar_acasxu_batch_input_001",
        help="Directory for per-route artifacts when --write-detailed-results is enabled.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=script_dir / "parnv_raw_cegar_input_001_batch_results.csv",
        help="CSV summary path.",
    )
    parser.add_argument(
        "--mode",
        choices=["p", "par"],
        default="par",
        help=(
            "Verification route: p runs CROWN-pruned Marabou only; "
            "par runs CROWN-pruned Merge/CEGAR only."
        ),
    )
    parser.add_argument(
        "--marabou-timeout-seconds",
        type=int,
        default=1200,
        help="Timeout passed to Marabou for each query. The CROWN-pruning flow uses 1200 seconds.",
    )
    parser.add_argument(
        "--property-timeout-seconds",
        type=int,
        default=1200,
        help=(
            "Wall-clock timeout for each network/property task, including CROWN, "
            "pruning, Merge/CEGAR, and Marabou calls."
        ),
    )
    parser.add_argument(
        "--max-refinement-steps",
        type=int,
        default=None,
        help="Maximum number of Merge undo steps per raw CEGAR run.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-7,
        help="Numerical tolerance for counterexample checks in raw CEGAR.",
    )
    parser.add_argument(
        "--marabou-dir",
        type=Path,
        default=None,
        help=(
            "Path to the Marabou repository directory that contains maraboupy. "
            "Equivalent to setting NARV_MARABOU_DIR."
        ),
    )
    parser.add_argument(
        "--continue-on-environment-error",
        action="store_true",
        help=(
            "Continue writing one CSV row per task when the Marabou import preflight fails. "
            "By default the batch stops immediately so environment errors are not reported as UNKNOWN."
        ),
    )
    parser.add_argument(
        "--quiet-runs",
        action="store_true",
        help="Capture per-run stdout and keep the batch log compact.",
    )
    parser.add_argument(
        "--write-detailed-results",
        action="store_true",
        help=(
            "Write per-network/property JSON logs and intermediate artifacts under "
            "--results-directory. By default the batch only writes the summary CSV."
        ),
    )
    return parser.parse_args()


def property_sort_key(property_id: str) -> tuple[str, int]:
    prefix, _, suffix = str(property_id).rpartition("_")
    if suffix.isdigit():
        return prefix, int(suffix)
    return str(property_id), -1


def _query_result_from_exception(exc: Exception) -> str:
    error_text = str(exc).lower()
    if "timeout" in error_text or "timed out" in error_text:
        return "TIMEOUT"
    return "ERROR"


def _prepare_marabou_import(args: argparse.Namespace) -> None:
    if args.marabou_dir is not None:
        os.environ["NARV_MARABOU_DIR"] = args.marabou_dir.resolve().as_posix()
    dynamically_import_marabou(query_type="adversarial")


def query_result_to_verification_result(query_result: Any) -> str:
    normalized = str(query_result).strip().upper()
    if normalized == "UNSAT":
        return "VERIFIED"
    if normalized == "SAT":
        return "UNSAFE"
    if normalized == "TIMEOUT":
        return "TIMEOUT"
    if normalized == "ERROR":
        return "ERROR"
    return "UNKNOWN"


def _is_correct_verification_result(value: Any) -> bool:
    return str(value).strip().upper() in {"VERIFIED", "UNSAFE"}


def route_names_for_mode(mode: str) -> tuple[str, ...]:
    routes_by_mode = {
        "p": ("direct_pruned",),
        "par": ("merge_cegar",),
    }
    try:
        return routes_by_mode[mode]
    except KeyError as exc:
        raise ValueError("unsupported verification mode: {}".format(mode)) from exc


def _queue_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _queue_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_queue_safe(item) for item in value]
    return str(value)


def _put_worker_result(result_queue: Any, result: dict[str, Any]) -> None:
    try:
        result_queue.put(_queue_safe(result))
    except Exception:
        pass


def _run_direct_pruned_route_worker(payload: dict[str, Any], result_queue: Any) -> None:
    route_started = time.perf_counter()
    route_name = "direct_pruned"
    try:
        stdout_context = (
            contextlib.redirect_stdout(io.StringIO())
            if payload.get("quiet_runs")
            else contextlib.nullcontext()
        )
        with stdout_context:
            result_map = direct_verify_with_marabou(
                original_nnet_file=payload["network_path"],
                property_file=payload["test_property"],
                output_dir=payload["direct_results_directory"],
                marabou_timeout=payload["marabou_timeout_seconds"],
                property_id=str(payload["property_id"]),
                write_detailed_results=bool(payload.get("write_detailed_results")),
            )
        query_result = str(result_map.get("query_result", "UNKNOWN"))
        verification_result = str(
            result_map.get("result", query_result_to_verification_result(query_result))
        )
        _put_worker_result(result_queue, {
            "route": route_name,
            "status": "completed",
            "verification_result": verification_result,
            "query_result": query_result,
            "elapsed_seconds": time.perf_counter() - route_started,
        })
    except Exception as exc:
        _put_worker_result(result_queue, {
            "route": route_name,
            "status": "error",
            "verification_result": _query_result_from_exception(exc),
            "query_result": _query_result_from_exception(exc),
            "elapsed_seconds": time.perf_counter() - route_started,
        })


def _run_merge_cegar_route_worker(payload: dict[str, Any], result_queue: Any) -> None:
    route_started = time.perf_counter()
    route_name = "merge_cegar"
    try:
        stdout_context = (
            contextlib.redirect_stdout(io.StringIO())
            if payload.get("quiet_runs")
            else contextlib.nullcontext()
        )
        with stdout_context:
            result_map = cegar_verify_with_marabou(
                original_nnet_file=payload["network_path"],
                property_file=payload["test_property"],
                output_dir=payload["cegar_output_dir"],
                max_refinement_steps=payload["max_refinement_steps"],
                marabou_timeout=payload["marabou_timeout_seconds"],
                tolerance=payload["tolerance"],
                property_id=str(payload["property_id"]),
                write_detailed_results=bool(payload.get("write_detailed_results")),
            )
        query_result = str(result_map.get("query_result", "UNKNOWN"))
        verification_result = str(result_map.get("result", query_result_to_verification_result(query_result)))
        _put_worker_result(result_queue, {
            "route": route_name,
            "status": "completed",
            "verification_result": verification_result,
            "query_result": query_result,
            "elapsed_seconds": time.perf_counter() - route_started,
        })
    except Exception as exc:
        _put_worker_result(result_queue, {
            "route": route_name,
            "status": "error",
            "verification_result": _query_result_from_exception(exc),
            "query_result": _query_result_from_exception(exc),
            "elapsed_seconds": time.perf_counter() - route_started,
        })


def _terminate_process(process: mp.Process, grace_seconds: float = 3.0) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=grace_seconds)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=grace_seconds)


def run_property_verification(
        args: argparse.Namespace,
        network_path: Path,
        property_id: str,
        test_property: dict[str, Any],
        results_directory: Path,
) -> dict[str, Any]:
    property_started = time.perf_counter()
    selected_routes = route_names_for_mode(args.mode)
    direct_output_dir = (results_directory / "direct_pruned" / network_path.stem / str(property_id)).resolve()
    cegar_output_dir = (results_directory / "merge_cegar" / network_path.stem / str(property_id)).resolve()
    if args.write_detailed_results and "direct_pruned" in selected_routes:
        direct_output_dir.mkdir(parents=True, exist_ok=True)
    if args.write_detailed_results and "merge_cegar" in selected_routes:
        cegar_output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "network_path": network_path.as_posix(),
        "property_id": str(property_id),
        "test_property": test_property,
        "direct_results_directory": direct_output_dir.as_posix(),
        "cegar_output_dir": cegar_output_dir.as_posix(),
        "marabou_timeout_seconds": int(args.marabou_timeout_seconds),
        "max_refinement_steps": args.max_refinement_steps,
        "tolerance": float(args.tolerance),
        "quiet_runs": bool(args.quiet_runs),
        "write_detailed_results": bool(args.write_detailed_results),
    }

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    worker_targets = {
        "direct_pruned": _run_direct_pruned_route_worker,
        "merge_cegar": _run_merge_cegar_route_worker,
    }
    processes = {
        route_name: ctx.Process(
            target=worker_targets[route_name],
            args=(payload, result_queue),
            name="{}_{}_{}".format(route_name, network_path.stem, property_id),
        )
        for route_name in selected_routes
    }

    for process in processes.values():
        process.start()

    route_results: dict[str, dict[str, Any]] = {}
    winner: dict[str, Any] | None = None
    timed_out = False
    deadline = property_started + float(args.property_timeout_seconds)

    try:
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                timed_out = True
                break
            try:
                result = result_queue.get(timeout=min(0.5, remaining))
            except Empty:
                if all(not process.is_alive() for process in processes.values()):
                    break
                continue

            route_name = str(result.get("route", ""))
            if route_name:
                route_results[route_name] = result
            if _is_correct_verification_result(result.get("verification_result")):
                winner = result
                break
            if len(route_results) >= len(processes) and all(not process.is_alive() for process in processes.values()):
                break
    finally:
        for route_name, process in processes.items():
            if winner is not None and route_name == winner.get("route"):
                process.join(timeout=1.0)
                if process.is_alive():
                    _terminate_process(process)
            else:
                _terminate_process(process)
        try:
            result_queue.close()
        except Exception:
            pass

    elapsed = time.perf_counter() - property_started
    if winner is not None:
        verification_result = str(winner.get("verification_result", "UNKNOWN"))
        query_result = str(winner.get("query_result", "UNKNOWN"))
    elif timed_out:
        verification_result = "TIMEOUT"
        query_result = "TIMEOUT"
    elif len(selected_routes) == 1:
        selected_result = route_results.get(selected_routes[0])
        verification_result = str(selected_result.get("verification_result", "UNKNOWN")) if selected_result else "UNKNOWN"
        query_result = str(selected_result.get("query_result", "UNKNOWN")) if selected_result else "UNKNOWN"

    return {
        "verification_result": verification_result,
        "query_result": query_result,
        "elapsed_seconds": elapsed,
    }


def _query_result_for_csv(query_result: Any) -> str:
    text = str(query_result).strip()
    return text.lower() if text else "unknown"


def main() -> None:
    args = parse_args()
    networks_dir = args.networks_dir.resolve()
    property_file = args.property_file.resolve()
    results_directory = args.results_directory.resolve()
    output_csv = args.output_csv.resolve()

    if not networks_dir.exists():
        raise FileNotFoundError("networks directory not found: {}".format(networks_dir))
    if not property_file.exists():
        raise FileNotFoundError("property file not found: {}".format(property_file))

    networks = sorted(networks_dir.glob("*.nnet"), key=lambda path: path.name)
    if not networks:
        raise FileNotFoundError("no .nnet files found in {}".format(networks_dir))

    properties = read_test_properties(property_file)
    property_items = sorted(properties.items(), key=lambda item: property_sort_key(item[0]))
    if not property_items:
        raise ValueError("no properties found in {}".format(property_file))

    marabou_preflight_error = None
    try:
        _prepare_marabou_import(args)
    except ModuleNotFoundError as exc:
        marabou_preflight_error = exc
        if not args.continue_on_environment_error:
            raise RuntimeError(
                "Marabou import preflight failed before starting the batch. "
                "Fix the Marabou Python environment or pass --marabou-dir; "
                "otherwise every run would fail for the same reason."
            ) from exc

    if args.write_detailed_results:
        results_directory.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["net_name", "property_id", "query_result", "elapsed_seconds"]
    should_write_header = (not output_csv.exists()) or output_csv.stat().st_size == 0
    with output_csv.open("a", newline="", encoding="utf-8-sig" if should_write_header else "utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()

        total_runs = len(networks) * len(property_items)
        run_index = 0
        for network_path in networks:
            for property_id, test_property in property_items:
                run_index += 1
                started_at = datetime.now().isoformat(timespec="seconds")
                print(
                    "[{}/{}] {} {} started_at={}".format(
                        run_index,
                        total_runs,
                        network_path.name,
                        property_id,
                        started_at,
                    ),
                    flush=True,
                )
                start = time.perf_counter()
                error_type = ""
                error_message = ""
                query_result = "UNKNOWN"
                verification_result = "UNKNOWN"
                try:
                    if marabou_preflight_error is not None:
                        raise marabou_preflight_error
                    property_result = run_property_verification(
                        args=args,
                        network_path=network_path,
                        property_id=str(property_id),
                        test_property=test_property,
                        results_directory=results_directory,
                    )
                    query_result = property_result["query_result"]
                    verification_result = property_result["verification_result"]
                except Exception as exc:
                    error_type = type(exc).__name__
                    error_message = str(exc)
                    query_result = _query_result_from_exception(exc)
                    verification_result = query_result

                elapsed = time.perf_counter() - start
                writer.writerow(
                    {
                        "net_name": network_path.name,
                        "property_id": property_id,
                        "query_result": _query_result_for_csv(query_result),
                        "elapsed_seconds": "{:.6f}".format(elapsed),
                    }
                )
                csv_file.flush()
                print(
                    "[{}/{}] {} {} result={} elapsed={:.3f}s{}".format(
                        run_index,
                        total_runs,
                        network_path.name,
                        property_id,
                        verification_result,
                        elapsed,
                        "" if not error_type else " error={}: {}".format(error_type, error_message),
                    ),
                    flush=True,
                )

    print("done. csv saved to: {}".format(output_csv.as_posix()))


if __name__ == "__main__":
    main()
