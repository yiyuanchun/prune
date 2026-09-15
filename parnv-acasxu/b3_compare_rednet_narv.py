from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from core.cegar.raw_cegar import cegar_verify_with_marabou
from core.configuration import consts
from core.import_marabou import dynamically_import_marabou
from core.nnet.read_nnet import network_from_nnet_file
from core.pre_process.rednet_pipeline import prepare_rednet_nnet
from core.utils.verification_properties_utils import read_test_properties


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="B3 experiment: compare NARv-like Merge/CEGAR with REDNet + NARv on ACAS Xu."
    )
    parser.add_argument(
        "--networks-dir",
        type=Path,
        default=script_dir / "experiments" / "ACASXu" / "networks",
    )
    parser.add_argument(
        "--property-file",
        type=Path,
        default=script_dir / "experiments" / "ACASXu" / "inputs" / "input_001.txt",
    )
    parser.add_argument(
        "--route",
        choices=["narv", "rednet-narv"],
        required=True,
        help="Run exactly one route per process to avoid warm-cache bias between routes.",
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=Path(consts.results_directory) / "b3_rednet_narv_acasxu",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--marabou-timeout-seconds", type=int, default=3600)
    parser.add_argument("--max-refinement-steps", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--equivalence-samples", type=int, default=128)
    parser.add_argument("--equivalence-tolerance", type=float, default=1e-5)
    parser.add_argument("--stability-tolerance", type=float, default=1e-9)
    parser.add_argument("--activation-margin", type=float, default=1e-9)
    parser.add_argument("--marabou-dir", type=Path, default=None)
    parser.add_argument("--write-detailed-results", action="store_true")
    parser.add_argument("--network-glob", default="*.nnet")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _property_sort_key(property_id: str) -> tuple[str, int]:
    prefix, _, suffix = str(property_id).rpartition("_")
    return (prefix, int(suffix)) if suffix.isdigit() else (str(property_id), -1)


def _relu_count(network) -> int:
    return int(sum(len(layer.nodes) for layer in network.layers[1:-1] if layer.type_name == "hidden"))


def _prepare_marabou(args: argparse.Namespace) -> None:
    if args.marabou_dir is not None:
        os.environ["NARV_MARABOU_DIR"] = args.marabou_dir.resolve().as_posix()
    dynamically_import_marabou(query_type="adversarial")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _default_output_csv(args: argparse.Namespace) -> Path:
    route_tag = args.route.replace("-", "_")
    return Path(__file__).resolve().parent / "b3_{}_results.csv".format(route_tag)


def _run_one(
        args: argparse.Namespace,
        network_path: Path,
        property_id: str,
        test_property: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    original_network = network_from_nnet_file(network_path.as_posix())
    original_relu_count = _relu_count(original_network)
    run_dir = (
        args.results_directory.resolve()
        / args.route
        / network_path.stem
        / str(property_id)
    )
    if args.write_detailed_results or args.route == "rednet-narv":
        run_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_report: dict[str, Any] | None = None
    verification_input = network_path
    if args.route == "rednet-narv":
        rednet_path = run_dir / "rednet_exact.nnet"
        preprocessing_report = prepare_rednet_nnet(
            original_nnet_file=network_path,
            test_property=test_property,
            output_nnet_file=rednet_path,
            equivalence_samples=args.equivalence_samples,
            equivalence_tolerance=args.equivalence_tolerance,
            seed=0,
            stability_tolerance=args.stability_tolerance,
            activation_margin=args.activation_margin,
        )
        verification_input = rednet_path

    cegar_started = time.perf_counter()
    result = cegar_verify_with_marabou(
        original_nnet_file=verification_input,
        property_file=test_property,
        output_dir=run_dir / "cegar",
        max_refinement_steps=args.max_refinement_steps,
        marabou_timeout=args.marabou_timeout_seconds,
        tolerance=args.tolerance,
        property_id=str(property_id),
        write_detailed_results=args.write_detailed_results,
    )
    cegar_time = time.perf_counter() - cegar_started
    total_time = time.perf_counter() - started

    reduction = (preprocessing_report or {}).get("reduction", {})
    equivalence = (preprocessing_report or {}).get("equivalence", {})
    reduced_relu_count = int(reduction.get("reduced_relu_count", original_relu_count))
    return {
        "route": args.route,
        "net_name": network_path.name,
        "property_id": str(property_id),
        "query_result": str(result.get("query_result", "UNKNOWN")),
        "verification_result": str(result.get("result", "UNKNOWN")),
        "total_time_seconds": float(total_time),
        "cegar_time_seconds": float(cegar_time),
        "rednet_preprocessing_time_seconds": float(
            (preprocessing_report or {}).get("preprocessing_time_seconds", 0.0)
        ),
        "rednet_crown_time_seconds": float((preprocessing_report or {}).get("crown_time_seconds", 0.0)),
        "rednet_rewrite_time_seconds": float((preprocessing_report or {}).get("reduction_time_seconds", 0.0)),
        "original_relu_count": original_relu_count,
        "reduced_relu_count": reduced_relu_count,
        "relu_reduction_ratio": float(reduction.get("reduction_ratio", 1.0)),
        "stable_inactive_removed": int(reduction.get("total_inactive_removed", 0)),
        "stable_active_eliminated": int(reduction.get("total_active_eliminated", 0)),
        "reconstructed_units": int(reduction.get("total_reconstructed_units", 0)),
        "equivalence_passed": bool(equivalence.get("passed", args.route == "narv")),
        "equivalence_max_abs_error": float(equivalence.get("max_abs_error", 0.0)),
        "iterations": int(result.get("iterations", 0) or 0),
        "refinement_steps": int(result.get("total_refinement_steps", 0) or 0),
        "narv_internal_dead_relu_pruned": int(result.get("dead_relu_pruned", 0) or 0),
        "rednet_hidden_sizes": _json(reduction.get("reduced_hidden_sizes", [])),
    }


def main() -> None:
    args = parse_args()
    _prepare_marabou(args)
    networks_dir = args.networks_dir.resolve()
    property_file = args.property_file.resolve()
    args.results_directory = args.results_directory.resolve()
    output_csv = (args.output_csv or _default_output_csv(args)).resolve()

    networks = sorted(networks_dir.glob(args.network_glob), key=lambda path: path.name)
    if not networks:
        raise FileNotFoundError("No networks matched {} in {}".format(args.network_glob, networks_dir))
    properties = sorted(read_test_properties(property_file).items(), key=lambda item: _property_sort_key(item[0]))
    tasks = [(network, pid, prop) for network in networks for pid, prop in properties]
    if args.limit is not None:
        tasks = tasks[: max(0, int(args.limit))]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "route", "net_name", "property_id", "query_result", "verification_result",
        "total_time_seconds", "cegar_time_seconds", "rednet_preprocessing_time_seconds",
        "rednet_crown_time_seconds", "rednet_rewrite_time_seconds", "original_relu_count",
        "reduced_relu_count", "relu_reduction_ratio", "stable_inactive_removed",
        "stable_active_eliminated", "reconstructed_units", "equivalence_passed",
        "equivalence_max_abs_error", "iterations", "refinement_steps",
        "narv_internal_dead_relu_pruned", "rednet_hidden_sizes",
    ]
    with output_csv.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, (network_path, property_id, test_property) in enumerate(tasks, start=1):
            print("[{}/{}] {} {} {}".format(index, len(tasks), args.route, network_path.name, property_id), flush=True)
            try:
                row = _run_one(args, network_path, property_id, test_property)
            except Exception as exc:
                row = {key: "" for key in fieldnames}
                row.update({
                    "route": args.route,
                    "net_name": network_path.name,
                    "property_id": str(property_id),
                    "query_result": "ERROR",
                    "verification_result": "ERROR:{}:{}".format(type(exc).__name__, exc),
                })
            writer.writerow(row)
            csv_file.flush()
    print("done. csv saved to {}".format(output_csv), flush=True)


if __name__ == "__main__":
    main()
