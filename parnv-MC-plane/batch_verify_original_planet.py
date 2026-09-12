"""Batch robustness verification of original MNIST/CIFAR-10 classifiers with Planet."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

from core.nnet.read_nnet import network_from_nnet_file
from core.utils.cifar10_property_utils import (
    build_cifar10_adversarial_property,
    build_cifar10_property_id,
    load_cifar10_sample,
)
from core.utils.mnist_property_utils import (
    build_mnist_adversarial_property,
    build_mnist_property_id,
    load_mnist_sample,
)
from core.utils.planet_query_utils import (
    prepare_planet_backend,
    verify_network_with_planet,
)


DEFAULT_DATASET_ROOT = Path("/home/gpu/yyc_project /data")
DEFAULT_NETWORK_DIRS = {
    "mnist": Path("/home/gpu/yyc_projects/Prune/data/models/mnist"),
    "cifar10": Path("/home/gpu/yyc_projects/Prune/data/models/cifar10"),
}
DEFAULT_OUTPUT_THRESHOLD = 2.220446049250313e-16
CSV_FIELDS = [
    "dataset",
    "network",
    "sample_index",
    "property_id",
    "true_label",
    "predicted_label",
    "correctly_classified",
    "status",
    "planet_status",
    "classification_seconds",
    "verification_seconds",
    "elapsed_seconds",
    "query_base_path",
    "error_type",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify unmodified MNIST or CIFAR-10 .nnet classifiers directly with Planet. "
            "Misclassified center images are skipped."
        )
    )
    parser.add_argument(
        "dataset",
        choices=["mnist", "cifar10"],
        help="Dataset/model family to verify.",
    )
    parser.add_argument(
        "--networks-dir",
        type=Path,
        default=None,
        help="Directory containing the original 10-output .nnet classifiers.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root containing MNIST IDX files or CIFAR-10 Python batch files.",
    )
    parser.add_argument(
        "--dataset-split",
        choices=["train", "test"],
        default="train",
        help="Dataset split to verify (default: train).",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=49)
    parser.add_argument(
        "--verification-epsilon",
        type=float,
        default=0.02,
        help="L-infinity input perturbation radius (default: 0.02).",
    )
    parser.add_argument(
        "--output-threshold",
        "--mnist-output-threshold",
        "--cifar10-output-threshold",
        dest="output_threshold",
        type=float,
        default=DEFAULT_OUTPUT_THRESHOLD,
        help=(
            "Wrong-logit minus correct-logit lower bound. This matches "
            "batch_verify_mnist_cifar10_inputs.py by default."
        ),
    )
    parser.add_argument(
        "--planet-bin",
        type=Path,
        default=None,
        help="Compiled Planet executable; PLANET_BIN is used when omitted.",
    )
    parser.add_argument(
        "--planet-timeout-seconds",
        "--property-timeout-seconds",
        dest="planet_timeout_seconds",
        type=int,
        default=3600,
        help=(
            "Wall-clock timeout for the complete property of one sample, "
            "shared by all Planet query branches (default: 3600)."
        ),
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=None,
        help="Directory in which generated .rlv queries are retained.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV result path.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing CSV with the same header.",
    )
    return parser.parse_args()


def _float_tag(value: float) -> str:
    return "{:.12g}".format(float(value)).replace("-", "m").replace(".", "p")


def _resolve_defaults(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    if args.networks_dir is None:
        args.networks_dir = DEFAULT_NETWORK_DIRS[args.dataset]
    run_tag = "{}_{}_{}_{}_eps_{}".format(
        args.dataset,
        args.dataset_split,
        args.start_index,
        args.end_index,
        _float_tag(args.verification_epsilon),
    )
    if args.results_directory is None:
        args.results_directory = script_dir / "results" / "original_planet" / run_tag
    if args.output_csv is None:
        args.output_csv = script_dir / "original_planet_{}_results.csv".format(run_tag)


def _index_range(start_index: int, end_index: int) -> range:
    step = 1 if end_index >= start_index else -1
    return range(start_index, end_index + step, step)


def _load_sample(args: argparse.Namespace, sample_index: int) -> tuple[Any, int]:
    loader_args = {
        "dataset_root": args.dataset_root.as_posix(),
        "dataset_split": args.dataset_split,
        "sample_index": int(sample_index),
    }
    if args.dataset == "mnist":
        return load_mnist_sample(**loader_args)
    return load_cifar10_sample(**loader_args)


def _build_property(
        args: argparse.Namespace,
        sample: Any,
        label: int,
        sample_index: int,
) -> tuple[str, dict[str, Any]]:
    common_args = {
        "sample": sample,
        "label": int(label),
        "epsilon": float(args.verification_epsilon),
        "output_threshold": float(args.output_threshold),
    }
    if args.dataset == "mnist":
        property_id = build_mnist_property_id(
            args.dataset_split,
            sample_index,
            args.verification_epsilon,
        )
        test_property = build_mnist_adversarial_property(**common_args)
    else:
        property_id = build_cifar10_property_id(
            args.dataset_split,
            sample_index,
            args.verification_epsilon,
        )
        test_property = build_cifar10_adversarial_property(**common_args)

    # Each query is one wrong-vs-correct branch using the same threshold as the
    # preprocessed batch flow. The classifier itself is passed to Planet unchanged.
    test_property["_planet_direct_output_comparison"] = True
    return property_id, test_property


def _validate_network_dimensions(network: Any, dataset: str, network_path: Path) -> None:
    expected_inputs = 784 if dataset == "mnist" else 3072
    input_count = len(network.layers[0].nodes)
    output_count = len(network.layers[-1].nodes)
    if input_count != expected_inputs or output_count != 10:
        raise ValueError(
            "{} has input/output dimensions {}/{}, expected {}/10 for {}".format(
                network_path,
                input_count,
                output_count,
                expected_inputs,
                dataset,
            )
        )


def _predict(network: Any, sample: Any) -> int:
    flattened = sample.reshape(-1).tolist()
    if len(flattened) != len(network.layers[0].nodes):
        raise ValueError(
            "sample has {} inputs, but network expects {}".format(
                len(flattened),
                len(network.layers[0].nodes),
            )
        )
    output = network.speedy_evaluate(
        {index: float(value) for index, value in enumerate(flattened)}
    )
    return int(output.argmax())


def _status_from_planet(planet_status: str) -> str:
    return {
        "unsat": "verified",
        "sat": "unsafe",
        "timeout": "timeout",
        "error": "error",
    }.get(planet_status, "unknown")


def _format_seconds(value: float | None) -> str:
    return "" if value is None else "{:.6f}".format(value)


def _empty_row(dataset: str, network_name: str, sample_index: int) -> dict[str, Any]:
    return {
        field: ""
        for field in CSV_FIELDS
    } | {
        "dataset": dataset,
        "network": network_name,
        "sample_index": sample_index,
    }


def _open_csv(output_csv: Path, append: bool):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if append and output_csv.exists() and output_csv.stat().st_size > 0:
        with output_csv.open("r", newline="", encoding="utf-8-sig") as existing_file:
            existing_header = next(csv.reader(existing_file), [])
        if existing_header != CSV_FIELDS:
            raise ValueError(
                "existing CSV header does not match this script: {}".format(output_csv)
            )
        return output_csv.open("a", newline="", encoding="utf-8")
    return output_csv.open("w", newline="", encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    _resolve_defaults(args)

    if args.start_index < 0 or args.end_index < 0:
        raise ValueError("sample indices must be non-negative")
    if args.verification_epsilon < 0:
        raise ValueError("verification epsilon must be non-negative")
    if args.planet_timeout_seconds <= 0:
        raise ValueError("Planet timeout must be positive")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError("dataset root not found: {}".format(args.dataset_root))
    if not args.networks_dir.is_dir():
        raise FileNotFoundError("networks directory not found: {}".format(args.networks_dir))

    networks = sorted(args.networks_dir.glob("*.nnet"), key=lambda path: path.name)
    if not networks:
        raise FileNotFoundError("no .nnet files found in {}".format(args.networks_dir))

    planet_bin = prepare_planet_backend(args.planet_bin)
    args.results_directory.mkdir(parents=True, exist_ok=True)
    sample_indices = list(_index_range(args.start_index, args.end_index))
    samples = [
        (sample_index, *_load_sample(args, sample_index))
        for sample_index in sample_indices
    ]

    total_runs = len(networks) * len(samples)
    run_index = 0
    with _open_csv(args.output_csv, args.append) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if not args.append or csv_file.tell() == 0:
            writer.writeheader()

        for network_path in networks:
            network = None
            network_error: Exception | None = None
            try:
                network = network_from_nnet_file(network_path.as_posix())
                _validate_network_dimensions(network, args.dataset, network_path)
            except Exception as exc:
                network_error = exc

            for sample_index, sample, label in samples:
                run_index += 1
                total_started = time.perf_counter()
                row = _empty_row(args.dataset, network_path.name, sample_index)
                row["true_label"] = int(label)
                classification_seconds = None
                verification_seconds = None

                try:
                    if network_error is not None:
                        raise network_error

                    classification_started = time.perf_counter()
                    prediction = _predict(network, sample)
                    classification_seconds = time.perf_counter() - classification_started
                    row["predicted_label"] = prediction
                    row["correctly_classified"] = prediction == int(label)

                    property_id, test_property = _build_property(
                        args,
                        sample,
                        int(label),
                        sample_index,
                    )
                    row["property_id"] = property_id
                    if prediction != int(label):
                        row["status"] = "skipped_misclassified"
                        row["planet_status"] = "not_run"
                    else:
                        query_base_path = (
                            args.results_directory
                            / network_path.stem
                            / property_id
                            / "original_network.rlv"
                        )
                        row["query_base_path"] = query_base_path.as_posix()
                        verification_started = time.perf_counter()
                        result = verify_network_with_planet(
                            network=network,
                            test_property=test_property,
                            timeout=args.planet_timeout_seconds,
                            overall_timeout=args.planet_timeout_seconds,
                            save_query_path=query_base_path,
                            planet_bin=planet_bin,
                        )
                        verification_seconds = time.perf_counter() - verification_started
                        planet_status = str(result.get("status", "unknown")).lower()
                        row["planet_status"] = planet_status
                        row["status"] = _status_from_planet(planet_status)
                        if planet_status == "error":
                            row["error_type"] = "PlanetError"
                            row["error_message"] = result.get("raw_solver_output", "")
                except Exception as exc:
                    row["status"] = "error"
                    row["planet_status"] = "not_run"
                    row["error_type"] = type(exc).__name__
                    row["error_message"] = str(exc)

                elapsed_seconds = time.perf_counter() - total_started
                row["classification_seconds"] = _format_seconds(classification_seconds)
                row["verification_seconds"] = _format_seconds(verification_seconds)
                row["elapsed_seconds"] = _format_seconds(elapsed_seconds)
                writer.writerow(row)
                csv_file.flush()
                print(
                    "[{}/{}] network={} sample={} label={} prediction={} status={} elapsed={:.3f}s".format(
                        run_index,
                        total_runs,
                        network_path.name,
                        sample_index,
                        label,
                        row["predicted_label"],
                        row["status"],
                        elapsed_seconds,
                    ),
                    flush=True,
                )

    print("done. csv saved to: {}".format(args.output_csv.resolve().as_posix()))


if __name__ == "__main__":
    main()
