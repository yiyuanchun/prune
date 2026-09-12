from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from core.configuration import consts
from core.import_marabou import dynamically_import_marabou
from core.cegar.raw_cegar import cegar_verify_with_marabou
from core.nnet.read_nnet import network_from_nnet_file
from core.utils.mnist_property_utils import (
    build_mnist_adversarial_property,
    build_mnist_property_id,
    load_mnist_sample,
)
from core.utils.cifar10_property_utils import (
    build_cifar10_adversarial_property,
    build_cifar10_property_id,
    load_cifar10_sample,
)
from parnv import extract_query_result, one_experiment


DEFAULT_MNIST_DATASET_ROOT = Path("/home/gpu/yyc_project /data")
DEFAULT_MNIST_NETWORKS_DIR = Path("/home/gpu/yyc_projects/Prune/data/models/mnist")
DEFAULT_CIFAR10_DATASET_ROOT = Path("/home/gpu/yyc_project /data")
DEFAULT_CIFAR10_NETWORKS_DIR = Path("/home/gpu/yyc_projects/Prune/data/models/cifar10")
DEFAULT_OUTPUT_THRESHOLD = 2.220446049250313e-16


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Batch-verify MNIST or CIFAR-10 .nnet models with the direct-pruned "
            "and Merge/CEGAR flows."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["mnist", "cifar10"],
        default="mnist",
        help="Dataset family to verify.",
    )
    parser.add_argument(
        "--networks-dir",
        type=Path,
        default=None,
        help="Directory containing 10-output classifier .nnet files.",
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=None,
        help="Directory where PARnv per-run result files are written.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV summary path.",
    )
    parser.add_argument(
        "--mode",
        choices=["p", "par"],
        default="par",
        help=(
            "Verification mode: p runs CROWN pruning followed by direct Marabou; "
            "par runs LP-Merge + CEGAR."
        ),
    )
    parser.add_argument(
        "-m",
        "--mechanism",
        choices=["marabou", "marabou_with_ar"],
        default="marabou_with_ar",
    )
    parser.add_argument(
        "-a",
        "--abstraction-type",
        choices=["naive", "alg2", "global", "kmeans"],
        default="global",
    )
    parser.add_argument(
        "-r",
        "--refinement-type",
        choices=["cegar", "weight_based", "global"],
        default="global",
    )
    parser.add_argument(
        "-as",
        "--abstraction-sequence",
        type=int,
        choices=[100, 250],
        default=100,
    )
    parser.add_argument(
        "-rs",
        "--refinement-sequence",
        type=int,
        choices=[50, 100],
        default=50,
    )
    parser.add_argument(
        "--marabou-timeout-seconds",
        type=int,
        default=None,
        help="Timeout passed to Marabou for each query. Defaults to 3600 seconds.",
    )
    parser.add_argument(
        "--property-timeout-seconds",
        type=int,
        default=None,
        help=(
            "Wall-clock timeout for each network/property task, including CROWN, "
            "pruning, Merge/CEGAR, and Marabou calls. Defaults to 3600 seconds."
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
        "--dataset-root",
        "--mnist-dataset-root",
        "--cifar10-dataset-root",
        dest="dataset_root",
        type=Path,
        default=None,
        help="Root containing MNIST IDX files or CIFAR-10 Python batch files.",
    )
    parser.add_argument(
        "--dataset-split",
        "--mnist-dataset-split",
        "--cifar10-dataset-split",
        dest="dataset_split",
        choices=["train", "test"],
        default="train",
        help="Dataset split used to generate robustness properties.",
    )
    parser.add_argument(
        "--start-index",
        "--mnist-start-index",
        "--cifar10-start-index",
        dest="start_index",
        type=int,
        default=0,
        help="First sample index to verify, inclusive.",
    )
    parser.add_argument(
        "--end-index",
        "--mnist-end-index",
        "--cifar10-end-index",
        dest="end_index",
        type=int,
        default=49,
        help="Last sample index to verify, inclusive.",
    )
    parser.add_argument(
        "--verification-epsilon",
        type=float,
        default=0.02,
        help="L-infinity perturbation radius for robustness properties.",
    )
    parser.add_argument(
        "--output-threshold",
        "--mnist-output-threshold",
        "--cifar10-output-threshold",
        dest="output_threshold",
        type=float,
        default=DEFAULT_OUTPUT_THRESHOLD,
        help="Threshold used for pairwise outputs: output[i] >= threshold.",
    )
    return parser.parse_args()


def normalize_abstraction_type(value: str) -> str:
    if value == "naive":
        return "complete"
    if value == "alg2":
        return "heuristic_alg2"
    return value


def normalize_refinement_type(value: str) -> str:
    if value == "weight_based":
        return "cetar"
    return value


def _float_tag(value: float) -> str:
    text = "{:.12g}".format(float(value))
    return text.replace("-", "m").replace(".", "p")


def _resolve_defaults(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    if args.dataset_root is None:
        args.dataset_root = (
            DEFAULT_MNIST_DATASET_ROOT
            if args.dataset == "mnist"
            else DEFAULT_CIFAR10_DATASET_ROOT
        )
    if args.networks_dir is None:
        args.networks_dir = (
            DEFAULT_MNIST_NETWORKS_DIR
            if args.dataset == "mnist"
            else DEFAULT_CIFAR10_NETWORKS_DIR
        )
    if args.results_directory is None:
        args.results_directory = (
            Path(consts.results_directory)
            / "raw_cegar_{}_{}_{}_{}_eps_{}".format(
                args.dataset,
                args.dataset_split,
                args.start_index,
                args.end_index,
                _float_tag(args.verification_epsilon),
            )
        )
    if args.output_csv is None:
        args.output_csv = script_dir / "parnv_{}_{}_{}_{}_eps_{}_batch_results.csv".format(
            args.dataset,
            args.dataset_split,
            args.start_index,
            args.end_index,
            _float_tag(args.verification_epsilon),
        )
    if args.marabou_timeout_seconds is None:
        args.marabou_timeout_seconds = 3600
    if args.property_timeout_seconds is None:
        args.property_timeout_seconds = 3600


def _index_range(start_index: int, end_index: int) -> range:
    if end_index >= start_index:
        return range(start_index, end_index + 1)
    return range(start_index, end_index - 1, -1)


def _build_property_items(args: argparse.Namespace) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    property_items: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for sample_index in _index_range(args.start_index, args.end_index):
        common_loader_args = {
            "dataset_root": args.dataset_root.as_posix(),
            "dataset_split": args.dataset_split,
            "sample_index": int(sample_index),
        }
        if args.dataset == "mnist":
            sample, label = load_mnist_sample(**common_loader_args)
            property_id = build_mnist_property_id(
                dataset_split=args.dataset_split,
                sample_index=int(sample_index),
                epsilon=float(args.verification_epsilon),
            )
            test_property = build_mnist_adversarial_property(
                sample=sample,
                label=int(label),
                epsilon=float(args.verification_epsilon),
                output_threshold=float(args.output_threshold),
            )
        else:
            sample, label = load_cifar10_sample(**common_loader_args)
            property_id = build_cifar10_property_id(
                dataset_split=args.dataset_split,
                sample_index=int(sample_index),
                epsilon=float(args.verification_epsilon),
            )
            test_property = build_cifar10_adversarial_property(
                sample=sample,
                label=int(label),
                epsilon=float(args.verification_epsilon),
                output_threshold=float(args.output_threshold),
            )

        sample_metadata = {
            "dataset": args.dataset,
            "sample": [float(value) for value in sample.tolist()],
            "label": int(label),
            "sample_index": int(sample_index),
            "dataset_split": args.dataset_split,
            "epsilon": float(args.verification_epsilon),
            "output_threshold": float(args.output_threshold),
            "property_id": property_id,
        }
        property_items.append((property_id, test_property, sample_metadata))
    return property_items


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


def verification_result_to_csv_status(verification_result: Any) -> str:
    normalized = str(verification_result).strip().upper()
    if normalized == "VERIFIED":
        return "unsat"
    if normalized == "UNSAFE":
        return "sat"
    if normalized == "TIMEOUT":
        return "timeout"
    if normalized == "SKIPPED":
        return "skipped"
    return str(verification_result)


def _predict_sample_label(network: Any, sample_metadata: dict[str, Any]) -> int:
    sample = sample_metadata["sample"]
    expected_input_size = len(network.layers[0].nodes)
    if len(sample) != expected_input_size:
        raise ValueError(
            "{} sample has {} inputs, but network expects {}.".format(
                sample_metadata.get("dataset", "image"),
                len(sample),
                expected_input_size,
            )
        )
    sample_dict = {index: float(value) for index, value in enumerate(sample)}
    output = network.speedy_evaluate(sample_dict)
    prediction = int(output.argmax())
    sample_metadata["center_prediction"] = prediction
    return prediction


def _classifies_sample_correctly(network: Any, sample_metadata: dict[str, Any]) -> bool:
    prediction = _predict_sample_label(network, sample_metadata)
    return prediction == int(sample_metadata["label"])


def _read_nnet_dimensions(network_path: Path) -> tuple[int, int]:
    with network_path.open("r", encoding="utf-8") as network_file:
        line = network_file.readline()
        while line and line.strip().startswith("//"):
            line = network_file.readline()
    if not line:
        raise ValueError("empty .nnet file: {}".format(network_path))
    values = [value.strip() for value in line.split(",") if value.strip()]
    if len(values) < 3:
        raise ValueError(
            "invalid .nnet header in '{}': expected numLayers,inputSize,outputSize".format(
                network_path
            )
        )
    return int(values[1]), int(values[2])


def _validate_network_dimensions(network_path: Path, dataset: str) -> None:
    expected_input_size = 784 if dataset == "mnist" else 3072
    input_size, output_size = _read_nnet_dimensions(network_path)
    if input_size != expected_input_size or output_size != 10:
        raise ValueError(
            "{} model '{}' has input/output dimensions {}/{}, expected {}/10. "
            "The verifier expects the original 10-logit classifier, not an already "
            "folded pairwise-output network.".format(
                dataset,
                network_path,
                input_size,
                output_size,
                expected_input_size,
            )
        )


def main() -> None:
    args = parse_args()
    _resolve_defaults(args)
    networks_dir = args.networks_dir.resolve()
    results_directory = args.results_directory.resolve()
    output_csv = args.output_csv.resolve()

    if not networks_dir.exists():
        raise FileNotFoundError("networks directory not found: {}".format(networks_dir))
    dataset_root = args.dataset_root
    if not dataset_root.exists():
        raise FileNotFoundError(
            "{} dataset root not found: {}".format(args.dataset, dataset_root)
        )

    networks = sorted(networks_dir.glob("*.nnet"), key=lambda path: path.name)
    if not networks:
        raise FileNotFoundError("no .nnet files found in {}".format(networks_dir))
    for network_path in networks:
        _validate_network_dimensions(network_path, args.dataset)

    property_items = _build_property_items(args)
    if not property_items:
        raise ValueError("no {} properties were generated".format(args.dataset))

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

    results_directory.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    abstraction_type = normalize_abstraction_type(args.abstraction_type)
    refinement_type = normalize_refinement_type(args.refinement_type)

    fieldnames = ["sample_index", "status", "elapsed_seconds"]
    should_write_header = True
    if output_csv.exists() and output_csv.stat().st_size > 0:
        with output_csv.open("r", newline="", encoding="utf-8-sig") as existing_csv_file:
            existing_header = next(csv.reader(existing_csv_file), [])
        if existing_header != fieldnames:
            raise ValueError(
                "output CSV '{}' uses header {}, expected {}. "
                "Use a new --output-csv path or clear the old file first.".format(
                    output_csv,
                    existing_header,
                    fieldnames,
                )
            )
        should_write_header = False
    with output_csv.open("a", newline="", encoding="utf-8-sig" if should_write_header else "utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()

        total_runs = len(networks) * len(property_items)
        run_index = 0
        for network_path in networks:
            classification_network = None
            classification_network_error = None
            try:
                classification_network = network_from_nnet_file(network_path.as_posix())
            except Exception as exc:
                classification_network_error = exc

            for property_id, test_property, sample_metadata in property_items:
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
                verification_result = "UNKNOWN"
                try:
                    if classification_network_error is not None:
                        raise classification_network_error
                    if not _classifies_sample_correctly(classification_network, sample_metadata):
                        verification_result = "SKIPPED"
                    elif marabou_preflight_error is not None:
                        raise marabou_preflight_error
                    else:
                        stdout_context = (
                            contextlib.redirect_stdout(io.StringIO())
                            if args.quiet_runs
                            else contextlib.nullcontext()
                        )
                        with stdout_context:
                            if args.mode == "par":
                                cegar_output_dir = (
                                    results_directory
                                    / "merge_cegar"
                                    / network_path.stem
                                    / str(property_id)
                                ).as_posix()
                                result_map = cegar_verify_with_marabou(
                                    original_nnet_file=network_path.as_posix(),
                                    property_file=test_property,
                                    output_dir=cegar_output_dir,
                                    max_refinement_steps=args.max_refinement_steps,
                                    marabou_timeout=args.marabou_timeout_seconds,
                                    tolerance=args.tolerance,
                                    property_id=str(property_id),
                                )
                                verification_result = str(result_map.get("result", "UNKNOWN"))
                            else:
                                direct_output_dir = (
                                    results_directory
                                    / "direct_pruned"
                                    / network_path.stem
                                    / str(property_id)
                                ).as_posix()
                                Path(direct_output_dir).mkdir(parents=True, exist_ok=True)
                                result = one_experiment(
                                    nnet_filename=network_path.name,
                                    property_id=str(property_id),
                                    mechanism=args.mechanism,
                                    refinement_type=refinement_type,
                                    abstraction_type=abstraction_type,
                                    refinement_sequence=args.refinement_sequence,
                                    abstraction_sequence=args.abstraction_sequence,
                                    results_directory=direct_output_dir,
                                    model_path=network_path.as_posix(),
                                    custom_test_property=test_property,
                                    sample_metadata=sample_metadata,
                                    run_args={
                                        **vars(args),
                                        "network_path": network_path.as_posix(),
                                        "property_id": str(property_id),
                                    },
                                    marabou_timeout_seconds=args.marabou_timeout_seconds,
                                )
                                query_result = extract_query_result(result)
                                verification_result = query_result_to_verification_result(query_result)
                except Exception as exc:
                    error_type = type(exc).__name__
                    error_message = str(exc)
                    verification_result = _query_result_from_exception(exc)

                elapsed = time.perf_counter() - start
                csv_status = verification_result_to_csv_status(verification_result)
                writer.writerow(
                    {
                        "sample_index": (
                            "" if sample_metadata is None else sample_metadata.get("sample_index", "")
                        ),
                        "status": csv_status,
                        "elapsed_seconds": "{:.6f}".format(elapsed),
                    }
                )
                csv_file.flush()
                print(
                    "[{}/{}] {} {} status={} elapsed={:.3f}s{}".format(
                        run_index,
                        total_runs,
                        network_path.name,
                        property_id,
                        csv_status,
                        elapsed,
                        "" if not error_type else " error={}: {}".format(error_type, error_message),
                    ),
                    flush=True,
                )

    print("done. csv saved to: {}".format(output_csv.as_posix()))


if __name__ == "__main__":
    main()
