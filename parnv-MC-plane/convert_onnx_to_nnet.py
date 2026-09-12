from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from core.nnet.read_nnet import (
    _load_onnx_dense_layers,
    read_nnet_parameters,
    write_nnet_file,
)


DEFAULT_ONNX_PATH = Path(
    "/home/gpu/yyc_projects/Prune/data/models/cifar10/cifar10_fc_relu_100x6.onnx"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a sequential fully connected ReLU ONNX network to the .nnet format."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_ONNX_PATH,
        help="Input .onnx file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .nnet file. By default it is written beside the input file.",
    )
    parser.add_argument(
        "--input-min",
        type=float,
        default=0.0,
        help="Input minimum recorded in .nnet metadata. Default: 0.0.",
    )
    parser.add_argument(
        "--input-max",
        type=float,
        default=1.0,
        help="Input maximum recorded in .nnet metadata. Default: 1.0.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def _input_path_candidates(path: Path) -> Iterable[Path]:
    expanded = path.expanduser()
    yield expanded

    stripped_name = expanded.name.lstrip()
    if stripped_name != expanded.name:
        yield expanded.with_name(stripped_name)

    normalized_text = str(expanded).replace("/ ", "/").replace("\\ ", "\\")
    normalized = Path(normalized_text)
    if normalized != expanded:
        yield normalized


def resolve_input_path(path: Path) -> Path:
    checked = []
    for candidate in _input_path_candidates(path):
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.is_file():
            if candidate != path.expanduser().resolve():
                print("input path adjusted to: {}".format(candidate))
            return candidate

    raise FileNotFoundError(
        "ONNX file not found. Checked: {}".format(
            ", ".join(str(candidate) for candidate in checked)
        )
    )


def _validate_export(
        source_weights: list[np.ndarray],
        source_biases: list[np.ndarray],
        output_path: Path,
) -> dict:
    exported = read_nnet_parameters(str(output_path))
    exported_weights = exported["weights"]
    exported_biases = exported["biases"]

    if len(source_weights) != len(exported_weights):
        raise ValueError("Exported .nnet layer count does not match the ONNX network.")

    for layer_index, (source, converted) in enumerate(
            zip(source_weights, exported_weights)
    ):
        source_array = np.asarray(source, dtype=float)
        converted_array = np.asarray(converted, dtype=float)
        if source_array.shape != converted_array.shape:
            raise ValueError(
                "Layer {} weight shape changed from {} to {}.".format(
                    layer_index,
                    source_array.shape,
                    converted_array.shape,
                )
            )
        if not np.allclose(source_array, converted_array, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "Layer {} weights changed during ONNX-to-NNet conversion.".format(
                    layer_index
                )
            )

    for layer_index, (source, converted) in enumerate(
            zip(source_biases, exported_biases)
    ):
        source_array = np.asarray(source, dtype=float)
        converted_array = np.asarray(converted, dtype=float)
        if source_array.shape != converted_array.shape:
            raise ValueError(
                "Layer {} bias shape changed from {} to {}.".format(
                    layer_index,
                    source_array.shape,
                    converted_array.shape,
                )
            )
        if not np.allclose(source_array, converted_array, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "Layer {} biases changed during ONNX-to-NNet conversion.".format(
                    layer_index
                )
            )

    return exported


def convert_onnx_to_nnet(
        input_path: Path,
        output_path: Path | None = None,
        input_min: float = 0.0,
        input_max: float = 1.0,
        force: bool = False,
) -> Path:
    input_path = resolve_input_path(input_path)
    if input_path.suffix.lower() != ".onnx":
        raise ValueError("Input file must use the .onnx suffix: {}".format(input_path))
    if input_min > input_max:
        raise ValueError("--input-min cannot be greater than --input-max")

    output_path = (
        input_path.with_suffix(".nnet")
        if output_path is None
        else output_path.expanduser().resolve()
    )
    if output_path.suffix.lower() != ".nnet":
        raise ValueError("Output file must use the .nnet suffix: {}".format(output_path))
    if output_path.exists() and not force:
        raise FileExistsError(
            "Output already exists: {}. Pass --force to overwrite it.".format(output_path)
        )

    weights, biases = _load_onnx_dense_layers(str(input_path))
    input_size = int(np.asarray(weights[0]).shape[1])
    metadata = {
        "inputMinimums": [float(input_min)] * input_size,
        "inputMaximums": [float(input_max)] * input_size,
        "inputMeans": [0.0] * input_size,
        "inputRanges": [1.0] * input_size,
        "outputMean": 0.0,
        "outputRange": 1.0,
        "symmetric": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_nnet_file(
        str(output_path),
        weights=weights,
        biases=biases,
        metadata=metadata,
    )
    exported = _validate_export(weights, biases, output_path)

    print("ONNX file: {}".format(input_path))
    print("NNet file: {}".format(output_path))
    print("layer sizes: {}".format(exported["layerSizes"]))
    print("conversion and parameter validation passed")
    return output_path


def main() -> None:
    args = parse_args()
    convert_onnx_to_nnet(
        input_path=args.input,
        output_path=args.output,
        input_min=args.input_min,
        input_max=args.input_max,
        force=args.force,
    )


if __name__ == "__main__":
    main()
