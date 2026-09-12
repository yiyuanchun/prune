from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Tuple

import numpy as np


_TRAIN_BATCH_NAMES = tuple("data_batch_{}".format(index) for index in range(1, 6))
_TEST_BATCH_NAMES = ("test_batch",)


def _candidate_batch_directories(dataset_root: str) -> list[Path]:
    normalized_root = Path(os.path.abspath(os.path.expanduser(dataset_root)))
    return [
        normalized_root,
        normalized_root / "cifar-10-batches-py",
        normalized_root / "CIFAR10" / "cifar-10-batches-py",
        normalized_root / "cifar10" / "cifar-10-batches-py",
    ]


def _resolve_batch_directory(dataset_root: str) -> Path:
    checked = []
    for directory in _candidate_batch_directories(dataset_root):
        checked.append(directory)
        if (directory / "data_batch_1").is_file() and (directory / "test_batch").is_file():
            return directory
    raise FileNotFoundError(
        "Could not find CIFAR-10 Python batch files. Expected data_batch_1 and "
        "test_batch under one of: {}".format(", ".join(str(path) for path in checked))
    )


def _payload_value(payload: dict[Any, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
        encoded_name = name.encode("utf-8")
        if encoded_name in payload:
            return payload[encoded_name]
    raise KeyError("CIFAR-10 batch payload is missing keys {}".format(names))


@lru_cache(maxsize=8)
def _load_batch(batch_path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(batch_path, "rb") as batch_file:
        payload = pickle.load(batch_file, encoding="latin1")

    data = np.asarray(_payload_value(payload, "data"))
    labels = np.asarray(_payload_value(payload, "labels", "fine_labels"), dtype=np.int64)
    if data.ndim != 2 or data.shape[1] != 3 * 32 * 32:
        raise ValueError(
            "CIFAR-10 batch '{}' has data shape {}, expected (N, 3072).".format(
                batch_path,
                data.shape,
            )
        )
    if data.shape[0] != labels.shape[0]:
        raise ValueError(
            "CIFAR-10 data/labels length mismatch in '{}': {} vs {}.".format(
                batch_path,
                data.shape[0],
                labels.shape[0],
            )
        )
    return data, labels


def load_cifar10_sample(
        dataset_root: str,
        dataset_split: str,
        sample_index: int,
) -> Tuple[np.ndarray, int]:
    if dataset_split not in {"train", "test"}:
        raise ValueError(
            "Unsupported CIFAR-10 split '{}'. Expected 'train' or 'test'.".format(
                dataset_split
            )
        )
    if sample_index < 0:
        raise IndexError("CIFAR-10 sample index must be non-negative.")

    batch_directory = _resolve_batch_directory(dataset_root)
    batch_names = _TRAIN_BATCH_NAMES if dataset_split == "train" else _TEST_BATCH_NAMES
    remaining_index = int(sample_index)
    total_samples = 0
    for batch_name in batch_names:
        data, labels = _load_batch(str(batch_directory / batch_name))
        batch_size = int(data.shape[0])
        total_samples += batch_size
        if remaining_index < batch_size:
            sample = data[remaining_index].astype(np.float32) / 255.0
            return sample.reshape(-1), int(labels[remaining_index])
        remaining_index -= batch_size

    raise IndexError(
        "Sample index {} is out of range for CIFAR-10 split '{}' with {} samples.".format(
            sample_index,
            dataset_split,
            total_samples,
        )
    )


def build_cifar10_adversarial_property(
        sample: np.ndarray,
        label: int,
        epsilon: float,
        output_threshold: float = 2.220446049250313e-16,
) -> dict:
    sample_array = np.asarray(sample, dtype=np.float32).reshape(-1)
    if sample_array.size != 3 * 32 * 32:
        raise ValueError(
            "CIFAR-10 verification expects 3072 flattened CHW inputs, got {}.".format(
                sample_array.size
            )
        )
    if label < 0 or label >= 10:
        raise ValueError("CIFAR-10 label must be in [0, 9], got {}.".format(label))
    if epsilon < 0:
        raise ValueError("verification epsilon must be non-negative")

    threshold = float(output_threshold)
    input_bounds = []
    for index, value in enumerate(sample_array.tolist()):
        lower = max(0.0, float(value) - float(epsilon))
        upper = min(1.0, float(value) + float(epsilon))
        input_bounds.append((index, {"Lower": lower, "Upper": upper}))

    output_bounds = []
    for class_index in range(10):
        if class_index == label:
            output_bounds.append((class_index, {"Lower": -1.0, "Upper": 0.0}))
        else:
            output_bounds.append(
                (class_index, {"Lower": threshold, "Upper": threshold})
            )

    return {
        "type": "adversarial",
        "_adversarial_output_order": "higher_is_better",
        "_adversarial_query_mode": "disjunction",
        "_adversarial_violation_operator": "ge",
        "_adversarial_threshold": threshold,
        "_adversarial_reference_label": int(label),
        "input": input_bounds,
        "output": output_bounds,
    }


def build_cifar10_property_id(
        dataset_split: str,
        sample_index: int,
        epsilon: float,
) -> str:
    epsilon_str = ("{:.6f}".format(epsilon)).rstrip("0").rstrip(".")
    epsilon_str = epsilon_str.replace(".", "p")
    return "cifar10_{}_idx{}_eps{}".format(dataset_split, sample_index, epsilon_str)
