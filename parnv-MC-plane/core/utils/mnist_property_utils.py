import gzip
import os
import struct
from typing import Tuple

import numpy as np


_SPLIT_TO_FILES = {
    "train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
    "test": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
}


def _candidate_roots(dataset_root: str) -> list:
    normalized_root = os.path.abspath(os.path.expanduser(dataset_root))
    return [
        normalized_root,
        os.path.join(normalized_root, "MNIST"),
        os.path.join(normalized_root, "raw"),
        os.path.join(normalized_root, "MNIST", "raw"),
    ]


def _resolve_idx_path(dataset_root: str, base_filename: str) -> str:
    checked_paths = []
    for root in _candidate_roots(dataset_root):
        for suffix in ("", ".gz"):
            candidate = os.path.join(root, base_filename + suffix)
            checked_paths.append(candidate)
            if os.path.exists(candidate):
                return candidate
    raise FileNotFoundError(
        "Could not find MNIST file '{}'. Checked: {}".format(
            base_filename,
            ", ".join(checked_paths),
        )
    )


def _open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def _read_idx_images(path: str) -> np.ndarray:
    with _open_maybe_gzip(path) as f:
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Invalid MNIST images magic number {} in '{}'".format(magic, path))
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    images = raw.reshape(num_images, num_rows * num_cols).astype(np.float32) / 255.0
    return images


def _read_idx_labels(path: str) -> np.ndarray:
    with _open_maybe_gzip(path) as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Invalid MNIST labels magic number {} in '{}'".format(magic, path))
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    labels = raw.reshape(num_labels)
    return labels


def load_mnist_sample(dataset_root: str, dataset_split: str, sample_index: int) -> Tuple[np.ndarray, int]:
    if dataset_split not in _SPLIT_TO_FILES:
        raise ValueError("Unsupported dataset split '{}'. Expected 'train' or 'test'.".format(dataset_split))

    image_file, label_file = _SPLIT_TO_FILES[dataset_split]
    image_path = _resolve_idx_path(dataset_root, image_file)
    label_path = _resolve_idx_path(dataset_root, label_file)

    images = _read_idx_images(image_path)
    labels = _read_idx_labels(label_path)

    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            "MNIST images/labels length mismatch: {} vs {}".format(images.shape[0], labels.shape[0])
        )
    if sample_index < 0 or sample_index >= images.shape[0]:
        raise IndexError(
            "Sample index {} is out of range for split '{}' with {} samples".format(
                sample_index,
                dataset_split,
                images.shape[0],
            )
        )

    return images[sample_index], int(labels[sample_index])


def build_mnist_adversarial_property(
        sample: np.ndarray,
        label: int,
        epsilon: float,
        output_threshold: float = 2.220446049250313e-16,
) -> dict:
    if epsilon < 0:
        raise ValueError("verification epsilon must be non-negative")
    threshold = float(output_threshold)

    input_bounds = []
    for index, value in enumerate(sample.tolist()):
        lower = max(0.0, value - epsilon)
        upper = min(1.0, value + epsilon)
        input_bounds.append((index, {"Lower": lower, "Upper": upper}))

    # The existing adversarial reduction code identifies the target label as the
    # output with the smallest lower bound. Setting the correct label to -1 and
    # the rest to 0 preserves that convention while working with argmax logits.
    output_bounds = []
    for class_index in range(10):
        if class_index == label:
            output_bounds.append((class_index, {"Lower": -1.0, "Upper": 0.0}))
        else:
            output_bounds.append((class_index, {"Lower": threshold, "Upper": threshold}))

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


def build_mnist_property_id(dataset_split: str, sample_index: int, epsilon: float) -> str:
    epsilon_str = ("{:.6f}".format(epsilon)).rstrip("0").rstrip(".")
    epsilon_str = epsilon_str.replace(".", "p")
    return "mnist_{}_idx{}_eps{}".format(dataset_split, sample_index, epsilon_str)
