import pickle

import numpy as np

from core.utils.cifar10_property_utils import (
    build_cifar10_adversarial_property,
    build_cifar10_property_id,
    load_cifar10_sample,
)


def _write_batch(path, values, labels):
    data = np.stack(
        [np.full((3 * 32 * 32,), value, dtype=np.uint8) for value in values],
        axis=0,
    )
    with path.open("wb") as batch_file:
        pickle.dump({b"data": data, b"labels": labels}, batch_file)


def test_load_cifar10_sample_from_python_batches(tmp_path):
    batch_dir = tmp_path / "cifar-10-batches-py"
    batch_dir.mkdir()
    _write_batch(batch_dir / "data_batch_1", [0, 255], [3, 7])
    for index in range(2, 6):
        _write_batch(batch_dir / "data_batch_{}".format(index), [0], [0])
    _write_batch(batch_dir / "test_batch", [128], [5])

    train_sample, train_label = load_cifar10_sample(str(tmp_path), "train", 1)
    test_sample, test_label = load_cifar10_sample(str(tmp_path), "test", 0)

    assert train_sample.shape == (3072,)
    assert np.allclose(train_sample, 1.0)
    assert train_label == 7
    assert np.allclose(test_sample, 128.0 / 255.0)
    assert test_label == 5


def test_build_cifar10_adversarial_property():
    sample = np.full((3072,), 0.5, dtype=np.float32)
    test_property = build_cifar10_adversarial_property(
        sample=sample,
        label=4,
        epsilon=0.02,
    )

    assert test_property["type"] == "adversarial"
    assert test_property["_adversarial_reference_label"] == 4
    assert len(test_property["input"]) == 3072
    assert len(test_property["output"]) == 10
    assert build_cifar10_property_id("test", 12, 0.02) == "cifar10_test_idx12_eps0p02"
