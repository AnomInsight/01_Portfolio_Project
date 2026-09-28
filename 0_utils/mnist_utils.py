"""
mnist_utils.py
--------------
MNIST-specific data loading wrapper.
Preprocessing is delegated to image_utils.py.
"""

from pathlib import Path

import idx2numpy

from image_utils import prepare_tabular, prepare_cnn


SEED = 42


def _default_mnist_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "MNIST_data"


def load_mnist_raw(data_dir: str | Path | None = None):
    """Load raw MNIST IDX files from disk.

    Returns
    -------
    train_images, train_labels, test_images, test_labels
    """
    base = Path(data_dir) if data_dir is not None else _default_mnist_dir()
    train_images = idx2numpy.convert_from_file(str(base / "train-images.idx3-ubyte"))
    train_labels = idx2numpy.convert_from_file(str(base / "train-labels.idx1-ubyte"))
    test_images = idx2numpy.convert_from_file(str(base / "t10k-images.idx3-ubyte"))
    test_labels = idx2numpy.convert_from_file(str(base / "t10k-labels.idx1-ubyte"))
    return train_images, train_labels, test_images, test_labels


def prepare_flat_data(
    test_size: float = 0.2,
    random_state: int = SEED,
    data_dir: str | Path | None = None,
):
    """Load MNIST + run tabular pipeline (normalize -> flatten -> split).

    Returns
    -------
    X_train, X_val, y_train, y_val, X_test, test_labels
    """
    train_images, train_labels, test_images, test_labels = load_mnist_raw(data_dir)
    X_train, X_val, y_train, y_val, X_test = prepare_tabular(
        train_images, test_images, train_labels,
        val_size=test_size, random_state=random_state,
    )
    return X_train, X_val, y_train, y_val, X_test, test_labels


def prepare_cnn_data(
    test_size: float = 0.2,
    random_state: int = SEED,
    data_dir: str | Path | None = None,
):
    """Load MNIST + run CNN pipeline (normalize -> add_channel_dim -> split).

    Returns
    -------
    X_train, X_val, y_train, y_val, X_test, test_labels
    """
    train_images, train_labels, test_images, test_labels = load_mnist_raw(data_dir)
    X_train, X_val, y_train, y_val, X_test = prepare_cnn(
        train_images, test_images, train_labels,
        val_size=test_size, random_state=random_state,
    )
    return X_train, X_val, y_train, y_val, X_test, test_labels
