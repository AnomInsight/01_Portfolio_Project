"""
image_utils.py
--------------
General-purpose image preprocessing utilities for classification projects.
Works with any grayscale or RGB image dataset stored as NumPy arrays.

Typical usage
-------------
from image_utils import normalize, flatten, add_channel_dim, split

X_train, X_val, y_train, y_val = split(X, y)
X_train = normalize(X_train)
X_train = flatten(X_train)         # for tabular models (LR, SVM, RF)
X_train = add_channel_dim(X_train) # for CNNs (adds channel dimension)
"""

import numpy as np
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(images: np.ndarray, scale: float = 255.0) -> np.ndarray:
    """Scale pixel values to [0, 1]. Works on any shape."""
    return images.astype(np.float32) / scale


def standardize(images: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization computed per dataset.
    Fit on training data, then apply same transform to val/test."""
    mean = images.mean()
    std = images.std()
    return (images.astype(np.float32) - mean) / (std + 1e-8)


def standardize_per_channel(images: np.ndarray) -> np.ndarray:
    """Standardize per channel across the batch (N, H, W, C)."""
    out = images.astype(np.float32)
    for c in range(out.shape[-1]):
        ch = out[..., c]
        out[..., c] = (ch - ch.mean()) / (ch.std() + 1e-8)
    return out


# ---------------------------------------------------------------------------
# Shape transforms
# ---------------------------------------------------------------------------

def flatten(images: np.ndarray) -> np.ndarray:
    """Flatten spatial dims to a 1-D vector per sample.
    (N, H, W) or (N, H, W, C) -> (N, H*W*C)
    """
    return images.reshape(images.shape[0], -1)


def add_channel_dim(images: np.ndarray) -> np.ndarray:
    """Add trailing channel dim for CNNs.
    (N, H, W) -> (N, H, W, 1)
    No-op if channel dim already present.
    """
    if images.ndim == 3:
        return np.expand_dims(images, axis=-1)
    return images


def resize_batch(images: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize each image to (height, width) using nearest-neighbour.
    Requires scipy. Input: (N, H, W) or (N, H, W, C).
    """
    from scipy.ndimage import zoom

    def _resize_one(img):
        factors = (height / img.shape[0], width / img.shape[1])
        if img.ndim == 3:
            factors = factors + (1,)
        return zoom(img, factors, order=1)

    return np.stack([_resize_one(img) for img in images])


# ---------------------------------------------------------------------------
# Train / val / test splitting
# ---------------------------------------------------------------------------

def split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val split.

    Returns
    -------
    X_train, X_val, y_train, y_val
    """
    return train_test_split(
        X,
        y,
        test_size=val_size,
        stratify=y if stratify else None,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Ready-to-use pipelines
# ---------------------------------------------------------------------------

def prepare_tabular(
    train_images: np.ndarray,
    test_images: np.ndarray,
    train_labels: np.ndarray,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Full pipeline for flat/tabular models (LR, SVM, RF).

    Steps: normalize -> flatten -> split

    Returns
    -------
    X_train, X_val, y_train, y_val, X_test
    """
    X_train_all = normalize(flatten(train_images))
    X_test = normalize(flatten(test_images))
    X_train, X_val, y_train, y_val = split(X_train_all, train_labels, val_size, random_state)
    return X_train, X_val, y_train, y_val, X_test


def prepare_cnn(
    train_images: np.ndarray,
    test_images: np.ndarray,
    train_labels: np.ndarray,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Full pipeline for CNNs.

    Steps: normalize -> add_channel_dim -> split

    Returns
    -------
    X_train, X_val, y_train, y_val, X_test
    """
    X_train_all = add_channel_dim(normalize(train_images))
    X_test = add_channel_dim(normalize(test_images))
    X_train, X_val, y_train, y_val = split(X_train_all, train_labels, val_size, random_state)
    return X_train, X_val, y_train, y_val, X_test
