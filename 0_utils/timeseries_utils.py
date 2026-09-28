"""
timeseries_utils.py
-------------------
General-purpose preprocessing utilities for time series projects.

Use cases: forecasting, sequence classification, anomaly detection,
sensor / finance / demand / IoT data.

IMPORTANT: time series must never be split randomly.
Always split by time order to avoid leakage from future data.

Typical usage
-------------
from timeseries_utils import time_split, standardize_from_train, create_sliding_windows

X_train, X_val, X_test, y_train, y_val, y_test = time_split(X, y)
X_train, X_val, X_test, mean, std = standardize_from_train(X_train, X_val, X_test)
X_windows, y_windows = create_sliding_windows(data, window_size=24, horizon=1)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Splitting (time-order safe)
# ---------------------------------------------------------------------------

def time_split(
    X,
    y,
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> tuple:
    """Split data by time order: no shuffle, no leakage.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    n = len(X)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    return (
        X[:train_end], X[train_end:val_end], X[val_end:],
        y[:train_end], y[train_end:val_end], y[val_end:],
    )


def sort_by_time(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Sort dataframe by timestamp column and reset index."""
    return df.sort_values(time_col).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Scaling (train-only fit to prevent leakage)
# ---------------------------------------------------------------------------

def standardize_from_train(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple:
    """Zero-mean, unit-variance scaling using train statistics only.

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled, mean, std
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    return (
        (X_train - mean) / std,
        (X_val - mean) / std,
        (X_test - mean) / std,
        mean,
        std,
    )


def minmax_from_train(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple:
    """Min-max scaling to [0, 1] using train min/max only.

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled, min_val, max_val
    """
    min_val = X_train.min(axis=0)
    max_val = X_train.max(axis=0)
    denom = (max_val - min_val) + 1e-8

    return (
        (X_train - min_val) / denom,
        (X_val - min_val) / denom,
        (X_test - min_val) / denom,
        min_val,
        max_val,
    )


def inverse_standardize(X_scaled: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Reverse standardization to recover original scale."""
    return X_scaled * std + mean


def inverse_minmax(
    X_scaled: np.ndarray,
    min_val: np.ndarray,
    max_val: np.ndarray,
) -> np.ndarray:
    """Reverse min-max scaling to recover original scale."""
    return X_scaled * (max_val - min_val) + min_val


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int],
) -> pd.DataFrame:
    """Add lag columns: target_lag_1, target_lag_7, etc."""
    out = df.copy()
    for lag in lags:
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: list[int],
) -> pd.DataFrame:
    """Add rolling mean and rolling std columns."""
    out = df.copy()
    for w in windows:
        out[f"{target_col}_rollmean_{w}"] = out[target_col].rolling(w).mean()
        out[f"{target_col}_rollstd_{w}"] = out[target_col].rolling(w).std()
    return out


def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Extract hour, day-of-week, month, quarter from a datetime column."""
    out = df.copy()
    dt = pd.to_datetime(out[time_col])
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["quarter"] = dt.dt.quarter
    return out


def drop_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop NaN rows (needed after lag/rolling feature creation)."""
    return df.dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sliding windows (for LSTM / CNN-1D / sequence models)
# ---------------------------------------------------------------------------

def create_sliding_windows(
    data: np.ndarray,
    window_size: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a sequence into supervised (X, y) samples.

    Parameters
    ----------
    data        : shape (timesteps,) or (timesteps, features)
    window_size : number of past steps used as input
    horizon     : number of future steps to predict

    Returns
    -------
    X : shape (samples, window_size) or (samples, window_size, features)
    y : shape (samples,) if horizon=1, else (samples, horizon)
    """
    X, y = [], []
    n = len(data)

    for i in range(n - window_size - horizon + 1):
        X.append(data[i: i + window_size])
        y.append(data[i + window_size: i + window_size + horizon])

    X = np.array(X)
    y = np.array(y)

    if horizon == 1:
        y = y[:, 0]

    return X, y
