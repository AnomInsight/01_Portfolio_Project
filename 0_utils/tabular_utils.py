"""
tabular_utils.py
----------------
General-purpose preprocessing utilities for tabular data projects.

Use cases: classification and regression on structured CSV/DataFrame data
(churn, credit scoring, fraud, house prices, etc.).

Typical usage
-------------
from tabular_utils import split, drop_high_nan, encode_categoricals, standardize_from_train

X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)
X_train, X_val, X_test = standardize_from_train(X_train, X_val, X_test)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split(
    X,
    y,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple:
    """Two-step stratified split into train / val / test.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    strat = y if stratify else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=strat, random_state=random_state,
    )
    val_frac = val_size / (1.0 - test_size)
    strat2 = y_train_val if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_frac,
        stratify=strat2, random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame showing missing counts and percentages per column."""
    total = df.isnull().sum()
    pct = (total / len(df) * 100).round(2)
    return pd.DataFrame({"missing": total, "pct": pct}).query("missing > 0").sort_values("pct", ascending=False)


def drop_high_nan(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns where more than `threshold` fraction of values are NaN."""
    mask = df.isnull().mean() < threshold
    dropped = df.columns[~mask].tolist()
    if dropped:
        print(f"Dropped columns (>{threshold*100:.0f}% NaN): {dropped}")
    return df.loc[:, mask]


def fill_numeric_median(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN in numeric columns with column median."""
    out = df.copy()
    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].fillna(out[num_cols].median())
    return out


def fill_categorical_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN in categorical/object columns with column mode."""
    out = df.copy()
    cat_cols = out.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        out[col] = out[col].fillna(out[col].mode()[0])
    return out


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode all object/category columns.

    Returns
    -------
    encoded_df, encoders  (keep encoders to inverse-transform later)
    """
    out = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in out.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        out[col] = le.fit_transform(out[col].astype(str))
        encoders[col] = le
    return out, encoders


def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One-hot encode specified columns, drop first to avoid dummy trap."""
    return pd.get_dummies(df, columns=columns, drop_first=True)


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


# ---------------------------------------------------------------------------
# Class balance
# ---------------------------------------------------------------------------

def class_distribution(y: np.ndarray | pd.Series) -> pd.DataFrame:
    """Return a count/percentage table of class distribution."""
    s = pd.Series(y).value_counts()
    return pd.DataFrame({"count": s, "pct": (s / s.sum() * 100).round(2)})


def check_class_imbalance(y: np.ndarray | pd.Series, threshold: float = 0.15) -> None:
    """Warn if minority class is below threshold fraction."""
    dist = class_distribution(y)
    minority_pct = dist["pct"].min()
    if minority_pct < threshold * 100:
        print(f"Warning: minority class is {minority_pct:.1f}% — consider oversampling or class weights.")
    else:
        print(f"Class balance looks acceptable (minority: {minority_pct:.1f}%).")
