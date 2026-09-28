"""
eval_utils.py
-------------
General-purpose model evaluation utilities for classification and regression.
Works with any sklearn-compatible model output.

Typical usage
-------------
from eval_utils import classification_summary, regression_summary, plot_confusion_matrix

classification_summary(y_true, y_pred, y_proba)
regression_summary(y_true, y_pred)
plot_confusion_matrix(y_true, y_pred, save_path="figures/cm.png")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    digits: int = 4,
    label: str = "",
) -> pd.DataFrame:
    """Print and return per-class metrics as a DataFrame."""
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*40}")
    if label:
        print(f"  {label}")
    print(f"  Accuracy: {acc:.{digits}f}")
    print(classification_report(y_true, y_pred, digits=digits))

    report = classification_report(y_true, y_pred, output_dict=True)
    rows = []
    for k, v in report.items():
        if isinstance(v, dict):
            row = {"class": k}
            row.update(v)
            rows.append(row)
    return pd.DataFrame(rows)


def top_confusion_pairs(y_true: np.ndarray, y_pred: np.ndarray, n: int = 5) -> list[tuple]:
    """Return the top-n most confused (true, predicted, count) pairs."""
    cm = confusion_matrix(y_true, y_pred)
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)

    flat_idx = np.argsort(cm_off.ravel())[::-1]
    pairs = []
    for idx in flat_idx:
        t, p = np.unravel_index(idx, cm_off.shape)
        c = int(cm_off[t, p])
        if c == 0:
            break
        pairs.append((int(t), int(p), c))
        if len(pairs) == n:
            break
    return pairs


def overfitting_report(
    train_acc: float,
    val_acc: float,
    test_acc: float,
) -> None:
    """Print a simple overfitting summary."""
    print(f"\nTrain accuracy : {train_acc:.4f}")
    print(f"Val accuracy   : {val_acc:.4f}")
    print(f"Test accuracy  : {test_acc:.4f}")
    print(f"Train-val gap  : {train_acc - val_acc:+.4f}")
    print(f"Val-test gap   : {val_acc - test_acc:+.4f}")

    if train_acc - val_acc > 0.05:
        print("  Warning: possible overfitting (train-val gap > 5pp).")
    else:
        print("  Overfitting gap looks acceptable.")


def confidence_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_conf: np.ndarray,
    high_conf_threshold: float = 0.90,
) -> None:
    """Print mean confidence for correct and incorrect predictions."""
    is_correct = y_pred == y_true
    print(f"\nMean confidence (correct)  : {y_conf[is_correct].mean():.4f}")
    print(f"Mean confidence (incorrect): {y_conf[~is_correct].mean():.4f}")
    n_hce = int(np.sum((~is_correct) & (y_conf >= high_conf_threshold)))
    print(f"High-confidence errors (>= {high_conf_threshold}): {n_hce}")


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def regression_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> dict:
    """Print and return MAE, RMSE, R² for regression tasks."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{'='*40}")
    if label:
        print(f"  {label}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
) -> None:
    """Heatmap confusion matrix with optional save."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_confidence_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_conf: np.ndarray,
    title: str = "Confidence Distribution",
    save_path: str | None = None,
) -> None:
    """Overlay confidence histograms for correct vs incorrect predictions."""
    is_correct = y_pred == y_true
    plt.figure(figsize=(10, 5))
    sns.histplot(y_conf[is_correct], bins=20, stat="density",
                 color="green", alpha=0.5, label="Correct")
    sns.histplot(y_conf[~is_correct], bins=20, stat="density",
                 color="red", alpha=0.5, label="Incorrect")
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_learning_curves(
    history: dict,
    title: str = "Learning Curves",
    save_path: str | None = None,
) -> None:
    """Plot train/val loss and accuracy from a Keras history dict."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="Train Acc")
    axes[1].plot(history["val_accuracy"], label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str] | None = None,
    top_n: int = 20,
    title: str = "Feature Importances",
    save_path: str | None = None,
) -> None:
    """Horizontal bar chart of top-n feature importances."""
    idx = np.argsort(importances)[-top_n:][::-1]
    labels = [feature_names[i] for i in idx] if feature_names else [str(i) for i in idx]

    plt.figure(figsize=(8, 6))
    plt.barh(range(top_n), importances[idx], align="center")
    plt.yticks(range(top_n), labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
