# %%
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mnist_utils import prepare_flat_data

# %%
# Part A-C: Load + flatten + split + normalize
X_train, X_val, y_train, y_val, X_test_flat, test_labels = prepare_flat_data(
    test_size=0.2,
    random_state=42,
    normalize=True,
)

# %%
def run_svm(
    X_train, y_train, 
    X_val, y_val, 
    X_test_flat, test_labels,
    kernel='linear', 
    C=1.0, 
    gamma='scale'
):
    """
    Train an SVM model with the specified kernel and regularization parameter C.
    Evaluate the model on the validation set and return the accuracy and classification report.
    """
    # Initialize the SVM model
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma)

    # Train the model
    start_time = time.perf_counter()
    svm_model.fit(X_train, y_train)
    training_time = time.perf_counter() - start_time
    
    # Make predictions
    pred_train = svm_model.predict(X_train)
    pred_val = svm_model.predict(X_val)
    pred_test = svm_model.predict(X_test_flat)

    # Calculate accuracy
    train_acc = accuracy_score(y_train, pred_train)
    val_acc = accuracy_score(y_val, pred_val)
    test_acc = accuracy_score(test_labels, pred_test)

    # Generate classification report
    class_report = classification_report(y_val, pred_val, output_dict=True)
    
    val_precision = class_report['weighted avg']['precision']
    val_recall = class_report['weighted avg']['recall']
    val_f1 = class_report['weighted avg']['f1-score']
    
    cm_val = confusion_matrix(y_val, pred_val)

    return {
        "model": svm_model,
        "kernel": kernel,
        "C": C,
        "gamma": gamma,
        "train_time": training_time,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "val_precision_weighted": val_precision,
        "val_recall_weighted": val_recall,
        "val_f1_weighted": val_f1,
        "val_confusion_matrix": cm_val,
    }

results = []

# %%
# Linear search
for C in [1, 10]:
    results.append(run_svm(X_train, y_train, X_val, y_val, X_test_flat, test_labels, kernel="linear", C=C))

# %%
# RBF search
for C in [10]:
    for gamma in ["scale", 0.01]:
        results.append(run_svm(X_train, y_train, X_val, y_val, X_test_flat, test_labels, kernel="rbf", C=C, gamma=gamma))

best = max(results, key=lambda r: r["val_acc"])

# %%
print("Best config:", best["kernel"], best["C"], best["gamma"])
print("Train/Val/Test:", best["train_acc"], best["val_acc"], best["test_acc"])
print("Training time:", best["train_time"])
print("Validation Precision (Weighted):", best["val_precision_weighted"])
print("Validation Recall (Weighted):", best["val_recall_weighted"])
print("Validation F1 (Weighted):", best["val_f1_weighted"])
print("Validation Confusion Matrix:\n", best["val_confusion_matrix"])

print("\nAll configs:")
for r in sorted(results, key=lambda x: x["val_acc"], reverse=True):
    print(
        f"kernel={r['kernel']:<6} C={r['C']:<4} gamma={str(r['gamma']):<6} | "
        f"train={r['train_acc']:.4f} val={r['val_acc']:.4f} test={r['test_acc']:.4f} | "
        f"time={r['train_time']:.2f}s"
    )
# %%
import matplotlib.pyplot as plt
import seaborn as sns

cm = best["val_confusion_matrix"]

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.title(f"SVM Confusion Matrix (Val) | kernel={best['kernel']}, C={best['C']}, gamma={best['gamma']}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# %%
# Top 3 confusion pairs (ignore diagonal)
cm_off = cm.copy()
np.fill_diagonal(cm_off, 0)

flat_idx = np.argsort(cm_off.ravel())[::-1]

top3 = []
for idx in flat_idx:
    true_label, pred_label = np.unravel_index(idx, cm_off.shape)
    count = cm_off[true_label, pred_label]
    if count == 0:
        break
    top3.append((true_label, pred_label, int(count)))
    if len(top3) == 3:
        break

print("\nTop 3 confusion pairs (true -> predicted):")
for t, p, c in top3:
    print(f"{t} -> {p}: {c}")
# %%
