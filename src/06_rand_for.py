# %%
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Part A: Load data
train_images = idx2numpy.convert_from_file("../data/MNIST_data/train-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file("../data/MNIST_data/train-labels.idx1-ubyte")
test_images = idx2numpy.convert_from_file("../data/MNIST_data/t10k-images.idx3-ubyte")
test_labels = idx2numpy.convert_from_file("../data/MNIST_data/t10k-labels.idx1-ubyte")

# Part B: Flatten
n_train, h, w = train_images.shape
n_test = test_images.shape[0]
X_train_flat = train_images.reshape(n_train, h * w)
X_test_flat = test_images.reshape(n_test, h * w)

# Part C: Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_flat,
    train_labels,
    test_size=0.2,
    stratify=train_labels,
    random_state=42
)

# %%
# # Try different hyperparameters
# configs = [
#     {"n_estimators": 50, "max_depth": 5, "min_samples_leaf": 1},
#     {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 5},
#     {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 1},
#     {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 10}
# ]

# for config in configs:
#     print(f"Training with config: {config}")
#     rf = RandomForestClassifier(
#         n_estimators=config["n_estimators"],
#         max_depth=config["max_depth"],
#         min_samples_leaf=config["min_samples_leaf"],
#         n_jobs=-1,
#         random_state=42
#     )
#     rf.fit(X_train, y_train)
#     pred_val = rf.predict(X_val)
#     acc_val = accuracy_score(y_val, pred_val)
#     print(f"Validation Accuracy: {acc_val:.4f}\n")

# %%
# Part D: Train Random Forest
rf = RandomForestClassifier(
    n_estimators=250,
    max_depth=25,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42
)

start_time = time.time()
rf.fit(X_train, y_train)
end_time = time.time()
training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} seconds")

# %%
# Part E: Evaluate
pred_train = rf.predict(X_train)
pred_test = rf.predict(X_test_flat)
pred_val = rf.predict(X_val)

def eval_split(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    print(f"{name} - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(classification_report(y_true, y_pred))
    
eval_split("Train", y_train, pred_train)
eval_split("Validation", y_val, pred_val)

# %%
# Accuracy
acc_train = accuracy_score(y_train, pred_train)
acc_test = accuracy_score(test_labels, pred_test)
acc_val = accuracy_score(y_val, pred_val)

print(f"Train Accuracy: {acc_train:.4f}")
print(f"Test Accuracy: {acc_test:.4f}")
print(f"Validation Accuracy: {acc_val:.4f}")

train_val_gap = acc_train - acc_val
val_test_gap = acc_val - acc_test
print(f"\nTrain/Val Accuracy Gap: {train_val_gap:.4f}")
print(f"Val/Test Accuracy Gap: {val_test_gap:.4f}")

# %%
# Feature importance
importances = rf.feature_importances_
top_indices = np.argsort(importances)[-20:][::-1]
print("Top 20 feature importances:", top_indices.tolist())
plt.figure(figsize=(8, 6))
plt.barh(range(20), importances[top_indices][:20], align='center')
plt.yticks(range(20), top_indices[:20])
plt.xlabel('Importance')
plt.ylabel('Feature Index')
plt.title('Top 20 Feature Importances')
plt.gca().invert_yaxis()  # highest importance at top
plt.show()

# %%
# Visualize the importance as an image
importance_img = importances.reshape(28, 28)
plt.figure(figsize=(6, 6))
plt.imshow(importance_img, cmap="hot")
plt.colorbar()
plt.title("Random Forest Pixel Importances")
plt.axis("off")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# %%
# Error Analysis

# Confusion matrix
cm = confusion_matrix(y_val, pred_val)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

cm_off =cm.copy()
np.fill_diagonal(cm_off, 0)

flat_sorted = np.argsort(cm_off.flatten())[::-1]
top_misclass = np.unravel_index(flat_sorted[:5], cm_off.shape)

print("Top 5 misclassifications (true, pred):")
for true, pred in zip(*top_misclass):
    print(f"True: {true}, Predicted: {pred}, Count: {cm[true, pred]}")

# %%
# 3) Directionality check (is confusion symmetric?)
print("\nDirectionality check:")
for t, p, c in zip(*top_misclass, cm_off.flatten()[np.argsort(cm_off.flatten())[::-1][:5]]):
    reverse = cm[p, t]
    print(f"{t}->{p}: {c} vs {p}->{t}: {reverse}")
    
# %%
# 4) Misclassified example images
mis_idx = np.where(pred_val != y_val)[0]
print("Total misclassified on val:", len(mis_idx))

# pick 10 random mistakes
rng = np.random.default_rng(42)
pick = rng.choice(mis_idx, size=min(10, len(mis_idx)), replace=False)

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for ax, idx in zip(axes.flat, pick):
    img = X_val[idx].reshape(28, 28)   # flattened -> 28x28
    ax.imshow(img, cmap="gray")
    ax.set_title(f"T:{y_val[idx]} P:{pred_val[idx]}")
    ax.axis("off")

plt.suptitle("Random Forest Misclassified Validation Examples")
plt.tight_layout()
plt.show()
# %%
