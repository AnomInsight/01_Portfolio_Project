# %%
# Imports
import os
import time
import pandas as pd
import idx2numpy
import numpy as np
from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

FIG_DIR = "../reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# %%
# Part A: Load data
train_images = idx2numpy.convert_from_file("../data/MNIST_data/train-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file("../data/MNIST_data/train-labels.idx1-ubyte")
test_images = idx2numpy.convert_from_file("../data/MNIST_data/t10k-images.idx3-ubyte")
test_labels = idx2numpy.convert_from_file("../data/MNIST_data/t10k-labels.idx1-ubyte")

# %%
# Evaluation preprocessing

# Part C: Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    train_images,
    train_labels,
    test_size=0.2,
    stratify=train_labels,
    random_state=SEED
)

# Part C.1: Scale pixel values for CNN
X_train = X_train.astype(np.float32) / 255.0
X_val = X_val.astype(np.float32) / 255.0
X_test = test_images.astype(np.float32) / 255.0

X_train = np.expand_dims(X_train, axis=-1)  # (N, 28, 28, 1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print("X_train.shape:", X_train.shape)
print("X_val.shape:", X_val.shape)
print("X_test.shape:", X_test.shape)

# %%
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# %%
start_time = time.perf_counter()
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

training_time = time.perf_counter() - start_time

print(f"Training Time: {training_time:.2f} seconds")

# %%
# Final evaluation on train/val/test
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=0)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Train Loss: {train_loss:.4f}")
print(f"Validation Loss: {val_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Training Time: {training_time:.2f} seconds")

# Classification report and per-class summary
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
report = classification_report(test_labels, y_pred, output_dict=True)

print("\nClassification Report (Test):")
print(classification_report(test_labels, y_pred, digits=4))

# Confusion matrix
cm = confusion_matrix(test_labels, y_pred)
print("\nConfusion Matrix:\n", cm)

cm_off = cm.copy()
np.fill_diagonal(cm_off, 0)

flat_idx = np.argsort(cm_off.ravel())[::-1]
top5 = []
for idx in flat_idx:
    t, p = np.unravel_index(idx, cm_off.shape)
    c = cm_off[t, p]
    if c == 0:
        break
    top5.append((t, p, int(c)))
    if len(top5) == 5:
        break

print("\nTop 5 confusion pairs (true -> predicted):")
for t, p, c in top5:
    print(f"{t} -> {p}: {c}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("CNN Confusion Matrix (Test)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/confusion_matrix_test.png", dpi=200)
plt.show()

per_class_rows = []
for d in range(10):
    row = report[str(d)]
    per_class_rows.append({
        "digit": d,
        "precision": row["precision"],
        "recall": row["recall"],
        "f1": row["f1-score"],
        "support": int(row["support"])
    })

per_class_df = pd.DataFrame(per_class_rows)
print("\nPer-class Performance Table:")
print(per_class_df.to_string(index=False))

macro = report["macro avg"]
weighted = report["weighted avg"]

print("\nMacro Averages:")
print(f"precision={macro['precision']:.4f}, recall={macro['recall']:.4f}, f1={macro['f1-score']:.4f}")

print("\nWeighted Averages:")
print(f"precision={weighted['precision']:.4f}, recall={weighted['recall']:.4f}, f1={weighted['f1-score']:.4f}")

# %%
# Part C: Misclassification analysis with confidence

# Full probability outputs on test set
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_conf = np.max(y_pred_probs, axis=1)

# Part D: Confidence and calibration evidence
is_correct = (y_pred == test_labels)
correct_conf = y_conf[is_correct]
incorrect_conf = y_conf[~is_correct]

print("\nConfidence Summary:")
print(f"Correct predictions mean confidence: {correct_conf.mean():.4f}")
print(f"Incorrect predictions mean confidence: {incorrect_conf.mean():.4f}")

high_conf_threshold = 0.90
high_conf_errors = np.sum((~is_correct) & (y_conf >= high_conf_threshold))
print(f"High-confidence errors (conf >= {high_conf_threshold:.2f}): {high_conf_errors}")

plt.figure(figsize=(10, 5))
sns.histplot(correct_conf, bins=20, stat="density", color="green", alpha=0.5, label="Correct")
sns.histplot(incorrect_conf, bins=20, stat="density", color="red", alpha=0.5, label="Incorrect")
plt.title("Confidence Distribution: Correct vs Incorrect")
plt.xlabel("Predicted confidence")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/confidence_distribution.png", dpi=200)
plt.show()

# Indices of mistakes
mis_idx = np.where(y_pred != test_labels)[0]
print(f"\nTotal misclassified test samples: {len(mis_idx)}")

# Sample up to 24 mistakes for visualization
n_show = min(24, len(mis_idx))
sample_idx = mis_idx[:n_show]

fig, axes = plt.subplots(4, 6, figsize=(12, 8))
for ax, idx in zip(axes.flat, sample_idx):
    ax.imshow(X_test[idx].squeeze(), cmap="gray")
    ax.set_title(
        f"idx:{idx}\nT:{test_labels[idx]} P:{y_pred[idx]}\nConf:{y_conf[idx]:.2f}",
        fontsize=8
    )
    ax.axis("off")

# Hide any unused axes
for ax in axes.flat[n_show:]:
    ax.axis("off")

plt.suptitle("Misclassified Test Examples")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/misclassified_examples_grid.png", dpi=200)
plt.show()

# Build a review table
mis_rows = []
for idx in mis_idx:
    mis_rows.append({
        "index": int(idx),
        "true_label": int(test_labels[idx]),
        "pred_label": int(y_pred[idx]),
        "confidence": float(y_conf[idx]),
    })

mis_df = pd.DataFrame(mis_rows).sort_values("confidence", ascending=False)
mis_df["category"] = ""
mis_df["notes"] = ""
print("\nTop 20 highest-confidence errors:")
print(mis_df.head(20).to_string(index=False))
# %%
# Plot exactly the 20 highest-confidence errors you listed
top20_idx = [
    2654, 2035, 3520, 1014, 4176, 1393, 9729, 2939, 2597, 2135,
    2462, 3422, 340, 1878, 1530, 1226, 1247, 6560, 5937, 3808
]

# Build a small table for these indices
top20_rows = []
for idx in top20_idx:
    top20_rows.append({
        "index": int(idx),
        "true_label": int(test_labels[idx]),
        "pred_label": int(y_pred[idx]),
        "confidence": float(y_conf[idx]),
        "category": "",
        "notes": ""
    })
top20_df = pd.DataFrame(top20_rows).sort_values("confidence", ascending=False)

# Plot 4x5 grid
fig, axes = plt.subplots(4, 5, figsize=(14, 10))
for ax, (_, row) in zip(axes.flat, top20_df.iterrows()):
    idx = int(row["index"])
    ax.imshow(X_test[idx].squeeze(), cmap="gray")
    ax.set_title(
        f"idx:{idx} | T:{int(row['true_label'])} P:{int(row['pred_label'])}\nConf:{row['confidence']:.4f}",
        fontsize=9
    )
    ax.axis("off")

plt.suptitle("Top 20 Highest-Confidence Misclassifications", fontsize=14)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/top20_high_confidence_errors.png", dpi=200)
plt.show()

# Optional: save for your report/manual labeling
# fig.savefig("../reports/top20_high_conf_errors.png", dpi=200, bbox_inches="tight")

print("\nLabeling template:")
print(top20_df.to_string(index=False))
# %%
raw_labels = """
idx:2654 - model failure
idx:2035 - understandable confusion
idx:3520 - understandable confusion
idx:1014 - model failure
idx:4176 - understandable confusion
idx:1393 - understandable confusion
idx:9729 - understandable confusion
idx:2939 - model failure
idx:2597 - understandable confusion
idx:2135 - model failure
idx:2462 - understandable confusion
idx:3422 - understandable confusion
idx:340 - understandable confusion
idx:1878 - model failure
idx:1530 - understandable confusion
idx:1226 - understandable confusion
idx:1247 - model failure
idx:6560 - model failure
idx:5937 - understandable confusion
idx:3808 - model failure
"""

# Parse lines like: idx:2654 - model failure
look_decisions = {}
for line in raw_labels.strip().splitlines():
    line = line.strip()
    if not line or "-" not in line:
        continue
    left, right = line.split("-", 1)
    idx = int(left.replace("idx:", "").strip())
    cat = right.strip().lower()
    if cat in {"model failure", "understandable confusion"}:
        look_decisions[idx] = cat

# Apply category to top20 table
top20_df["category"] = top20_df["index"].map(look_decisions).fillna("")

# Optional auto-notes for labeled rows
top20_df["notes"] = np.where(
    top20_df["category"] != "",
    "Predicted " + top20_df["pred_label"].astype(str)
    + " instead of " + top20_df["true_label"].astype(str)
    + " | conf=" + top20_df["confidence"].round(4).astype(str),
    ""
)

# Sync labels back into mis_df
for _, r in top20_df.iterrows():
    if r["category"] != "":
        mis_df.loc[mis_df["index"] == int(r["index"]), ["category", "notes"]] = [r["category"], r["notes"]]

# Show unlabeled (if any)
unlabeled = top20_df[top20_df["category"] == ""]
print("\nUnlabeled top20 rows:")
print(unlabeled[["index", "true_label", "pred_label", "confidence"]].to_string(index=False))

# Labeled summary
labeled_top20 = top20_df[top20_df["category"] != ""].copy()
print("\nTop20 labeled rows:")
print(labeled_top20[["index", "true_label", "pred_label", "confidence", "category", "notes"]].to_string(index=False))

print("\nTop20 category counts:")
print(labeled_top20["category"].value_counts())

print("\nTop20 category percentages:")
print((labeled_top20["category"].value_counts(normalize=True) * 100).round(2))

# %%
# Part E: Robustness checks (optional)

def _rotate_batch(x, angle_deg):
    rotated = [rotate(img.squeeze(), angle=angle_deg, reshape=False, mode="nearest") for img in x]
    rotated = np.array(rotated, dtype=np.float32)
    return np.expand_dims(np.clip(rotated, 0.0, 1.0), axis=-1)


def _accuracy_on(data, labels):
    pred = np.argmax(model.predict(data, verbose=0), axis=1)
    return float(np.mean(pred == labels))


X_test_rot_p10 = _rotate_batch(X_test, angle_deg=10)
X_test_rot_m10 = _rotate_batch(X_test, angle_deg=-10)
X_test_noise = np.clip(X_test + np.random.normal(0, 0.12, X_test.shape).astype(np.float32), 0.0, 1.0)
X_test_dim = np.clip(X_test * 0.8, 0.0, 1.0)

robust_rows = [
    ("original", _accuracy_on(X_test, test_labels)),
    ("rotate +10deg", _accuracy_on(X_test_rot_p10, test_labels)),
    ("rotate -10deg", _accuracy_on(X_test_rot_m10, test_labels)),
    ("gaussian noise", _accuracy_on(X_test_noise, test_labels)),
    ("brightness x0.8", _accuracy_on(X_test_dim, test_labels)),
]

robust_df = pd.DataFrame(robust_rows, columns=["variant", "accuracy"])
print("\nRobustness Check Results:")
print(robust_df.to_string(index=False))

plt.figure(figsize=(8, 4))
sns.barplot(data=robust_df, x="variant", y="accuracy", color="#4c72b0")
plt.ylim(0.0, 1.0)
plt.title("Robustness Accuracy by Test Variant")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/robustness_variants_accuracy.png", dpi=200)
plt.show()
# %%
