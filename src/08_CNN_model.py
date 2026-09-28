# %%
# Imports
import os
import time

FIG_DIR = "../reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import matplotlib.pyplot as plt
from mnist_utils import prepare_cnn_data

# %%
# Part A-C: Load + split + normalize + reshape for CNN
X_train, X_val, y_train, y_val, X_test, test_labels = prepare_cnn_data(
    test_size=0.2,
    random_state=42,
    normalize=True,
)

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

# %%
# Predictions + classification report + confusion matrix
y_val_pred_probs = model.predict(X_val, verbose=0)
y_val_pred = np.argmax(y_val_pred_probs, axis=1)

print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))

# %%
# Plot training history
plt.figure(figsize=(12, 5))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/cnn_learning_curves.png", dpi=200, bbox_inches="tight")
plt.show()

# %%
# Build confusion matrix from validation predictions
cm_val = confusion_matrix(y_val, y_val_pred)

# Remove diagonal (correct predictions) so we only keep mistakes
cm_off = cm_val.copy()
np.fill_diagonal(cm_off, 0)

# Get top 3 confusion pairs
flat_idx = np.argsort(cm_off.ravel())[::-1]
top3 = []
for idx in flat_idx:
    t, p = np.unravel_index(idx, cm_off.shape)  # t=true, p=pred
    c = cm_off[t, p]
    if c == 0:
        break
    top3.append((t, p, int(c)))
    if len(top3) == 3:
        break

print("\nTop 3 confusion pairs (true -> predicted):")
for t, p, c in top3:
    print(f"{t} -> {p}: {c}")
# %%
