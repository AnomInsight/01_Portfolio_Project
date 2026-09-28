# %%
import time
import idx2numpy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Part A: Load data
train_images = idx2numpy.convert_from_file("../data/MNIST_data/train-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file("../data/MNIST_data/train-labels.idx1-ubyte")
test_images = idx2numpy.convert_from_file("../data/MNIST_data/t10k-images.idx3-ubyte")
test_labels = idx2numpy.convert_from_file("../data/MNIST_data/t10k-labels.idx1-ubyte")

# Keep raw split for both pipelines
X_train_img, X_val_img, y_train, y_val = train_test_split(
    train_images, 
    train_labels, 
    test_size=0.2, 
    stratify=train_labels, 
    random_state=42
)

# Flat features for RF/SVM
X_train_flat = X_train_img.reshape(len(X_train_img), -1).astype(np.float32) / 255.0
X_val_flat = X_val_img.reshape(len(X_val_img), -1).astype(np.float32) / 255.0
X_test_flat = test_images.reshape(len(test_images), -1).astype(np.float32) / 255.0

# CNN features
X_train_cnn = np.expand_dims(X_train_img.astype(np.float32) / 255.0, axis=-1)
X_val_cnn = np.expand_dims(X_val_img.astype(np.float32) / 255.0, axis=-1)
X_test_cnn = np.expand_dims(test_images.astype(np.float32) / 255.0, axis=-1)

# %%
# Ensemble method: Majority vote
def majority_vote(pred_matrix, n_classes=10):
    out = np.zeros(pred_matrix.shape[0], dtype=int)
    for i, row in enumerate(pred_matrix):
        out[i] = np.argmax(np.bincount(row, minlength=n_classes))
    return out

# %%
# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=250,
    max_depth=30,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train_flat, y_train)

rf_val_pred = rf.predict(X_val_flat)
rf_test_pred = rf.predict(X_test_flat)

# %%
# Train SVM
svm = SVC(kernel="rbf", C=10, gamma="scale")
svm.fit(X_train_flat, y_train)

svm_val_pred = svm.predict(X_val_flat)
svm_test_pred = svm.predict(X_test_flat)

# %%
# Train CNN
cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])


cnn.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

cnn.fit(
    X_train_cnn, y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=20,
    batch_size=128,
    callbacks=[early_stopping],
    verbose=1
)

cnn_val_pred = np.argmax(cnn.predict(X_val_cnn, verbose=0), axis=1)
cnn_test_pred = np.argmax(cnn.predict(X_test_cnn, verbose=0), axis=1)

# %%
# Ensemble voting
val_stack = np.stack([rf_val_pred, svm_val_pred, cnn_val_pred], axis=1)
test_stack = np.stack([rf_test_pred, svm_test_pred, cnn_test_pred], axis=1)

ens_val_pred = majority_vote(val_stack, n_classes=10)
ens_test_pred = majority_vote(test_stack, n_classes=10)

print("RF Test Accuracy:", accuracy_score(test_labels, rf_test_pred))
print("SVM Test Accuracy:", accuracy_score(test_labels, svm_test_pred))
print("CNN Test Accuracy:", accuracy_score(test_labels, cnn_test_pred))
print("Ensemble Val Accuracy:", accuracy_score(y_val, ens_val_pred))
print("Ensemble Test Accuracy:", accuracy_score(test_labels, ens_test_pred))

print("\nEnsemble Validation Classification Report:")
print(classification_report(y_val, ens_val_pred))

cm_ens = confusion_matrix(y_val, ens_val_pred)
print("\nEnsemble Validation Confusion Matrix:\n", cm_ens)

plt.figure(figsize=(10, 8))

sns.heatmap(cm_ens, annot=True, fmt="d", cmap="Blues")
plt.title("Ensemble Confusion Matrix (Validation)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()