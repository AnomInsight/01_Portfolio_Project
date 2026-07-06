# %%
# Part A: Load Data
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import idx2numpy
from sklearn.linear_model import LogisticRegression
import time


# Loading the data
train_images = idx2numpy.convert_from_file('../data/MNIST_data/train-images.idx3-ubyte')
train_labels = idx2numpy.convert_from_file('../data/MNIST_data/train-labels.idx1-ubyte')

test_images = idx2numpy.convert_from_file('../data/MNIST_data/t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('../data/MNIST_data/t10k-labels.idx1-ubyte')

print(f"Train images shape: {train_images.shape}, labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}, labels shape: {test_labels.shape}")

# %%
# Part B: Flatten images
n_train, h, w = train_images.shape
n_test, _, _ = test_images.shape

X_train_flat = train_images.reshape(n_train, h * w)
X_test_flat = test_images.reshape(n_test, h * w)

print(f"Flattened train shape: {X_train_flat.shape}")
print(f"Flattened test shape: {X_test_flat.shape}")

# %%
# Part C: Split training into train/val with stratification
X_train, X_val, y_train, y_val = train_test_split(
    X_train_flat, 
    train_labels, 
    test_size=0.2, 
    stratify=train_labels, 
    random_state=42)

X_train = X_train / 255.0
X_val = X_val / 255.0
X_test_flat = X_test_flat / 255.0

print(f"Train: {X_train.shape}, Val: {X_val.shape}")
print(f"Train classes: {np.bincount(y_train)}")
print(f"Val classes: {np.bincount(y_val)}")

# %%
# Part D: Train Logistic Regression
log_reg = LogisticRegression(C=1.0, solver='lbfgs', max_iter=300, random_state=42)

start_time = time.time()
log_reg.fit(X_train, y_train)
end_time = time.time()
training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} seconds")

# %%
# Evaluate on validation set
val_predictions = log_reg.predict(X_val)

# Compare with train and test set performance
train_predictions = log_reg.predict(X_train)
test_predictions = log_reg.predict(X_test_flat)

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

# Accuracy
acc_val = accuracy_score(y_val, val_predictions)
acc_train = accuracy_score(y_train, train_predictions)
acc_test = accuracy_score(test_labels, test_predictions)
print(f"Logistic Regression Validation Accuracy: {acc_val:.4f}")
print(f"Logistic Regression Train Accuracy: {acc_train:.4f}")
print(f"Logistic Regression Test Accuracy: {acc_test:.4f}")

# Classification Report
from sklearn.metrics import classification_report

print(classification_report(y_val, val_predictions))

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, val_predictions)
print("Confusion Matrix:")
print(conf_matrix)

# %%

# Coefficients
# Extract coefficients for each class
coef = log_reg.coef_  # Shape: (10, 784)

# # Reshape to see pixel importance for class 0
# class_0_weights = coef[2].reshape(28, 28)

# # Visualize which pixels matter for digit 0
# plt.figure(figsize=(8, 8))
# plt.imshow(class_0_weights, cmap='RdBu_r')
# plt.colorbar()
# plt.title("Pixel Importance for Digit 0")
# plt.show()

print(coef.shape)  # Should be (10, 784)

fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)

for digit, ax in enumerate(axes.flat):
    weights = coef[digit].reshape(28, 28)
    im = ax.imshow(weights, cmap="RdBu_r")
    ax.set_title(f"Digit {digit}")
    ax.axis("off")

fig.colorbar(im, ax=axes, shrink=0.85)
fig.suptitle("Pixel Importance per Digit")
plt.show()

# Recall
recall_val = recall_score(y_val, val_predictions, average='macro')
print(f"Logistic Regression Validation Recall: {recall_val:.4f}")
# %%

# Visualize
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
# %%
