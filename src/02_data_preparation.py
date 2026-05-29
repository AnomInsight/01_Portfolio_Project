# %%
# Part A: Load Data
import numpy as np
from sklearn.model_selection import train_test_split
import idx2numpy

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

print(f"Train: {X_train.shape}, Val: {X_val.shape}")
print(f"Train classes: {np.bincount(y_train)}")
print(f"Val classes: {np.bincount(y_val)}")
# %%
# Part D: Verify no overlap between splits

total_train_size = X_train.shape[0]
total_val_size = X_val.shape[0]
total_test_size = X_test_flat.shape[0]

print(f"Train size: {total_train_size}")
print(f"Val size: {total_val_size}")
print(f"Test size: {total_test_size}")

# Verify sizes match expected split (60k train → 48k train + 12k val, 10k test)
expected_train = int(0.8 * 60000)
expected_val = int(0.2 * 60000)
expected_test = 10000

assert total_train_size == expected_train, f"Expected {expected_train} train samples, got {total_train_size}"
assert total_val_size == expected_val, f"Expected {expected_val} val samples, got {total_val_size}"
assert total_test_size == expected_test, f"Expected {expected_test} test samples, got {total_test_size}"

print("All sizes correct!")

# %%
# Part E: Normalize pixel values to [0, 1]
X_train_norm = X_train / 255.0
X_val_norm = X_val / 255.0
X_test_norm = X_test_flat / 255.0

print(f"Pixel value range in train: [{X_train_norm.min()}, {X_train_norm.max()}]")
print(f"Pixel value range in val: [{X_val_norm.min()}, {X_val_norm.max()}]")
print(f"Pixel value range in test: [{X_test_norm.min()}, {X_test_norm.max()}]")
# # %%
# Standardization (optional, but often beneficial for certain models)
# # Compute mean/std from training data
# mean = X_train_norm.mean(axis=0)
# std = X_train_norm.std(axis=0)

# # Apply transformation to all 3 splits
# X_train_final = (X_train_norm - mean) / std
# X_val_final = (X_val_norm - mean) / std
# X_test_final = (X_test_norm - mean) / std

# %%
# Part F: Data Cleaning Check
# Check for NaNs or Infs
print(f"NaNs in train: {np.isnan(X_train_norm).sum()}, Infs in train: {np.isinf(X_train_norm).sum()}")
print(f"NaNs in val: {np.isnan(X_val_norm).sum()}, Infs in val: {np.isinf(X_val_norm).sum()}")
print(f"NaNs in test: {np.isnan(X_test_norm).sum()}, Infs in test: {np.isinf(X_test_norm).sum()}")

# Check for corrupted/empty images
empty_train = np.sum(X_train_norm.sum(axis=1) == 0)
empty_val = np.sum(X_val_norm.sum(axis=1) == 0)
empty_test = np.sum(X_test_norm.sum(axis=1) == 0)

print(f"Empty images in train: {empty_train}")
print(f"Empty images in val: {empty_val}")
print(f"Empty images in test: {empty_test}")

# Check for duplicates
# Convert rows to a view and check unique
X_train_view = np.ascontiguousarray(X_train_norm).view(np.dtype((np.void, X_train_norm.dtype.itemsize * X_train_norm.shape[1])))
unique_train, counts = np.unique(X_train_view, return_counts=True)
n_duplicates = np.sum(counts > 1)
print(f"Duplicate images in train: {n_duplicates}")

# No outlier detection needed here

# %%
# Part G: Save preprocessed data
np.save('../data/processed/X_train.npy', X_train_norm)
np.save('../data/processed/y_train.npy', y_train)
np.save('../data/processed/X_val.npy', X_val_norm)
np.save('../data/processed/y_val.npy', y_val)
np.save('../data/processed/X_test.npy', X_test_norm)
np.save('../data/processed/y_test.npy', test_labels)
# %%
# Part H: Document metadata ()