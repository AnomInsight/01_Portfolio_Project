# %%
import cProfile

# Baseline 1: Randomly select a label for each sample
import numpy as np
from sklearn.metrics import accuracy_score

def random_baseline(num_samples, num_classes):
    return np.random.randint(0, num_classes, size=num_samples)

# Baseline 2: Always predict the most frequent class
def most_frequent_baseline(labels, num_predictions=None):
    from collections import Counter
    most_common = Counter(labels).most_common(1)[0][0]
    n = num_predictions if num_predictions is not None else len(labels)
    return np.full(n, most_common)

# Load processed data
y_train = np.load('data/processed/y_train.npy')
y_test = np.load('data/processed/y_test.npy')
y_val = np.load('data/processed/y_val.npy')

num_classes = len(np.unique(y_train))

# Evaluate baselines on training set
print("=" * 50)
print("TRAINING SET")
print("=" * 50)
random_preds_train = random_baseline(len(y_train), num_classes)
most_freq_preds_train = most_frequent_baseline(y_train)

random_acc_train = accuracy_score(y_train, random_preds_train)
most_freq_acc_train = accuracy_score(y_train, most_freq_preds_train)

print(f"Random baseline accuracy: {random_acc_train:.4f}")
print(f"Most frequent baseline accuracy: {most_freq_acc_train:.4f}")

# Evaluate baselines on validation set
print("\n" + "=" * 50)
print("VALIDATION SET")
print("=" * 50)
random_preds_val = random_baseline(len(y_val), num_classes)
most_freq_preds_val = most_frequent_baseline(y_train, len(y_val))

random_acc_val = accuracy_score(y_val, random_preds_val)
most_freq_acc_val = accuracy_score(y_val, most_freq_preds_val)

print(f"Random baseline accuracy: {random_acc_val:.4f}")
print(f"Most frequent baseline accuracy: {most_freq_acc_val:.4f}")

# Evaluate baselines on test set
print("\n" + "=" * 50)
print("TEST SET")
print("=" * 50)
random_preds_test = random_baseline(len(y_test), num_classes)
most_freq_preds_test = most_frequent_baseline(y_train, len(y_test))

random_acc_test = accuracy_score(y_test, random_preds_test)
most_freq_acc_test = accuracy_score(y_test, most_freq_preds_test)

print(f"Random baseline accuracy: {random_acc_test:.4f}")
print(f"Most frequent baseline accuracy: {most_freq_acc_test:.4f}")

# Summary
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Most Frequent Baseline - Train: {most_freq_acc_train:.4f}, Val: {most_freq_acc_val:.4f}, Test: {most_freq_acc_test:.4f}")