# %%
import cProfile

# Baseline 1: Randomly select a label for each sample
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

def random_baseline(num_samples, num_classes):
    return np.random.randint(0, num_classes, size=num_samples)

# %%
# Baseline 2: Always predict the most frequent class
def most_frequent_baseline(labels):
    from collections import Counter
    most_common = Counter(labels).most_common(1)[0][0]
    return np.full(len(labels), most_common)

# %%
# Evaluate the performance of the baselines using accuracy

# Example true labels
true_labels = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
num_samples = len(true_labels)
num_classes = len(np.unique(true_labels))
labels = true_labels

# Random predictions
random_predictions = random_baseline(num_samples, num_classes)
print("Random predictions:", random_predictions)

# Accuracy
acc_random = accuracy_score(true_labels, random_predictions)
print(f"Random Classifier Accuracy: {acc_random:.4f}")

# Confusion matrix
cm_random = confusion_matrix(true_labels, random_predictions)
print("Confusion Matrix:\n", cm_random)

# Per-class recall
recall_random = recall_score(true_labels, random_predictions, average=None)
print("Per-class Recall:", recall_random)


# Most frequent predictions
most_frequent_predictions = most_frequent_baseline(labels)
print("Most frequent predictions:", most_frequent_predictions)

# Accuracy
acc_majority = accuracy_score(true_labels, most_frequent_predictions)
print(f"Most Frequent Classifier Accuracy: {acc_majority:.4f}")

# Confusion matrix
cm_majority = confusion_matrix(true_labels, most_frequent_predictions)
print("Confusion Matrix:\n", cm_majority)

# Per-class recall
recall_majority = recall_score(true_labels, most_frequent_predictions, average=None)
print("Per-class Recall:", recall_majority)

# performance of the baselines using cProfile
cProfile.run('random_baseline(num_samples, num_classes)')
cProfile.run('most_frequent_baseline(labels)')