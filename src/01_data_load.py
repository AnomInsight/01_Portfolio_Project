# %%
# Imports
from cProfile import label

import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
# %%
# Reading
images = idx2numpy.convert_from_file('../data/MNIST_data/train-images.idx3-ubyte')
labels = idx2numpy.convert_from_file('../data/MNIST_data/train-labels.idx1-ubyte')

# %%
print(images.shape)
print(labels.shape)

print(images.dtype)
print(images.min(), images.max())

print(np.unique(labels))
print(np.unique(labels, return_counts=True))

dig_count = zip(np.unique(labels), np.unique(labels, return_counts=True)[1])

for dig, count in dig_count:
    print(f'{dig}: {count}')

sorted_count = sorted(zip(np.unique(labels), np.unique(labels, return_counts=True)[1]), key=lambda x: x[1], reverse=True)

for dig, count in sorted_count:
    print(f'{dig}: {count}')

# Visualize some images
sorted_list = sorted(zip(np.unique(labels), np.unique(labels, return_counts=True)[1]), key=lambda x: x[1], reverse=True)
digits = [x[0] for x in sorted_list]
counts = [x[1] for x in sorted_list]
print(digits)
print(counts)

plot = plt.figure(figsize=(12, 4))
plt.bar([str(d) for d in digits], counts)
plt.title('MNIST Training Set Class Distribution')
plt.xlabel('Digits')
plt.ylabel('Counts')
plt.show()

# %%
np.random.seed(42)  # For reproducibility
plt.figure(figsize=(5, 20))
for i in range(10):
    rand = np.random.choice(np.where(labels == i)[0], 3, replace=False)
    for j in range(3):
        idx = rand[j]
        plt.subplot(10, 3, i*3 + j + 1)
        plt.imshow(images[idx], cmap='gray')
        plt.title(f'Label: {labels[idx]}')
        plt.axis('off')
plt.show()

np.random.seed(42)  # For reproducibility
plt.figure(figsize=(20, 5))
for j in range(3):
    for i in range(10):
        rand = np.random.choice(np.where(labels == i)[0], 3, replace=False)
        idx = rand[j]
        plt.subplot(3, 10, j*10 + i + 1)
        plt.imshow(images[idx], cmap='gray')
        plt.title(f'Label: {labels[idx]}')
        plt.axis('off')
plt.show()

# %%
# Flattening the images
n_samples, height, width = images.shape
X = images.reshape(n_samples, height * width)
print(X.shape)

q = images.flatten()
print(q.shape)


mean_pixel = X.mean()
avg_pixel = X.mean()
std_pixel = X.std()
print(f'Mean: {mean_pixel}, Average: {avg_pixel}, Standard Deviation: {std_pixel}')

# what % of pixels are pure black (0)?
sparsity = (X == 0).sum() / X.size * 100
print(f"Sparsity: {sparsity:.2f}%")
# %%
plt.hist(X.flatten(), bins=50)
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Pixel Intensity Distribution')
plt.show()
# %%
# Quality Check: Are there any corrupted images (e.g., all pixels are the same)?
# corrupted_images = np.where((X == X[:, 0][:, np.newaxis]).all(axis=1))[0]
# print(f'Number of corrupted images: {len(corrupted_images)}')

# All pixels are the same (either all 0s or all 255s)
boolean_check = (X == 0).all(axis=1)
indices = np.where(boolean_check)[0]
count = len(indices)

print(f'Number of corrupted images: {count}')

# Missing Data Check: Are there any missing labels?
missing_labels = np.where(labels == -1)[0]
print(f'Number of missing labels: {len(missing_labels)}')

# Check for duplicates
unique_images, counts = np.unique(X, axis=0, return_counts=True)
duplicate_indices = np.where(counts > 1)[0]
print(f'Number of duplicate images: {len(duplicate_indices)}')

# Size Check
print(images.shape)
print(labels.shape)

# Image darkness range check: Is there a wide range of pixel intensities, or are most images very dark or very bright?
darkness_range = X.mean(axis=1)
plt.hist(darkness_range, bins=50)
plt.xlabel('Average Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Image Darkness Range')
plt.show()


# Per-digit darkness range check
fig, ax = plt.subplots(figsize=(12, 6))
for digit in range(10):
    digit_images = X[labels == digit]
    digit_darkness = digit_images.mean(axis=1)
    ax.hist(digit_darkness, bins=50, alpha=0.5, label=f"Digit {digit}")
plt.legend()
plt.show()

# Compare 2 digits (e.g., 0 vs 1) to see if there are any noticeable differences in their pixel intensity distributions.
fig, ax = plt.subplots(figsize=(12, 6))
for digit in [0, 9]:
    digit_images = X[labels == digit]
    digit_darkness = digit_images.mean(axis=1)
    ax.hist(digit_darkness, bins=50, alpha=0.5, label=f"Digit {digit}")
plt.legend()
plt.show()
# %%
# for j in range(3):      # j = 0, 1, 2
#     for i in range(10):  # for each j, i = 0, 1, 2, ..., 9
#         print(j, i)
# %%
