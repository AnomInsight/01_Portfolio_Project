# 03 Data Preparation

## Objective
Transform raw data into clean, normalized inputs ready for modeling. Define and document your preprocessing pipeline.

**Key principle:** All preprocessing should be fit on **training data only**, then applied consistently to validation and test data. This prevents data leakage.

---

## Part A: Train/Validation/Test Split

### Task 1: Split the Data
Questions to answer:
1. What's a reasonable split? (e.g., 70% train, 15% val, 15% test? or 60/20/20?)
2. Should you stratify by class to maintain balance?
3. How will you ensure no data leaks between splits?
4. Will you use a random seed for reproducibility? (Yes!)

### Your Checklist
- [ ] Split defined and justified.
- [ ] Random seed set (e.g., `random_state=42`).
- [ ] Stratification checked if applicable.
- [ ] Sizes of each split computed and documented.

### Key Learning Point
Ask yourself: "Why does a validation set exist separate from a test set?"
- Training set: Used to fit the model.
- Validation set: Used to tune hyperparameters and compare models.
- Test set: Used only at the very end to report final performance.

---

## Part B: Normalization

### Task 2: Normalize Pixel Values
Questions to answer:
1. Should you normalize pixel values from [0, 255] to [0, 1]?
2. Should you standardize (z-score: mean=0, std=1)?
3. How will you compute mean and std? (Train set only!)
4. Will you apply the same transformation to val and test?

### Decision Points
- **Option A:** Divide by 255 (simple scaling to [0, 1]).
  - Pros: Simple, preserves relative relationships.
  - Cons: Doesn't account for distribution.

- **Option B:** Standardize (z-score).
  - Pros: Better for many ML algorithms.
  - Cons: Assumes roughly normal distribution.

### Your Checklist
- [ ] Decided on normalization strategy and justified it.
- [ ] Computed mean/std on training data only.
- [ ] Applied transformation to all splits.
- [ ] Verified pixel ranges after normalization.
- [ ] Saved normalization parameters (mean, std) for later inference.

### Key Learning Point
Ask yourself: "Why does normalizing input features matter?"
- Helps gradient descent converge faster.
- Puts all features on the same scale (important for distance-based models).
- Makes model weights interpretable.

---

## Part C: Flatten vs. Preserve Structure

### Task 3: Data Shape for Modeling
Questions to answer:
1. Will you flatten 28×28 images into 784-dimensional vectors?
2. Or will you keep them as 28×28 matrices (for CNNs later)?
3. Why does this choice matter?

### Your Checklist
- [ ] Decided on data shape (flat or image format).
- [ ] Documented the rationale.
- [ ] Created both formats (or at least documented why you didn't).

### Key Learning Point
- **Flattened (784-d vector):** Best for simple models (logistic regression, SVM, tree-based).
  - Loses spatial structure. Model treats all pixel positions equally.
  
- **Image format (28×28 matrix):** Best for CNNs.
  - Preserves spatial locality. Convolutional filters can learn local patterns.

Later, you might compare both to see if preserving structure helps!

---

## Part D: Handle Missing or Anomalous Data

### Task 4: Data Cleaning
Questions to answer:
1. Are there any NaN or missing values? (Unlikely in MNIST, but check!)
2. Are there any outliers or anomalies you noticed in Phase 2?
3. Should you remove duplicates? How?
4. Should you remove or flag suspicious samples?

### Your Checklist
- [ ] Checked for NaN, infinity, or invalid values.
- [ ] Decided on handling strategy (remove, fill, flag).
- [ ] Removed or flagged anomalies if any.

---

## Part E: Create Train/Val/Test Datasets

### Task 5: Save Prepared Data
Questions to answer:
1. How will you save the prepared data? (CSV, NumPy, Parquet?)
2. Will you save them as separate files?
3. How will you organize them for easy loading?

### Your Checklist
- [ ] Prepared train, val, test datasets.
- [ ] Saved to `data/processed/`.
- [ ] Created a simple script or function to load them reproducibly.
- [ ] Documented the preprocessing pipeline.

---

## Part F: Document the Pipeline

### Task 6: Create a Preprocessing Summary
Write down the exact steps:

```
Preprocessing Pipeline:
1. Split MNIST into train (60%), val (20%), test (20%) with stratification.
2. Normalize pixel values: [0, 255] → [0, 1] by dividing by 255.
3. Flatten 28×28 images to 784-dimensional vectors.
4. Removed [N] anomalous samples with [reason].
5. Final shapes: train X (50000, 784), train y (50000,), etc.
```

**Why document this?** Because when you build the model, you need to apply the **exact same** transformations to new data. No surprises.

### Your Checklist
- [ ] Written down all preprocessing steps in order.
- [ ] Noted parameters (e.g., normalization values).
- [ ] Ensured reproducibility.

---

## Output for This Phase

- [ ] Train/val/test split defined (sizes and rationale).
- [ ] Random seed set and documented.
- [ ] Normalization strategy chosen and applied.
- [ ] Data shape decided (flat vs. image format).
- [ ] Missing/anomalous data handled.
- [ ] Processed datasets saved to `data/processed/`.
- [ ] Preprocessing pipeline documented.
- [ ] Verification: train and val/test have no overlap. ✓

**When you're done, update CRISP_DM_LOG.md, then move to notebook 04.**

---

## Key Habits

1. **Fit only on training data.** Never fit scaler/encoder on the full dataset or test set.
2. **Document every transformation.** You'll need it for inference later.
3. **Use random seeds.** Reproducibility is a superpower.
4. **Verify no leakage.** Check that val and test truly don't appear in train.
5. **Save intermediate artifacts.** Saves time if you need to re-run later.
