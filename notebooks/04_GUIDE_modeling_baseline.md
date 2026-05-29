# 04 Modeling – Baseline

## Objective
Build simple baseline models to establish a performance floor and understand what you're working with.

**Key principle:** Start stupid, then add complexity only if justified. Baseline models teach you what the problem looks like before you invest in fancy architecture.

---

## Part A: Dummy Baseline

### Task 1: Build a Random Classifier
Questions to answer:
1. What happens if you just guess randomly?
2. What happens if you always predict the most common class?
3. Why does a dummy baseline matter?

### Your Task
Build two trivial models:
- **Random predictor:** Randomly guess 0–9 for each sample.
- **Majority class predictor:** Always predict the most common digit.

Evaluate both on your test set:
- Accuracy?
- Confusion matrix (is it just noise or mostly one class)?
- Per-class recall?

### Your Checklist
- [ ] Random classifier accuracy computed (~10% for 10 classes).
- [ ] Majority class classifier accuracy computed.
- [ ] Baseline established: "Any useful model should beat X% accuracy."

### Key Learning Point
Why do this? Because if your real model only beats the dummy baseline by 1%, it's not learning much. If it beats it by 80%, you've actually done something.

---

## Part B: Simple Linear Model (Logistic Regression)

### Task 2: Build and Fit Logistic Regression
Questions to answer:
1. Why start with logistic regression?
   - It's interpretable: you can see which pixels matter.
   - It's fast: trains in seconds.
   - It's a strong baseline: surprisingly effective on MNIST.

2. What hyperparameters do you need to set?
   - Regularization strength (C or alpha)?
   - Solver (lbfgs, sgd, etc.)?
   - Max iterations?
   - Why does regularization matter?

3. How will you avoid overfitting?
   - Use regularization.
   - Don't tune too aggressively on training set.

### Your Task
Implement logistic regression:
```
1. Use sklearn's LogisticRegression or similar.
2. Fit on training data.
3. Evaluate on train, validation, and test.
4. Report: accuracy, precision, recall, F1.
5. Show confusion matrix.
```

### Questions to Answer
- Accuracy on train, val, test?
- Do you see overfitting? (Train >> Val?)
- Which digits does it get right/wrong?
- Can you interpret the model? (e.g., which pixels are most important?)

### Your Checklist
- [ ] Logistic regression trained.
- [ ] Accuracy on train/val/test computed.
- [ ] Confusion matrix visualized.
- [ ] Per-class metrics (precision, recall, F1) computed.
- [ ] Interpretation attempted (can you explain why certain digits are confused?).

### Key Learning Point
Ask yourself: "Why would logistic regression struggle with MNIST?"
- It's a linear model; it can't learn complex nonlinear patterns.
- It treats all pixel locations equally (ignores spatial structure).
- But it's still surprisingly effective! Pixels actually encode digit identity linearly to some degree.

---

## Part C: Simple Tree-Based Model (Decision Tree or Random Forest)

### Task 3: Build and Fit a Tree-Based Classifier
Questions to answer:
1. Why add a tree-based model?
   - Decision trees can learn nonlinear patterns.
   - Random forests average multiple trees (more robust).
   - Still interpretable (feature importance).
   - Faster than neural networks.

2. What hyperparameters are important?
   - Tree depth?
   - Number of trees (if using random forest)?
   - Min samples per leaf?
   - Why do these matter?

### Your Task
Implement a tree-based model (e.g., Random Forest):
```
1. Use sklearn's RandomForestClassifier or similar.
2. Fit on training data.
3. Evaluate on train, val, test.
4. Report accuracy, precision, recall, F1.
5. Show confusion matrix.
6. Extract feature importances (which pixels matter?).
```

### Questions to Answer
- Accuracy on train/val/test?
- Does it improve over logistic regression? By how much?
- Is it overfitting?
- Which digits does it confuse?
- Can you interpret it? (What are the top important features?)

### Your Checklist
- [ ] Tree-based model trained.
- [ ] Accuracy on train/val/test computed.
- [ ] Confusion matrix visualized.
- [ ] Per-class metrics computed.
- [ ] Feature importances extracted and visualized.

### Key Learning Point
Ask yourself: "Why might Random Forest be better than logistic regression?"
- It can learn pixel neighborhoods (e.g., "if pixels at (10,10), (10,11), (10,12) are bright, it's likely a 1").
- But it still doesn't explicitly model spatial structure (a CNN would).

---

## Part D: Model Comparison

### Task 4: Compare Baselines
Create a comparison table:

| Model | Train Acc | Val Acc | Test Acc | Precision (avg) | Recall (avg) | Training Time | Notes |
|-------|-----------|---------|----------|-----------------|--------------|---------------|-------|
| Random | ~10% | ~10% | ~10% | – | – | <1s | Baseline |
| Logistic Reg | ? | ? | ? | ? | ? | ? | [Notes] |
| Random Forest | ? | ? | ? | ? | ? | ? | [Notes] |

### Your Task
1. Fill in the table.
2. Identify which model performs best overall.
3. For each model, note: "What does it do well? What does it struggle with?"

### Your Checklist
- [ ] Comparison table completed.
- [ ] Best baseline identified and justified.

---

## Part E: Error Analysis

### Task 5: Dig Into Mistakes
For your best baseline model, answer:
1. Which digits are most often confused?
2. Are there patterns in the confusion? (e.g., always 4→9, never 9→4?)
3. Can you look at misclassified examples and understand why?
   - Show 5–10 examples where the model was wrong.
   - Do they look genuinely hard, or is the model just confused?

### Your Checklist
- [ ] Top 5 confusion pairs identified.
- [ ] Misclassified examples visualized.
- [ ] Hypotheses formed about why those confusions happen.

### Key Learning Point
Misclassifications are gold. They tell you what the model doesn't understand.

---

## Part F: Takeaways

### Task 6: Summarize Baseline Findings
Answer these:
1. What's the best baseline accuracy you achieved?
2. Is this good enough for the business problem? (Recall your success criteria from Phase 1.)
3. If not, what's missing?
   - Linear separability not enough?
   - Need to preserve spatial structure?
   - Need more model complexity?
4. What should an advanced model do better?

### Your Checklist
- [ ] Best baseline documented.
- [ ] Gap between baseline and business target identified.
- [ ] Hypotheses for improvement listed (e.g., "CNN to preserve spatial structure").

---

## Output for This Phase

- [ ] Dummy baselines (random + majority) evaluated.
- [ ] Logistic regression trained, evaluated, confusion matrix shown.
- [ ] Tree-based model trained, evaluated, confusion matrix shown.
- [ ] Model comparison table filled in.
- [ ] Top confusion pairs identified.
- [ ] 5–10 misclassified examples visualized and analyzed.
- [ ] Summary: baseline findings and gaps identified.

**When you're done, update CRISP_DM_LOG.md, then move to notebook 05.**

---

## Key Habits

1. **Start simple.** Logistic regression beats 80% of fancy solutions most people build.
2. **Always use train/val/test consistently.** Report all three.
3. **Check confusion matrix, not just accuracy.** It tells a different story.
4. **Analyze errors.** They're your learning opportunity.
5. **Know when to stop.** If your baseline is already good enough, don't add complexity.
