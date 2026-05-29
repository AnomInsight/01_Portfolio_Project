# 05 Modeling – Advanced

## Objective
Build more sophisticated models that leverage image structure or nonlinear patterns. Compare against baselines.

**Key principle:** Only increase complexity if there's a clear rationale and measurable improvement.

---

## Part A: Decision: What's the Gap?

### Task 1: Identify Why Baseline Isn't Enough
Before building an advanced model, ask:
1. What's your current best baseline accuracy?
2. What's your business target from Phase 1?
3. What's the gap?
4. **Why** does the gap exist? What can't the baseline learn?

Examples:
- Gap = 5%: "Baseline misses spatial relationships. CNN might help."
- Gap = 0.5%: "Baseline is already near-optimal. More complexity won't help much."
- Gap = 15%: "Maybe you need more data or better features, not just a fancier model."

### Your Checklist
- [ ] Gap explicitly identified.
- [ ] Root cause hypothesized.
- [ ] Decision made: "Advanced model X will address this by..."

---

## Part B: Option 1 – Support Vector Machine (SVM)

### Task 2: Train an SVM (if gap warrants it)
Questions to answer:
1. Why SVM?
   - Finds nonlinear decision boundaries using kernels.
   - Works well on structured data.
   - Still interpretable (support vectors).
   - Slower than forests but often more accurate.

2. What kernel should you use?
   - Linear? RBF? Polynomial?
   - Each captures different nonlinearities.

3. How will you tune C (regularization)?
   - Use validation set to find a good value.
   - Don't overfit to training data.

### Your Task
```
1. Fit an SVM (e.g., sklearn's SVC) with an appropriate kernel.
2. Tune hyperparameters using validation set (not test!).
3. Evaluate on train, val, test.
4. Create confusion matrix.
5. Compare to baselines: did it improve? By how much?
```

### Questions to Answer
- Test accuracy?
- Improvement over best baseline?
- Training time?
- Is the improvement worth the added complexity?
- Which digits still confuse it?

### Your Checklist
- [ ] SVM trained and tuned.
- [ ] Accuracy on train/val/test computed.
- [ ] Confusion matrix visualized.
- [ ] Compared to baselines.

---

## Part C: Option 2 – Convolutional Neural Network (CNN)

### Task 3: Train a CNN (if warranted)
Questions to answer:
1. Why CNN?
   - Explicitly models spatial structure of images.
   - Convolutional filters learn local patterns (edges, shapes).
   - Much more complex than baselines (more hyperparameters).
   - Needs careful tuning to avoid overfitting.

2. Basic architecture (start simple!):
   - 1–2 convolutional layers?
   - Pooling to reduce dimensions?
   - Fully connected layers at the end?
   - Dropout for regularization?

3. Training considerations:
   - How many epochs?
   - Batch size?
   - Learning rate?
   - When to stop (early stopping on validation set)?

### Your Task
```
1. Design a simple CNN architecture.
2. Train it with careful monitoring (validation loss).
3. Use early stopping to avoid overfitting.
4. Evaluate on train, val, test.
5. Create confusion matrix.
6. Compare to baselines and SVM.
```

### Questions to Answer
- Test accuracy?
- Training time vs. SVM?
- Overfitting? (Train accuracy >> Test accuracy?)
- Is the improvement over simpler models worth the complexity?
- Which digits still confuse it?

### Your Checklist
- [ ] CNN architecture designed and justified.
- [ ] Trained with early stopping on validation set.
- [ ] Learning curves plotted (training loss vs. validation loss over epochs).
- [ ] Accuracy on train/val/test computed.
- [ ] Confusion matrix visualized.
- [ ] Compared to all previous models.

### Key Learning Point
Ask yourself: "Why might CNN fail despite being complex?"
- Overfitting: too many parameters, too little regularization, not enough data.
- Underfitting: architecture too simple, not training long enough.
- Bad hyperparameters: learning rate too high/low, batch size wrong.

This is where **validation set** becomes crucial. Use it to find when to stop.

---

## Part D: Ensemble or Hybrid Approach (Optional)

### Task 4: Combine Multiple Models (Advanced)
Questions to answer:
1. What if you average predictions from multiple models?
2. Does an ensemble beat any single model?
3. Is ensemble complexity justified?

### Your Task (Optional)
```
1. Train 2–3 different models (e.g., SVM, CNN, Random Forest).
2. Ensemble their predictions: average probabilities or majority voting.
3. Evaluate: does ensemble improve?
4. How much? Is it worth the added complexity?
```

### Your Checklist
- [ ] Ensemble trained (optional).
- [ ] Ensemble accuracy computed.
- [ ] Compared to single best model.

---

## Part E: Hyperparameter Tuning

### Task 5: Tune Your Best Advanced Model
Questions to answer:
1. Which hyperparameters matter most for your model?
2. How will you search for good values?
   - Grid search (try all combinations)?
   - Random search (try random combinations)?
   - Or manual tuning based on intuition?

3. How will you avoid overfitting to the validation set?
   - Use a separate validation set, not test.
   - Don't tune endlessly; stop when gains plateau.

### Your Task
```
1. Identify 2–3 key hyperparameters.
2. Manually tune or grid search on validation set.
3. Monitor: train accuracy, val accuracy, test accuracy.
4. Stop when val accuracy stops improving.
5. Report final test accuracy.
```

### Your Checklist
- [ ] Hyperparameters identified and justified.
- [ ] Tuning performed on validation set only.
- [ ] Test accuracy reported (not used for tuning).

### Key Learning Point
Tuning hyperparameters is an art. Too much tuning and you'll overfit the validation set. Balance exploration with restraint.

---

## Part F: Model Comparison Summary

### Task 6: Compare All Models
Create a comprehensive table:

| Model | Arch/Type | Test Acc | Val Acc | Precision (avg) | Recall (avg) | Training Time | Hyperparameters | Overfitting? | Notes |
|-------|-----------|----------|---------|-----------------|--------------|---------------|-----------------|--------------|-------|
| Logistic Reg | Linear | ? | ? | ? | ? | ? | C=1.0 | ? | Baseline |
| Random Forest | Tree ensemble | ? | ? | ? | ? | ? | n_trees=100 | ? | Baseline |
| SVM | Nonlinear | ? | ? | ? | ? | ? | C=?, kernel=? | ? | Option 1 |
| CNN | Conv layers | ? | ? | ? | ? | ? | layers=?, lr=? | ? | Option 2 |
| Ensemble | Mixed | ? | ? | ? | ? | ? | – | ? | Optional |

### Your Task
1. Fill in the table with all models you trained.
2. Highlight the best performer.
3. Add a "Notes" column explaining pros/cons.

### Your Checklist
- [ ] All models compared in a single table.
- [ ] Best model identified.

---

## Part G: Final Advanced Model Selection

### Task 7: Decide: Baseline or Advanced?
Answer:
1. Which model is best overall?
2. Is it worth switching from your best baseline?
   - Improvement in accuracy?
   - Trade-offs in complexity, speed, interpretability?
3. Could you justify this choice to a stakeholder?

### Your Checklist
- [ ] Best model decided.
- [ ] Decision justified in writing (pros/cons vs. simpler alternatives).

---

## Output for This Phase

- [ ] SVM trained, tuned, and evaluated (if attempted).
- [ ] CNN trained, tuned, and evaluated (if attempted).
- [ ] Ensemble trained and evaluated (if attempted).
- [ ] Comprehensive model comparison table.
- [ ] Learning curves plotted (especially for CNN).
- [ ] Hyperparameter tuning documented.
- [ ] Final model decision and justification.

**When you're done, update CRISP_DM_LOG.md, then move to notebook 06.**

---

## Key Habits

1. **Don't add complexity without reason.** Simpler is better unless advanced model clearly helps.
2. **Tune on validation set only.** Never peek at test set.
3. **Plot learning curves.** They tell you if you're overfitting or underfitting.
4. **Document hyperparameters.** You'll need them for reproducibility and deployment.
5. **Know when to stop.** If gains plateau, stop tuning and move on.
