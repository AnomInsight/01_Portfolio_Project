# CRISP-DM Decision Log

Use this file to document key decisions, assumptions, and learnings as you progress through the project.
This becomes evidence of your thought process for a hiring manager.

## Business Understanding

**Date:** [Fill in]

### Problem Statement
[What problem are you solving? Why does it matter?]

### Success Metrics
[How will you measure success? What's acceptable?]

### Assumptions
[What are you assuming about the business, the data, the model?]

### Key Decisions
- [ ] Decision 1: [e.g., "Using accuracy as primary metric because false positives and negatives are equally costly"]
- [ ] Decision 2: [e.g., "Targeting 95% accuracy as it balances performance and complexity"]

### Questions/Risks
[What could go wrong? What's uncertain?]

---

## Data Understanding

**Date:** [Fill in]

### Dataset Summary
[How many samples? Class balance? Data type and shape?]

### Key Observations
- Observation 1: [e.g., "Classes are perfectly balanced with 6000 samples each"]
- Observation 2: [e.g., "Pixels range from 0–255, grayscale"]

### Surprising Findings
[Anything unexpected or interesting?]

### Hypotheses for Modeling
[What patterns do you expect the model to learn?]

---

## Data Preparation

**Date:** [Fill in]

### Preprocessing Steps
[What transformations are you applying? Why?]

### Train/Validation/Test Split
[How are you splitting? What's the rationale?]

### Decisions Made
- [ ] Decision 1: [e.g., "Normalizing pixel values to [0, 1] because most models work better with bounded inputs"]
- [ ] Decision 2: [e.g., "Using stratified split to maintain class balance across train/val/test"]

### Potential Leakage
[Are you ensuring train and test data don't influence each other?]

---

## Modeling Phase 1: Baseline

**Date:** [Fill in]

### Models Tried
- Model A (e.g., Random/Majority Classifier):
  - Rationale: [Why this first?]
  - Accuracy: [?]
  - Training Time: [?]
  - Interpretation: [What does this teach you?]

- Model B (e.g., Logistic Regression):
  - Rationale: [Why this one?]
  - Accuracy: [?]
  - Training Time: [?]
  - Interpretation: [What are the key features? Confusion patterns?]

### Baseline Findings
[What did you learn from these simple models?]

---

## Modeling Phase 2: Advanced

**Date:** [Fill in]

### Models Tried
- Model C (e.g., SVM or Random Forest):
  - Rationale: [Why upgrade to this?]
  - Accuracy: [?]
  - Training Time: [?]
  - Improvement over baseline: [+X%? Worth it?]

- Model D (e.g., Convolutional Neural Network):
  - Rationale: [When to use this?]
  - Accuracy: [?]
  - Training Time: [?]
  - Complexity vs. gain: [Is the improvement worth the added complexity?]

### Comparison
[Which model is best? By what criteria?]

---

## Evaluation & Error Analysis

**Date:** [Fill in]

### Metrics Summary
[Per-class precision/recall, confusion patterns, hardest/easiest digits]

### Error Analysis
[Which digits are confused? Which are never wrong? Why?]

### Misclassification Examples
- Hard case 1: [e.g., "Digit 4 confused with 9 because of similar stroke patterns"]
- Hard case 2: [e.g., "Digit 3 sometimes mislabeled as 5 when written sloppily"]

### Robustness Checks
[Does the model perform well on all data? Any biases?]

---

## Final Recommendation

**Date:** [Fill in]

### Chosen Model
[Which model did you choose and why?]

### Justification
[Accuracy? Interpretability? Speed? Trade-offs acknowledged?]

### Limitations
[What can this model NOT do well?]

### Next Steps
[What would you improve in a second iteration?]

---

## Key Learnings

[What is the most important lesson from this project? What will you do differently next time?]

---

## Reflection

[How did this project change your thinking about data science workflows? What surprised you?]
