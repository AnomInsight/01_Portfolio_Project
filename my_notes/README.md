# Portfolio Project: Handwritten Digit Recognition (MNIST)

## Business Scenario

**Company:** MailSort Inc., a mail automation and logistics company  
**Problem:** Manually reading and sorting handwritten ZIP codes from scanned envelopes is time-consuming and error-prone.  
**Goal:** Build an automated handwritten digit classifier to reduce manual effort and accelerate sorting workflows.

### Business Objectives
- Automate digit recognition from scanned handwritten ZIP codes.
- Reduce manual intervention, improving throughput.
- Maintain high reliability (avoid misclassifications that lead to misdirected mail).

### Data Science Objectives
- Build a classifier for digits 0–9 with high accuracy and per-digit precision/recall.
- Understand which digits are inherently harder to classify and why.
- Produce a model that is accurate, interpretable, and deployable.

### Success Criteria
- Overall test accuracy ≥ 95%.
- Per-digit recall ≥ 90% (minimize false negatives; misdirected mail is costly).
- Inference time acceptable for batch processing (preferably milliseconds per digit).
- Clear documentation of failure modes and tradeoff analysis.

### Non-Goals
- Achieve state-of-the-art accuracy (this is a learning project).
- Deploy to production (but design as if we will).
- Solve segmentation or localization of digits in a full envelope image.

---

## Project Structure

```
.
├── README.md                      # This file
├── notebooks/                     # Jupyter notebooks for exploration and learning
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling_baseline.ipynb
│   ├── 05_modeling_advanced.ipynb
│   └── 06_evaluation_and_analysis.ipynb
├── src/                           # Reusable Python modules
│   ├── __init__.py
│   ├── data/                      # Data loading and preprocessing
│   ├── features/                  # Feature engineering (if applicable)
│   ├── models/                    # Model definitions and utilities
│   └── evaluation/                # Evaluation metrics and analysis
├── data/                          # Data storage
│   ├── raw/                       # Original MNIST data
│   ├── processed/                 # Cleaned and prepared data
│   └── predictions/               # Model predictions for analysis
├── reports/                       # Outputs: figures, tables, analysis
│   └── figures/
├── models/                        # Trained model checkpoints
├── pyproject.toml                 # Project dependencies and metadata
├── .gitignore                     # Git ignore rules
└── CRISP_DM_LOG.md               # Your journal of decisions
```

---

## CRISP-DM Workflow

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology:

1. **Business Understanding** → Framing the problem, defining success metrics.
2. **Data Understanding** → Exploratory analysis of MNIST dataset.
3. **Data Preparation** → Preprocessing, splitting, normalization.
4. **Modeling** → Building and comparing baseline and advanced models.
5. **Evaluation** → Comprehensive evaluation, error analysis, and recommendation.
6. **Deployment** → Designing a production-ready inference pipeline (conceptual for this project).

Each phase has a corresponding notebook. Move through them sequentially; use your learnings from earlier phases to inform later decisions.

---

## How to Use This Project

### Start Here
1. Read this README to understand the business context.
2. Open `notebooks/01_business_understanding.ipynb` and fill in your problem framing.
3. Progress to notebook 02 once you've formalized the problem.

### Key Learning Habits
- **Document decisions:** Write down *why* you make choices, not just what you do.
- **Experiment intentionally:** Change one thing at a time; compare results.
- **Analyze failures:** Misclassifications are learning opportunities.
- **Avoid peeking:** Don't use test data until the evaluation phase.
- **Keep it reproducible:** Use seeds, version dependencies, save preprocessing pipelines.

### When to Move Code to `src/`
Once a notebook cell becomes reusable (e.g., a preprocessing function, a model wrapper, an evaluation metric), move it to `src/` and import it back. This habit keeps your notebooks clean and your code modular.

---

## Dependencies

See `pyproject.toml` for the full list. Key packages:
- `numpy`, `pandas` → data manipulation
- `scikit-learn` → classical ML models and preprocessing
- `matplotlib`, `seaborn` → visualization
- `jupyter`, `ipykernel` → notebooks

Install with:
```bash
uv venv
source .venv/Scripts/activate  # Windows
uv pip install -r requirements.txt
```

---

## Expected Timeline

This is a self-paced learning project. Expect:
- Phases 1–3 (Business, Data, Prep): ~2–3 hours
- Phases 4–5 (Modeling & Evaluation): ~3–5 hours
- Phase 6 (Deployment/Reflection): ~1–2 hours

**Total:** ~8–12 hours for a thorough portfolio project.

---

## Deliverables

By the end, you should have:
- ✅ Completed notebooks (01–06) with code and analysis.
- ✅ A model comparison table in `reports/`.
- ✅ Error analysis and confusion matrix in `reports/`.
- ✅ A saved baseline and advanced model in `models/`.
- ✅ A README summary explaining your decisions and tradeoffs.
- ✅ A `CRISP_DM_LOG.md` documenting key decisions and learnings.

This becomes your portfolio piece.

---

## Learning Checklist

As you progress, check off these milestones:

- [ ] Business problem is clearly defined in notebook 01.
- [ ] Dataset is explored, visualized, and understood in notebook 02.
- [ ] Data preprocessing pipeline is designed and documented in notebook 03.
- [ ] Baseline model (logistic regression or similar) is built and evaluated.
- [ ] Advanced model is built and compared against baseline.
- [ ] Evaluation includes per-class metrics, confusion matrix, and error analysis.
- [ ] Misclassifications are analyzed and explained.
- [ ] Final recommendation is documented with pros/cons of each approach.
- [ ] Deployment considerations are discussed (even if not fully implemented).
- [ ] All decisions are logged in `CRISP_DM_LOG.md`.

---

## References

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- CRISP-DM Methodology: https://www.sas.com/en_us/insights/analytics/crisp-dm.html
- Scikit-learn Documentation: https://scikit-learn.org/stable/
