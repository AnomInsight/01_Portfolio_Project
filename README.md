# MNIST Handwritten Digit Classifier (Portfolio Project)

End-to-end machine learning case project that frames digit recognition as a real business problem in mail sorting, then solves it using a structured CRISP-DM workflow.

## Why This Project

Manual reading of handwritten ZIP code digits is slow and error-prone in high-volume mail operations. This project evaluates whether a machine learning pipeline can classify digits reliably enough to reduce manual workload while keeping routing risk low.

## At a Glance

- Goal: classify digits 0-9 from grayscale images with strong per-class performance.
- Methodology: CRISP-DM (business framing -> data understanding -> prep -> modeling -> evaluation).
- Stack: Python, NumPy, pandas, scikit-learn, matplotlib, seaborn.
- Portfolio focus: model comparison, error analysis, and business-facing recommendations.

## Results Snapshot

Detailed metrics are documented in the full case study:

- [Case_Study_MNIST_Digit_Classifier.md](Case_Study_MNIST_Digit_Classifier.md)

Summary of the reported progression:

- Dummy baseline: ~10%
- Logistic Regression: 91.65%
- Random Forest: 96.87%
- SVM (RBF): 98.36%
- CNN (selected in case study): 99.27%

## Repository Contents

```text
.
├── README.md
├── Case_Study_MNIST_Digit_Classifier.md
├── pyproject.toml
├── uv.lock
├── src/
│   ├── 01_data_load.py
│   ├── 02_data_preparation.py
│   ├── 03_baseline_model.py
│   ├── 04_dummy_baseline.py
│   ├── 05_Log_reg.py
│   └── 06_rand_for.py
├── reports/
└── my_notes/
```

## Quick Start

### 1. Environment setup

```bash
uv venv
.venv\Scripts\activate
uv sync
```

### 2. Run scripts

From the project root, run any stage script in `src/`:

```bash
python src/01_data_load.py
python src/02_data_preparation.py
python src/03_baseline_model.py
python src/04_dummy_baseline.py
python src/05_Log_reg.py
python src/06_rand_for.py
```

## Project Navigation

- Start with this README for a fast portfolio overview.
- Read the complete story in [Case_Study_MNIST_Digit_Classifier.md](Case_Study_MNIST_Digit_Classifier.md).
- Check `reports/` for outputs and analysis artifacts.

## Portfolio Positioning

For GitHub portfolios, the strongest structure is:

1. README as a short recruiter-friendly landing page.
2. Case study as the deep technical and business narrative.

This repository follows that pattern.

## References

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- CRISP-DM Methodology: https://www.sas.com/en_us/insights/analytics/crisp-dm.html
- Scikit-learn Documentation: https://scikit-learn.org/stable/
