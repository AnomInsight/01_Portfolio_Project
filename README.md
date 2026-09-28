# MNIST Handwritten Digit Classification

A self-directed machine-learning portfolio project comparing classical classifiers and a convolutional neural network on the MNIST handwritten-digit benchmark. Mail sorting is used only as a hypothetical business context; no postal data or live operation was involved.

## Business Problem

In a hypothetical mail workflow, a digit recognizer could assist staff by proposing labels for cropped handwritten digits, with uncertain cases sent for review. This experiment asks how several models perform on MNIST and examines their errors. It does not establish performance on postal images or estimate operational cost savings.

## Methodology

The work is organized around CRISP-DM: frame the use case and illustrative success target, inspect and prepare MNIST, compare baseline and increasingly complex models, then review errors and synthetic image perturbations. The data pipeline uses a stratified 80/20 train-validation split of the 60,000 training examples and evaluates against the standard 10,000-example test set. The scripts report test metrics for multiple model runs, so the test set was not reserved for one final model-selection decision.

## Results

These are the values recorded in the case study, not a fresh run performed during this repository audit. They are benchmark results, not estimates of postal-operation performance.

| Model | MNIST test accuracy |
| --- | ---: |
| Random baseline | ~10% |
| Logistic Regression | 91.65% |
| Random Forest | 96.87% |
| SVM (RBF) | 98.36% |
| CNN | **99.27%** |
| Majority-vote Ensemble | 98.66% |

The recorded CNN result is 99.27% on 10,000 MNIST test images. Training runs, particularly the CNN, may not reproduce exactly across environments.

## Technology Stack

Python 3.11-3.13, NumPy, pandas, SciPy, scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn, idx2numpy, and uv.

## Repository Structure

```text
.
├── Case_Study_MNIST_Digit_Classifier.md
├── my_notes/                        # Project notes and working material
├── notebooks/                       # Markdown guides for CRISP-DM phases
├── reports/                         # Project records and supporting artifacts
├── src/
│   ├── 01_data_load.py
│   ├── 02_data_preparation.py
│   ├── 03_baseline_model.py
│   ├── 04_dummy_baseline.py
│   ├── 05_Log_reg.py
│   ├── 06_Tree_based_models.py
│   ├── 07_SVM_model.py
│   ├── 08_CNN_model.py
│   ├── 09_Ensemble_model.py
│   ├── dummy_baseline.py
│   └── evaluation.py
├── pyproject.toml
└── uv.lock
```

`data/` is local and git-ignored. It is not part of the public repository; scripts expect the four uncompressed MNIST IDX files under `data/MNIST_data/` and create preprocessed arrays under `data/processed/`.

## Setup

Install Python 3.11-3.13 and [uv](https://docs.astral.sh/uv/). From the repository root, create a Python 3.11 environment and install the locked dependencies:

```powershell
uv venv --python 3.11
uv sync
```

Optional activation in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Download the four gzip files from the [CVDF MNIST repository](https://github.com/cvdfoundation/mnist), decompress them, and place them in `data/MNIST_data/`. Rename the extracted files to the exact names expected by the scripts: `train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, and `t10k-labels.idx1-ubyte`.

Create the local input directory from the project root before placing the files:

```powershell
New-Item -ItemType Directory -Force data\MNIST_data | Out-Null
```

The scripts use paths relative to their working directory. From the project root, inspect the training data, prepare the arrays, and check the saved split shapes with:

```powershell
Set-Location src
$env:PYTHONPATH = "..\0_utils"
New-Item -ItemType Directory -Force ..\data\processed, ..\reports\figures | Out-Null
uv run python 01_data_load.py
uv run python 02_data_preparation.py
uv run python 03_baseline_model.py
```

The dataset-based dummy baseline uses project-root-relative paths, so run it from the root:

```powershell
Set-Location ..
uv run python src\dummy_baseline.py
```

Train and compare the models individually from `src/` (some runs, especially the SVM, may take time):

```powershell
Set-Location src
uv run python 05_Log_reg.py
uv run python 06_Tree_based_models.py
uv run python 07_SVM_model.py
uv run python 08_CNN_model.py
uv run python 09_Ensemble_model.py
```

`05_Log_reg.py` saves pixel-importance and validation confusion-matrix plots under `reports/figures/`. `evaluation.py` trains a CNN again and runs the extended test-set error and perturbation analysis:

```powershell
uv run python evaluation.py
```

Return to the project root when finished:

```powershell
Set-Location ..
```

## Case Study

See the [MNIST digit-classifier case study](Case_Study_MNIST_Digit_Classifier.md) for model details, reported metrics, error analysis, robustness checks, and clearly labeled conceptual recommendations.

## Limitations

MNIST contains small, centered, grayscale digit images and does not represent full envelopes, postal handwriting, scanners, or routing operations. The experiment does not include digit localization, a serving API, deployment, calibrated confidence thresholds, or validation on domain-specific data. Some model scripts access test metrics during model comparisons; a new untouched holdout would be needed for a formal final comparison.

## References

- [MNIST dataset files and format (CVDF)](https://github.com/cvdfoundation/mnist)
- [CRISP-DM overview (IBM)](https://www.ibm.com/docs/en/spss-modeler/18.5.0?topic=dm-crisp-help-overview)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
