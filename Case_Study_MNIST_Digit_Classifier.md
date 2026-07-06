# Case Study: Automated Handwritten Digit Classification for Mail Sorting

**Project type:** End-to-end machine learning — image classification  
**Methodology:** CRISP-DM  
**Stack:** Python · scikit-learn · TensorFlow/Keras · NumPy · Matplotlib  
**Duration:** Self-directed portfolio project  

---

## The Problem

Manual reading of handwritten ZIP codes is one of the last human-intensive steps in mail routing operations. A sorting worker reads each digit on an envelope, keys it in, and routes the piece — a process that is slow, fatiguing, and prone to transcription errors under volume. For a mid-scale postal or logistics operation, even a one-percent misrouting rate translates to thousands of misdirected items per day and a measurable customer-service cost.

The business question this project set out to answer was simple: *can a machine learning model read handwritten digits reliably enough to replace or meaningfully assist that manual step?*

The target was set deliberately: **overall accuracy of 95% or above**, with strong per-digit recall across all ten classes, and a clear picture of where and how the model fails — because in a routing context, confident wrong answers can be worse than uncertain ones.

---

## Approach

Rather than jumping straight to the most powerful model, the project followed a structured CRISP-DM process — moving from understanding the business context through data exploration, careful preparation, progressive modeling, and finally a rigorous evaluation phase that went well beyond a single accuracy number.

### Data

The dataset used was MNIST — 70,000 grayscale images of handwritten digits (0–9) at 28×28 pixels, loaded directly from raw IDX binary files. The training set of 60,000 images was split into a stratified 80/20 train/validation partition (48,000 / 12,000), preserving class balance throughout. The 10,000-image test set was held out completely until final evaluation.

Preprocessing was kept deliberately simple: pixel values scaled to [0, 1], with a channel dimension added for the CNN. No augmentation was applied during training — a deliberate choice that made the robustness findings more meaningful later.

### Modeling progression

Five distinct approaches were trained and evaluated, in order of complexity:

| Model | Test Accuracy | Notes |
| Dummy baseline | ~10% | Sanity lower bound — majority class |
| Logistic Regression | 91.65% | Strong linear baseline, pixel-space features |
| Random Forest | 96.87% | 250 trees, depth 25 — captured nonlinear interactions |
| SVM (RBF kernel, C=10) | 98.36% | Near-top performance; high training cost |
| **CNN (selected)** | **99.27%** | 2× Conv + MaxPool + Dense + Dropout |
| Majority-vote Ensemble | 98.66% | Combined LR + RF + SVM — did not beat CNN |

Every model beat the 95% business target. The CNN did so most convincingly, and with better generalization characteristics than the SVM (which showed near-perfect training accuracy and a larger train-test gap).

### CNN architecture

The selected CNN used two convolutional blocks — each with a Conv2D layer and MaxPooling — followed by a Flatten, a 128-unit Dense layer with 30% Dropout, and a 10-class Softmax output. It was trained with Adam (lr=0.001) and early stopping on validation loss, completing in approximately 52 seconds on CPU.

---

## Results

### Overall performance

The CNN achieved **99.27% accuracy on the held-out test set** (9,927 correct out of 10,000), with a training accuracy of 99.77% and a validation accuracy of 98.97%. The 0.5-point train-test gap is mild and consistent with healthy generalization rather than overfitting.

### Confusion patterns

The model's residual errors follow an interpretable pattern. The most frequent confusion pairs were:

- **8 predicted as 9** — shared loop structure in the upper half
- **3 predicted as 5** — similar open-curve geometry
- **5 predicted as 9** — tail and loop shape overlap in certain handwriting styles

These are the same pairs that trip up human readers with messy handwriting. That alignment is a useful signal — it means the model is generalizing from real shape features rather than memorizing artifacts.

### Error taxonomy

To go beyond aggregate metrics, the top 20 highest-confidence misclassifications were reviewed manually and categorized:

- **60% (12/20): Understandable confusion** — ambiguous handwriting where even a human examiner would hesitate. These represent the inherent difficulty ceiling of the task.
- **40% (8/20): Model failure** — cases where the image was relatively clear but the model predicted the wrong class with high confidence. These are the operationally dangerous errors.

The 40% model-failure category is the most important finding from an operational standpoint. A system that auto-accepts all predictions with high confidence will systematically route a meaningful share of those failures without any safeguard.

### How the model holds up under real-world conditions

Real envelopes arrive with smudges, skew, and varying exposure. To get a sense of how the model would handle that, the full test set was run through four modified versions and accuracy was measured each time:

| Test condition | Accuracy |
| Clean images (baseline) | 99.27% |
| Rotated +10° | 98.42% |
| Rotated −10° | 98.54% |
| Light Gaussian noise added | 99.21% |
| Reduced brightness (×0.8) | 99.27% |

The model is essentially immune to mild noise and brightness shifts. Rotation is where it shows sensitivity — an expected finding given that no rotational augmentation was used during training and convolutional architectures are not rotation-invariant by default.

### Confidence calibration

The model's confidence scores carry genuine signal. Correct predictions cluster at very high confidence (>0.98 on average), while incorrect predictions show lower confidence on average — but not reliably so. A meaningful portion of errors still arrive with high confidence (>0.90), which reinforces the case for a threshold-based routing policy rather than blind auto-acceptance.

---

## What This Means Operationally

The raw accuracy numbers tell a positive story, but the more important takeaways come from the failure analysis:

1. **Auto-acceptance is risky without a threshold.** High confidence does not guarantee a correct prediction. A blanket policy of auto-accepting all predictions would inherit the 40% model-failure share from the high-confidence error pool.

2. **A confidence gate significantly reduces operational risk.** Routing predictions below a confidence threshold (e.g., 0.90) to a manual review queue would catch the most uncertain cases with minimal throughput impact, given that the vast majority of predictions land above 0.95.

3. **Rotation is the main weak point.** If the deployment context involves tilted envelopes — which is common in real postal equipment — training with randomly rotated images would likely close that gap for little additional cost.

4. **Monitoring matters post-deployment.** Class-level accuracy, confusion hotspots, and the high-confidence error rate should all be tracked on a rolling basis. A drift in any of these metrics — even without a large overall accuracy drop — can be an early warning of distribution shift.

### Proposed deployment flow (conceptual)

```
Input image
    ↓
Validate (shape, type)
    ↓
Preprocess (scale to [0,1], reshape 28×28×1)
    ↓
CNN inference → class + confidence score
    ↓
Confidence ≥ 0.90 → auto-accept
Confidence < 0.90 → route to manual review queue
    ↓
Monitor: rolling accuracy · per-class recall · confusion drift · high-conf error rate
    ↓
Retrain trigger: sustained KPI drop | confusion shift | data distribution change
```

> *Note: this deployment design is conceptual. A live inference API, cloud pipeline, and automated retraining job were intentionally out of scope for this project iteration.*

---

## What I Would Do Differently at Scale

A few things that would matter in a real production environment that this project treated as out of scope:

- **Proper calibration analysis.** A reliability diagram and Expected Calibration Error (ECE) score would make the confidence threshold choice more principled rather than heuristic.
- **Domain-specific data.** MNIST is clean and balanced. Real envelope images have ink bleeding, torn paper, mixed fonts, and camera distortion. A production model would need fine-tuning on real capture samples.
- **Rotation augmentation.** The robustness test made the sensitivity clear. Training with ±15° random rotation would likely eliminate most of that gap.
- **Model serving infrastructure.** Exporting the model to TensorFlow SavedModel format or ONNX and wrapping it in a lightweight FastAPI service would be a natural next step.

---

## Skills Demonstrated

- **End-to-end ML workflow** following CRISP-DM across all phases
- **Progressive modeling** — baseline → classical ML → deep learning — with justified model selection
- **Deep evaluation beyond accuracy** — confusion analysis, manual error taxonomy, confidence behavior, robustness testing
- **Business framing** — translating model metrics into operational risk and deployment recommendations
- **Reusable code design** — extracted a general-purpose toolkit (`image_utils`, `tabular_utils`, `text_utils`, `eval_utils`, `timeseries_utils`, `common/`) for use across future projects
- **Reproducibility** — fixed seeds, version-locked dependencies via `uv`, structured project layout

---

## Figures

![Confusion Matrix — Test Set](figures/confusion_matrix_test.png)

![Learning Curves](figures/cnn_learning_curves.png)

![Confidence Distribution: Correct vs Incorrect](figures/confidence_distribution.png)

![Top-20 High-Confidence Errors](figures/top20_high_confidence_errors.png)

![Misclassified Examples Grid](figures/misclassified_examples_grid.png)

![Robustness Under Perturbation](figures/robustness_variants_accuracy.png)

---

*Full source code, evaluation pipeline, and decision log available in this repository.*
