# Case Study: MNIST Digit Classification in a Hypothetical Mail-Sorting Workflow

**Project type:** Self-directed image-classification portfolio project  
**Methodology:** CRISP-DM  
**Stack:** Python, scikit-learn, TensorFlow/Keras, NumPy, pandas, SciPy, Matplotlib  

## Hypothetical Business Scenario

This project uses a hypothetical mail-sorting scenario to frame handwritten digit recognition: a classifier might suggest labels for cropped digits and route uncertain cases for human review. The scenario is illustrative. No postal operator, operational workflow, postal image data, or production system was part of this project. MNIST is a benchmark of small, centered digit images, not a proxy for full envelopes or real postal capture conditions.

The project-defined success target was at least 95% overall accuracy on MNIST, alongside review of class-level performance and error types. This is an illustrative portfolio criterion, not a target supplied by a business stakeholder or evidence of operational suitability.

## Experiment

### Data and preparation

MNIST contains 70,000 grayscale images of handwritten digits 0-9, each 28 by 28 pixels: 60,000 training images and 10,000 test images. The project loads the uncompressed IDX files using `idx2numpy`. The training portion is split into 48,000 training and 12,000 validation examples with stratification and `random_state=42`. Pixel values are scaled to [0, 1]; the CNN pipeline adds a channel dimension. No augmentation is applied during training. See the [CVDF MNIST repository](https://github.com/cvdfoundation/mnist) for the dataset and file format.

The scripts evaluate test metrics for multiple model runs, and the SVM script reports test scores across configurations. Thus, although test examples are not used for fitting, the scripts do not preserve the test set exclusively for one final comparison after all model selection. Treat the comparisons as exploratory; a formal selection should use a new untouched evaluation set.

### Models and recorded results

The following values are reported in the project analysis. They have not been rerun as part of this documentation audit; script output can vary, especially for neural-network training.

| Model | MNIST test accuracy | Notes |
| --- | ---: | --- |
| Random baseline | ~10% | Approximate expected accuracy for a uniform guess over 10 classes |
| Logistic Regression | 91.65% | Flattened pixel features |
| Random Forest | 96.87% | 250 trees, maximum depth 25 |
| SVM (RBF) | 98.36% | `C=10`, `gamma="scale"` |
| **CNN** | **99.27%** | Two Conv2D/MaxPooling blocks, 128-unit dense layer, dropout |
| Majority-vote Ensemble | 98.66% | Current script combines Random Forest, SVM, and CNN |

The CNN's reported test accuracy is 99.27% (9,927 of 10,000 examples), with reported training accuracy of 99.77% and validation accuracy of 98.97%. The 0.50 percentage-point train-test difference is descriptive and does not by itself establish generalization to another dataset.

The CNN uses two convolutional blocks (32 and 64 filters, each followed by max pooling), a flattening layer, a 128-unit ReLU layer, 30% dropout, and a 10-class softmax output. It is compiled with Adam at a learning rate of 0.001 and trained with a maximum of 20 epochs, batch size 128, and early stopping on validation loss (patience 3, restoring best weights).

## Error and Robustness Analysis

### Confusions and manual review

The recorded analysis identifies 8-to-9, 3-to-5, and 5-to-9 among the frequent confusion pairs. These are observations from the MNIST experiment only; no comparison with human readers was conducted.

The 20 highest-confidence test errors were manually categorized in the evaluation script: 12 (60%) as “understandable confusion” and 8 (40%) as “model failure.” These are subjective labels on a selected sample of high-confidence errors, not a representative estimate of real-world error types or rates.

### Synthetic perturbation checks

The evaluation script applies four fixed transformations to the MNIST test images. The recorded accuracy values are:

| Test condition | Accuracy |
| --- | ---: |
| Clean images | 99.27% |
| Rotation +10 degrees | 98.42% |
| Rotation -10 degrees | 98.54% |
| Gaussian noise (standard deviation 0.12) | 99.21% |
| Brightness multiplied by 0.8 | 99.27% |

For these specific transformations, the recorded run showed limited sensitivity to the tested noise and brightness change and a larger accuracy decrease under rotation. These synthetic checks do not establish robustness to real mail images, and the results should not be generalized beyond the tested MNIST transformations.

### Confidence

The evaluation code compares mean maximum-softmax confidence for correct and incorrect predictions and counts errors above a 0.90 confidence threshold. Softmax scores are not a calibration assessment. The project does not include a reliability diagram, calibration metric, or measured review-queue coverage at that threshold.

## Conceptual Operational Considerations

For a future, hypothetical digit-assistance workflow, confidence-based review could be evaluated as one option. A threshold such as 0.90 would be an initial experiment setting, not a validated policy: calibration, error costs, and the fraction of cases routed to review would need to be measured on representative domain data. Any monitoring, retraining, or service architecture would likewise require separate design and validation.

Possible follow-up work, if the project were extended, includes obtaining appropriately governed domain-specific cropped-digit data, testing realistic image-quality variations, assessing calibration and selective prediction, and evaluating a fresh untouched holdout. This repository does not implement a production API, cloud pipeline, monitoring service, or automated retraining.

## Limitations

- MNIST images are standardized and centered; the experiment does not address envelope-level localization, segmentation, ZIP-code sequence parsing, scanner variation, or postal handwriting distributions.
- The target and workflow are hypothetical. No business-impact or cost-benefit analysis was measured.
- Test scores are accessed by multiple scripts and configurations, limiting their role as a single final unbiased comparison.
- Deep-learning training is not guaranteed to be bit-for-bit reproducible across environments; the CNN training script does not set all random seeds.
- Reported historical metrics and figures are not stored as run artifacts in this repository. The source scripts print metrics and generate selected plots when run.

## Skills Demonstrated

- Applying a CRISP-DM structure to a self-directed portfolio experiment.
- Comparing linear, tree-based, kernel, convolutional, and voting approaches.
- Examining class confusions, selected high-confidence errors, and synthetic perturbations.
- Separating benchmark findings from hypothetical business recommendations.
- Organizing reusable image, MNIST, evaluation, tabular, text, and time-series utilities under `0_utils/`.

## References

- [MNIST dataset and IDX file format (CVDF)](https://github.com/cvdfoundation/mnist)
- [CRISP-DM overview (IBM)](https://www.ibm.com/docs/en/spss-modeler/18.5.0?topic=dm-crisp-help-overview)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [TensorFlow Keras documentation](https://www.tensorflow.org/guide/keras)
