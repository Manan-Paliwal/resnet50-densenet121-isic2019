# ResNet50 vs DenseNet121: Skin Lesion Classification

A comparative deep-learning research project using the ISIC 2019 dataset.

**Author:** Manan Paliwal
**Framework:** PyTorch
**Dataset:** ISIC 2019 (25,331 images, 8 classes)
**Status:** Research in progress

## Research Objectives
- Compare ResNet50 and DenseNet121 architectures.
- Evaluate training from scratch and transfer learning.
- Analyze class imbalance and per-class performance.
- Compare accuracy, macro F1, and generalization.

## Experiments
1. ResNet50 trained from scratch
2. DenseNet121 trained from scratch
3. ResNet50 with transfer learning and fine-tuning
4. DenseNet121 with transfer learning and fine-tuning

## Repository Contents
- notebooks/: Experimental Jupyter notebooks
- results/: Training histories, metrics, and experiment configurations

## Verified Test Results

| Model | Test Accuracy | Macro F1 | Test Loss |
|---|---:|---:|---:|
| ResNet50 Scratch | 56.36% | 0.4120 | 3.0361 |
| DenseNet121 Scratch | 49.53% | 0.3491 | 1.5568 |
| ResNet50 Transfer Learning | 66.05% | 0.4623 | 1.8281 |
| DenseNet121 Transfer Learning | Pending | Pending | Pending |

## Reproducibility
The dataset and trained checkpoints are excluded due to their size.
Published notebooks have saved outputs removed and may require path adjustments.

## Disclaimer
For academic research only. Not intended for clinical diagnosis.
