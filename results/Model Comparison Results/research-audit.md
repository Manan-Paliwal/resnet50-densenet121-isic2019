# ISIC 2019 research audit

Six single-seed CNN experiments were evaluated on the same project-defined holdout of 3,808 dermoscopic images. This audit reconciles the saved predictions, metrics, confusion matrices and training records. The largest observed test macro F1 was **50.15%** for DenseNet121 transfer without augmentation (04). ResNet50 transfer with augmentation (05) had the highest observed accuracy (**66.12%**) and validation macro F1 (**53.17%**). The near-ties in test macro F1 among 04–06 and the differences in their training protocols limit conclusions about which approach is better.

## Data and evaluation

The project used the 25,331 labeled images in the ISIC 2019 challenge training set and made its own eight-class split. Evaluation covers these eight known classes, while the official challenge test protocol includes an additional outlier class. [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/)

| Partition | Images | Manifest groups | Images without lesion IDs |
| --- | ---: | ---: | ---: |
| Training | 17,705 | 9,751 | 1,443 |
| Validation | 3,818 | 2,090 | 322 |
| Test | 3,808 | 2,090 | 319 |

Image IDs, manifest groups and recorded lesion IDs did not overlap across partitions. The 2,084 images without a lesion ID use a fallback group per image. SHA-256 checks on the cached JPEG files found 50 groups of byte-identical files within partitions and zero across partitions. Two byte-identical groups with conflicting class labels were entirely within training. The checks establish separation for the recorded IDs and exact cache bytes. Patient independence and near-duplicate separation remain unverified.

The class order is MEL, NV, BCC, AK, BKL, DF, VASC and SCC. Each test image contributes to accuracy. Macro F1 weights each class equally, while weighted F1 weights classes by their test support. For example, the test set has 1,951 NV images, 33 VASC images and 29 DF images. The audit reconciled the full prediction exports, all six raw and normalized confusion matrices, the per-class tables and the reported test metrics. Experiments 01 and 04–06 joined directly by image ID. For experiments 02 and 03, the complete image-keyed error exports recover predictions for incorrect cases, and the remaining cases take their true label. Those reconstructions agreed exactly with the ordered full prediction exports.

## Training settings

All experiments used 224 × 224 model inputs from a 256 × 256 JPEG cache, ImageNet normalization, batch size 32, seed 42 and AdamW with weight decay 0.0001. Class-weighted cross-entropy uses inverse frequencies calculated on the training set. The augmentation runs applied horizontal and vertical flips, rotation up to 20 degrees and color jitter to training images. Validation and test preprocessing used deterministic resize and normalization.

- Experiments 01 and 02 initialized from scratch and trained for up to 25 epochs with a learning rate of 0.0003 and early-stopping patience of five. Experiment 02 stopped after 11 epochs.
- Experiments 03–06 started with ImageNet weights. Stage 1 trained the classifier for up to 10 epochs at 0.001, with early-stopping patience of three and the backbone and its BatchNorm statistics frozen. Stage 2 fine-tuned the full network for up to 15 epochs at 0.00001 for the backbone and 0.0001 for the classifier, with early-stopping patience of four. Experiment 06 stopped Stage 2 after 13 epochs.
- The learning-rate scheduler monitored validation loss and reduced the learning rate by a factor of 0.5 with patience of two. Validation macro F1 selected the best saved checkpoint. Experiment 03 evaluated the best Stage 2 checkpoint. Experiments 04–06 considered both stages and selected Stage 2 in each recorded run.

## Audited results

Scores are percentages. The selected epoch is local to the training stage shown.

| Experiment | Model and recipe | Epochs run | Selected epoch | Validation macro F1 | Test accuracy | Test macro F1 | Test weighted F1 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 01 | ResNet50, scratch | 25 | 23 | 40.46 | 56.36 | 41.20 | 58.29 |
| 02 | DenseNet121, scratch | 11 | 6 | 37.73 | 49.53 | 34.91 | 52.92 |
| 03 | ResNet50, transfer | 10 + 15 | Stage 2, epoch 11 | 49.04 | 66.05 | 46.23 | 66.20 |
| 04 | DenseNet121, transfer | 10 + 15 | Stage 2, epoch 12 | 51.01 | 65.28 | 50.15 | 66.45 |
| 05 | ResNet50, transfer + augmentation | 10 + 15 | Stage 2, epoch 15 | 53.17 | 66.12 | 49.97 | 67.38 |
| 06 | DenseNet121, transfer + augmentation | 10 + 13 | Stage 2, epoch 9 | 52.79 | 63.94 | 50.03 | 65.77 |

The table reports one fitted model per experiment. The paired bootstrap estimates split uncertainty for the audited predictions, using 10,000 percentile replicates and the same resampled groups for both models. Test images remained clustered by manifest group, and groups joined by an exact duplicate were treated as one cluster. The main analysis has 2,089 test clusters. Intervals are pointwise 95% estimates. They do not capture seed-to-seed variation or adjust for multiple comparisons.

| Comparison | Accuracy change, pp (95% CI) | Macro F1 change, pp (95% CI) | Weighted F1 change, pp (95% CI) |
| --- | ---: | ---: | ---: |
| 01 → 02 | -6.83 [-8.77, -4.91] | -6.28 [-9.57, -3.07] | -5.37 [-7.10, -3.65] |
| 01 → 03 | +9.69 [+7.59, +11.85] | +5.03 [+0.95, +9.26] | +7.91 [+5.93, +9.91] |
| 02 → 04 | +15.76 [+13.33, +18.08] | +15.24 [+11.27, +19.34] | +13.53 [+11.32, +15.62] |
| 03 → 05 | +0.08 [-1.41, +1.56] | +3.74 [+0.93, +6.49] | +1.18 [-0.22, +2.58] |
| 04 → 06 | -1.34 [-2.82, +0.20] | -0.13 [-3.40, +3.20] | -0.68 [-2.04, +0.77] |
| 05 → 06 | -2.18 [-3.87, -0.48] | +0.06 [-2.89, +2.85] | -1.61 [-3.17, -0.02] |

Excluding the 319 test images with missing lesion IDs left 3,489 images. Keeping one file per group of exact test duplicates left 3,797. In both sensitivity analyses, the 03 → 05 macro F1 interval remained above zero on the fixed predictions. The 04 → 06 and 05 → 06 macro F1 intervals continued to include zero.

## Per-class test macro components

Each cell shows precision / recall / F1 in percent. Rare-class scores are sensitive to single-image changes.

| Class | Test n | 01 | 02 | 03 | 04 | 05 | 06 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MEL | 648 | 43.7 / 48.3 / 45.9 | 40.6 / 44.4 / 42.4 | 49.7 / 58.0 / 53.5 | 48.5 / 60.2 / 53.7 | 55.4 / 56.2 / 55.8 | 50.0 / 59.6 / 54.4 |
| NV | 1951 | 85.0 / 66.4 / 74.5 | 85.9 / 59.9 / 70.6 | 84.0 / 78.0 / 80.9 | 87.8 / 75.2 / 81.0 | 86.4 / 77.3 / 81.6 | 89.6 / 71.9 / 79.8 |
| BCC | 507 | 44.5 / 57.4 / 50.1 | 36.6 / 38.1 / 37.3 | 59.7 / 70.4 / 64.6 | 59.1 / 62.7 / 60.9 | 65.1 / 56.2 / 60.3 | 62.6 / 55.2 / 58.7 |
| AK | 147 | 25.9 / 34.0 / 29.4 | 20.3 / 62.6 / 30.7 | 37.8 / 36.7 / 37.2 | 29.7 / 38.8 / 33.6 | 36.2 / 51.0 / 42.4 | 34.4 / 44.2 / 38.7 |
| BKL | 404 | 27.8 / 38.6 / 32.3 | 22.7 / 25.2 / 23.9 | 45.7 / 41.8 / 43.7 | 45.8 / 48.5 / 47.1 | 42.7 / 55.4 / 48.3 | 39.3 / 57.2 / 46.6 |
| DF | 29 | 20.0 / 13.8 / 16.3 | 5.3 / 17.2 / 8.1 | 16.7 / 17.2 / 16.9 | 28.9 / 37.9 / 32.8 | 14.9 / 37.9 / 21.4 | 19.5 / 58.6 / 29.3 |
| VASC | 33 | 65.6 / 63.6 / 64.6 | 50.0 / 51.5 / 50.7 | 51.2 / 66.7 / 57.9 | 63.2 / 72.7 / 67.6 | 57.4 / 81.8 / 67.5 | 54.7 / 87.9 / 67.4 |
| SCC | 89 | 15.1 / 18.0 / 16.4 | 11.7 / 23.6 / 15.6 | 19.3 / 12.4 / 15.1 | 23.2 / 25.8 / 24.5 | 20.0 / 25.8 / 22.5 | 23.1 / 28.1 / 25.4 |

## Interpretation and limitations

- **DenseNet121 augmentation (04 → 06).** This is the closest matched augmentation comparison. Validation macro F1 increased by 1.78 points, and the observed test macro F1 changed by −0.13 points. The test interval includes zero. The observed results provide no clear held-out macro F1 gain from augmentation in this single-seed comparison.
- **ResNet50 augmentation (03 → 05).** Test macro F1 increased by 3.74 points in the audited predictions. The training hardware, software and data-loader configuration also changed. The interval quantifies uncertainty for these fitted models, so the gain cannot be attributed to augmentation alone.
- **Architecture comparison (05 → 06).** Both augmented runs used an RTX 3090, but their pretrained weight variants and validation-loss aggregation differ. The notebooks also differ in how explicitly they configure CUDA TF32. The observed test macro F1 difference is 0.06 points and its interval includes zero.
- **Scratch versus transfer.** These comparisons change the optimization schedule as well as the initialization. The DenseNet121 comparison (02 → 04) also changes platforms. Interpret the differences as differences between full training recipes.
- **Loss and timing.** Experiments 01–03 and 05 compute a sample-count average of weighted batch means. Experiments 04 and 06 divide the summed weighted loss by the total target weight across the split. Their per-batch training objective still uses the target-weight denominator. The validation-loss difference can change scheduler behavior, and reported loss values across those definitions are unsuitable for direct comparison. Hardware and timing records differ, so elapsed time is descriptive. [PyTorch cross-entropy reduction](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html)
- **Reproducibility and scope.** Experiments 01–03 ran on Apple MPS with zero data-loader workers and reported PyTorch 2.14.0. Experiments 04–06 ran on an NVIDIA RTX 3090 with four workers, float32 and AMP disabled, using PyTorch 2.14.1+cu132 and TorchVision 0.29.1+cu132. A fixed seed cannot establish repeatability across these platforms. These results measure image-level performance on one dataset holdout. External cohorts, patient-level performance and clinical utility were outside the audit. [PyTorch reproducibility notes](https://docs.pytorch.org/docs/2.15/notes/randomness.html)

A controlled ResNet50 augmentation comparison would require a baseline without augmentation on the same CUDA and software configuration as 05, with the same selection rule. Architecture comparisons should also match loss aggregation and relevant precision settings. Multiple seeds would quantify training variability separately from test-set sampling uncertainty.

Cloud train and validation manifest membership could not be independently recovered from the run export. Test predictions match the permanent test split. The separate dataset staging manifests match the local permanent copies. Historical holdout reuse and incomplete decision logs also limit confirmatory interpretation. No evidence was found in the saved training code of test-set checkpoint selection.


## Published artifact layout

The six [experiment result folders](../) hold the publication copies. Transfer learning histories remain separate by stage. Each experiment includes its own training curves and normalized confusion matrix. The [shared test comparison](test_metrics_comparison.png) uses the recorded test JSONs.

The three legacy classification reports and stage summaries were regenerated from the saved predictions and histories. The 03-vs-05 comparison was regenerated from recorded aggregate outcomes after cloud execution. The original run omitted this comparison because its baseline artifacts were unavailable in that runtime. These are post-run reports. The original cloud artifact and checkpoint names remain in the local run and provenance directories.

[Primary paired intervals](paired_bootstrap_intervals.csv) and [sensitivity intervals](paired_bootstrap_sensitivity.csv) reproduce the saved audit data. The comparison graph uses the primary macro F1 intervals.
