# FP32 Baseline Accuracy Diagnosis Report

This report documents the systematic validation and diagnosis of the FP32 baseline reference model accuracy within the UAQE host-side pipeline.

---

## A. Codebase Findings
*   The pipeline code matches the model input/output contract perfectly.
*   The dataset loader in `real_dataset_adapter.py` correctly reads, transposes, and scales inputs for the model under NCHW layout.
*   No execution bugs or shape mismatch handling errors were found in the evaluator codebase.

## B. Dataset Findings
*   **Total classes on disk**: 9
*   **Total valid images**: 296
*   **Grayscale images**: 0 (all images converted to RGB mode standard)
*   **RGB images**: 296
*   **Extensions**: `.png`
*   **Corrupt files**: 0
*   Detailed counts are saved in [dataset_report.json](file:///D:/Quantization%20embedded/reports/fp32_accuracy_diagnosis/dataset_report.json).

## C. Class Mapping Findings
*   **Status**: `VERIFIED`
*   The mapping correctly uses class indices `0, 1, 2, 3, 4, 5, 6, 8, 9`. 
*   Index `7` is unused since no corresponding directory is present in the hackathon test dataset.
*   This mapping matches the output dimensions of the model (10 classes).

## D. Preprocessing Findings
*   **Status**: `VERIFIED`
*   The preprocessing matches the model expectation: resize to `128x128` (bilinear interpolation), color channels in `RGB`, layout transposed to `NCHW`, normalized via `rgb_0_1` (scaled to `[0, 1]` range by dividing by 255.0).

## E. Model Input/Output Findings
*   **Input Name**: `input`
*   **Input Shape**: `['batch', 3, 128, 128]` (NCHW)
*   **Input Dtype**: `float32` (elem_type = 1)
*   **Output Name**: `output`
*   **Output Shape**: `['batch', 10]`
*   **Output Dtype**: `float32`
*   Evaluator configuration and logic align perfectly with this contract.

## F. Independent FP32 Accuracy
*   **Measured Accuracy**: **37.162%** (110 / 296 correct predictions).

## G. Existing Evaluator FP32 Accuracy
*   **Measured Accuracy**: **37.16%**

## H. Difference Between Them
*   **Difference**: **0.00 pp** (completely identical, verifying the correctness of the evaluator implementation).

## I. Per-Class Performance
*   Clean (idx 0): **63.64%** (21 / 33)
*   Bridge (idx 1): **43.75%** (14 / 32)
*   CMP (idx 2): **40.00%** (12 / 30)
*   Crack (idx 3): **38.71%** (12 / 31)
*   LER (idx 4): **30.00%** (9 / 30)
*   Open (idx 5): **16.67%** (5 / 30)
*   Particle (idx 6): **30.00%** (9 / 30)
*   VIA (idx 8): **26.67%** (8 / 30)
*   Other (idx 9): **40.00%** (20 / 50)

## J. Prediction Distribution
*   Ground-truth vs predicted distribution is saved in [prediction_distribution.json](file:///D:/Quantization%20embedded/reports/fp32_accuracy_diagnosis/prediction_distribution.json). No output collapse or extreme class imbalance was observed, though some classes (e.g. LER and Open) are frequently misclassified as Other/Clean.

## K. Golden-Sample Results
*   Saved in [golden_samples.json](file:///D:/Quantization%20embedded/reports/fp32_accuracy_diagnosis/golden_samples.json) containing filename, shape, logits, and top-3 scores for 8 representative samples across classes.

## L. Model/Dataset Compatibility
*   **Status**: `VERIFIED`
*   The model class output shape matches the configured expected class mapping, and the preprocessing modes match.

## M. Root-Cause Assessment
*   **Primary Root Cause**: `GENUINELY LOW FP32 ACCURACY`
*   The low measured FP32 baseline accuracy (37.16%) is correct and genuine. The base model has poor general classification performance on this dataset.

## N. Confidence Level
*   **Confidence**: `HIGH`

## O. Recommended Next Action
*   Since the baseline model itself is genuinely weak, the correct path forward before Phase 6 hardware deployment is to acquire legitimate training/validation datasets to fine-tune/retrain the base model, or replace the base model with a higher-accuracy reference checkpoint.
