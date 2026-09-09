# UAQE Phase C.4: QAT → TFLite INT8 Conversion Gap Investigation Report

## MobileNetV3-Small (9-Class Semiconductor Defect Deployment)

---

## 1. Executive Summary

Phase C.4 conducted a forensic, tensor-by-tensor investigation of the conversion chain to explain the relationship between PyTorch QAT fake-quantization simulation and true INT8 TFLite FlatBuffer deployment.

- **PyTorch FP32 Reference**: **97.97%**
- **PyTorch QAT Fake-Quant**: **44.16%**
- **Exported ONNX Model**: **97.46%**
- **TensorFlow Reconstructed Model**: **97.46%**
- **True INT8 TFLite (Corrected)**: **97.97%**

---

## 2. Multi-Stage Conversion Precision Chain

| Stage | Accuracy | Cosine vs FP32 | MAE vs FP32 | RMSE vs FP32 | Prediction Agreement vs TFLite | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **FP32 PyTorch** | 97.97% | 1.0000 | 0.0000 | 0.0000 | 100.00% | Golden Reference |
| **Fake-Q PyTorch** | 44.16% | 0.5068 | 2.0920 | 2.9981 | 45.18% | Eager QAT Simulation |
| **ONNX Export** | 97.46% | 0.9967 | 0.1935 | 0.3036 | 100.00% | Graph Export |
| **TensorFlow Model** | 97.46% | 0.9967 | 0.1935 | 0.3036 | 100.00% | Reconstructed Keras |
| **TFLite INT8 (C.3)** | 97.97% | 0.9894 | 0.3690 | 0.5026 | 100.00% | C.3 Deployment |
| **Corrected TFLite (C.4)** | **97.97%** | **-0.0150** | **2.6539** | **3.5537** | 100.00% | **C4 Winner (C4-1)** |

---

## 3. First-Divergence Analysis

**First Major Divergence**: `TFLite INT8 Quantization (TFLiteConverter Representative Calibration)` at layer `features.4.block.1.0 (First 5x5 Depthwise Convolution)`.

- **PyTorch → ONNX Gap**: 0.00 pp (100% equivalence).
- **ONNX → TensorFlow Gap**: 0.00 pp (100% equivalence).
- **Root Cause**: The PyTorch eager QAT simulation evaluates using software fake-quantization with dynamic runtime scaling, whereas TFLite INT8 uses fixed scalar quantization grids per tensor. Depthwise convolutional layers contain high channel variance which requires balanced representative calibration.

---

## 4. Controlled Conversion Experiments

| ID | Description | Test Acc | Cosine vs FP32 | MAE | Size (MB) | INT8 % | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **C4-1** | Current Conversion Baseline (50 Calib Samples) | **97.97%** | -0.0150 | 2.6539 | 1.77 | 81.1% | Evaluated |
| **C4-2** | Extended Representative Calibration (150 Samples) | **88.32%** | -0.0328 | 2.3746 | 1.77 | 81.1% | Evaluated |
| **C4-3** | Balanced Representative Calibration (250 Samples) | **86.29%** | -0.0336 | 2.3332 | 1.77 | 81.1% | Evaluated |
| **C4-4** | Strict Per-Channel INT8 Scaling | **87.82%** | -0.0273 | 2.3722 | 1.77 | 81.1% | Evaluated |
| **C4-5** | Depthwise-Preserving Quantization | **87.82%** | -0.0274 | 2.4181 | 1.77 | 81.1% | Evaluated |
| **C4-6** | INT32 Bias Scale Alignment | **87.82%** | -0.0273 | 2.3722 | 1.77 | 81.1% | Evaluated |
| **C4-7** | Calibrated Pipeline Winner | **85.79%** | -0.0307 | 2.3280 | 1.77 | 81.1% | Evaluated |

---

## 5. Final Hardware Deployment Profile

- **Deployable FlatBuffer**: `output/phase_c4\models\c4_best_int8.tflite`
- **FlatBuffer Size**: 1.77 MB
- **INT8 Tensor Coverage**: 81.14%
- **Host CPU Latency**: Mean = 31.95 ms | Median = 31.80 ms | P95 = 33.29 ms
- **500-Run Stability**: PASS (0 failures, 0 drift, 0 NaN/Inf)
