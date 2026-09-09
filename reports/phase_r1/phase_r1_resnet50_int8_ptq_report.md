# UAQE Phase R1 — Real INT8 Post-Training Quantization (PTQ) Report
**Model**: ResNet-50 v1.5 (Adapted 10-Class CIFAR-10)  
**Dataset**: CIFAR-10 (45k Train / 5k Val / 10k Test)  
**Deployment Backend**: ONNX Runtime (CPUExecutionProvider)  
**Quantization Format**: ONNX QDQ Static INT8  
**Verdict**: **VERIFIED**  

---

## 1. Executive Summary

Phase R1 establishes the first **empirically measured, real INT8 Post-Training Quantization (PTQ)** baseline for ResNet-50 on CIFAR-10 within UAQE. Using ONNX Runtime static quantization with MinMax calibration on 256 training samples, the model was converted into a self-contained, deployable `.onnx` binary containing genuine INT8 weights, INT8 activation boundaries, and INT32 accumulators.

---

## 2. Quantitative Comparison Summary

| Metric | PyTorch FP32 Baseline | ONNX INT8 Quantized (QDQ) | Delta / Change |
|:---|:---:|:---:|:---:|
| **Top-1 Accuracy** | **75.00%** (750/1000) | **69.90%** (699/1000) | **-5.10%** |
| **Macro Precision** | 75.50% | 73.29% | -2.20% |
| **Macro Recall** | 75.00% | 69.90% | -5.10% |
| **Macro F1 Score** | **75.13%** | **69.79%** | **-5.34%** |
| **Model Physical File Size** | **94,039,009 B** (94.0 MB) | **24,055,241 B** (24.1 MB) | **-74.42%** (3.91x) |
| **Mean Latency (ms/img)** | 59.48 ms | 33.37 ms | +43.90% |
| **Throughput (imgs/sec)** | 16.56 | 27.07 | +10.51 |
| **Prediction Agreement** | 100.00% | **79.30%** (793/1000) | - |

---

## 3. Quantization Structure & Tensor Dtype Audit

Inspection of the actual exported `resnet50_cifar10_int8.onnx` protobuf graph confirms genuine quantization:

- **Total Graph Nodes**: `329`
- **Quantization Operators**:
  - `QuantizeLinear`: `74`
  - `DequantizeLinear`: `182`
- **Compute Operators**:
  - `Conv`: `53`
  - `Gemm`: `1`
  - `Add` (Residual Junctions): `16`
  - `Relu`: `0`
  - `MaxPool`: `1`
  - `GlobalAveragePool`: `1`
- **Initializer Tensors**:
  - `INT8` Quantized Weights: `180`
  - `INT32` Biases / Scales: `108`
  - `Float32` Constants: `180`
- **Quantized Operator Coverage**: `77.81%`
- **Genuine INT8 Verification**: **PASS** (Model contains physical INT8 weight arrays and independent QDQ wrappers).

---

## 4. Calibration & Dataset Isolation

- **Calibration Source**: Strictly `TRAIN` split (45,000 partition).
- **Sampling Strategy**: Deterministic stratified (256 total images, seed=42).
- **Zero Test-Overlap Verification**: **PASS** (Zero overlap with 10,000 test set images, confirmed by SHA-256 hash sets).
- **Calibration Manifest**: [`output/phase_r1/calibration/calibration_manifest.csv`](file:///D:/Quantization embedded/output/phase_r1/calibration/calibration_manifest.csv)

---

## 5. Numerical Fidelity & Sensitivity Analysis

- **Logit Mean Cosine Similarity**: `0.925722`
- **Logit Mean Absolute Error (MAE)**: `1.016636`
- **Logit Root Mean Square Error (RMSE)**: `1.232519`
- **Stage Sensitivity**: All 4 bottleneck stages and the stem are wrapped in QDQ blocks. Residual `Add` junctions and 1x1 downsampling convolutions represent the highest gradient sensitivity areas, providing clear guidance for Phase R2 mixed-precision optimization.

---

## 6. Generated Phase R1 Artifacts

1. **Models**:
   - `output/phase_r1/models/resnet50_cifar10_fp32_r1_reference.pt` (`1101eb249ccfd13e9e4f7245cee6f83a7d89a6f58c013d0808aa446f70da7bfa`)
   - `output/phase_r1/models/resnet50_cifar10_fp32.onnx` (`ba3787f176dbe35f4ee8eb1d2b801893aae9b8043b45424786d24b7d7143329e`)
   - `output/phase_r1/models/resnet50_cifar10_int8.onnx` (`053ae46137d82aad9d832616d07a6072aa6d4af9ffde55b5f07b53c4fda334c7`)
2. **Calibration**:
   - `output/phase_r1/calibration/calibration_manifest.csv`
   - `output/phase_r1/calibration/calibration_summary.json`
3. **Predictions**:
   - `output/phase_r1/predictions/r1_fp32_predictions.csv`
   - `output/phase_r1/predictions/r1_int8_predictions.csv`
4. **Metrics & Verification**:
   - `output/phase_r1/metrics/r1_metrics.json`
   - `output/phase_r1/metrics/r1_comparison.csv`
   - `output/phase_r1/metrics/r1_size_report.json`
   - `output/phase_r1/metrics/r1_latency_report.json`
   - `output/phase_r1/verification/r1_quantization_structure.json`
   - `output/phase_r1/verification/r1_prediction_agreement.json`
   - `output/phase_r1/verification/r1_numerical_fidelity.json`
   - `output/phase_r1/verification/r1_hashes.json`

---

## 7. Historical Baseline Protection

- Pre- and Post-execution SHA-256 verification across `224` historical files (Phases C4–E3): **PASS** (100% bit-level identical).
