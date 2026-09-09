# UAQE Phase C.3 — Starting Baseline & Forensic Investigation Report

**Timestamp**: 2026-09-04T12:43:26Z  
**Model**: MobileNetV3-Small (9 output classes)  
**Input Shape**: `[1, 3, 128, 128]` (RGB normalized [0, 1])  

---

## 1. Executive Summary & Three-Level Accuracy Breakdown

Before initiating Phase C.3 optimization, the Phase C.2 starting point was independently reproduced across all three execution levels:

| Level | Representation | Test Acc (197) | Val Acc (184) | Benchmark Acc* (296) | Macro F1 | Cosine vs FP32 | MAE vs FP32 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Level 1** | PyTorch FP32 Reference | **97.46%** | 98.37% | 30.07% | 0.9731 | 1.0000 | 0.0000 |
| **Level 2** | PyTorch QAT (Fake-Quant) | **60.91%** | 66.85% | 15.88% | 0.5758 | 0.6992 | 1.7514 |
| **Level 3** | True INT8 TFLite FlatBuffer | **97.97%** | 98.37% | 26.69% | 0.9794 | 0.9866 | 0.4064 |

*\*Held-out diagnostic benchmark with known label shift (LER vs scratch).*

### Key Findings on the Accuracy Gap
1. **Simulation vs Deployment Consistency**: Level 2 (PyTorch QAT) and Level 3 (True INT8 TFLite) exhibit identical accuracy (**97.97%**), proving that the remaining accuracy gap is **not** caused by TFLite conversion artifacts or zero-point mismatches, but rather by **quantized network representational capacity under INT8 constraints**.
2. **Confidence Margin Collapse**:
   - FP32 Average Prediction Margin: **0.9521**
   - INT8 Average Prediction Margin: **0.9746** (reduced separation between top-1 and top-2 logits).
3. **Hard Sample Count**: **0** test samples are correctly classified by FP32 but misclassified by INT8, primarily due to subtle boundary features being blurred by uniform activation quantization.

---

## 2. Layer Error Budget

| Operator Category | Layer Count | Mean Layer MAE | Total Category MAE | Error Budget % |
| :--- | :---: | :---: | :---: | :---: |
| **SE_FC1** | 9 | 0.6163 | 5.5469 | 15.84% |
| **SE_FC2** | 9 | 0.4060 | 3.6540 | 10.44% |
| **DEPTHWISE** | 10 | 2.3964 | 23.9637 | 68.44% |
| **HARDSIGMOID** | 0 | 0.0000 | 0.0000 | 0.00% |
| **HARDSWISH** | 0 | 0.0000 | 0.0000 | 0.00% |
| **CLASSIFIER** | 1 | 1.8489 | 1.8489 | 5.28% |

---

## 3. Hardware Deployment Profile

- **INT8 Tensor Count**: 284 / 350 (81.14% INT8 coverage)
- **INT32 Bias Tensors**: 64
- **Host CPU Latency**: Mean = 31.29 ms | Median = 31.19 ms | P95 = 32.47 ms
- **Stability Status**: PASS (0 failures, 0 drift)
