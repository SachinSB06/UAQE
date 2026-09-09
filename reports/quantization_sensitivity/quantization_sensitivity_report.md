# UAQE Phase A.2 — Quantization Error & Layer Sensitivity Analysis Report

- **Timestamp**: `2026-09-04T03:14:53Z`
- **Analysis Duration**: `3.97 s`
- **Reference ONNX Model**: `src/models/mobilenetv3_sem.onnx`
- **Deployed INT8 TFLite Model**: `src/outputs/exports/raspberrypi5/model.tflite`
- **Calibration & Evaluation Samples**: `50 deterministic samples`
- **Sensitivity Formula**: `Score = 0.40 * (1 - ActCosine) + 0.30 * min(1.0, ActMAE / (ActScale + 1e-6)) + 0.20 * (1 - WeightCosine) + 0.10 * ClippingRate`

---

## 1. Executive Summary & Root Cause Determination

Overall FP32 Accuracy: **37.16%** | Full INT8 TFLite Accuracy: **21.96%** (Delta: **-15.2 pp**)
Overall Prediction Agreement: **34.12%**

### Empirical Findings:
1. **Weight Quantization**: TFLiteConverter applied per-channel quantization on weights. Average weight cosine similarity across all layers was high (>0.99). Weight quantization is **NOT** the primary driver of accuracy collapse.
2. **Activation Quantization & Dynamic Range Discretization**: Intermediate activations were quantized using per-tensor uniform scaling. Discretization noise accumulates progressively across deep inverted residual blocks.
3. **First Major Error Spike**: Occurred at **features.1** with a cosine similarity drop of **0.1877**.
4. **Largest Error Spike**: Occurred at **features.5** with a cosine similarity drop of **0.2498**.
5. **Depthwise Convolutions Impact**: 11 depthwise layers showed mean activation cosine of **0.7659** and mean MAE of **0.5280** (Major Error Source: **True**).
6. **SE Blocks Impact**: 47 SE layers showed mean activation cosine of **0.6471** (Major Error Source: **True**).

---

## 2. Top 10 Most Sensitive Layers

| Rank | Layer Name | Block | Operator | Category | Act Cosine | Act MAE | W Cosine | Sensitivity | Recommendation |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `features/features.11/block/block.2/` | features.11 | Conv | se_fc2 | -0.4492 | 2.3616 | 1.0000 | **0.8798** | `QAT CANDIDATE` |
| 2 | `features/features.10/block/block.2/` | features.10 | Conv | se_fc2 | -0.3553 | 2.3644 | 1.0000 | **0.8423** | `QAT CANDIDATE` |
| 3 | `features/features.6/block/block.2/f` | features.6 | Conv | se_fc2 | -0.2002 | 1.7846 | 1.0000 | **0.7805** | `QAT CANDIDATE` |
| 4 | `features/features.8/block/block.2/f` | features.8 | Conv | se_fc2 | -0.1228 | 1.8927 | 1.0000 | **0.7500** | `QAT CANDIDATE` |
| 5 | `features/features.9/block/block.2/f` | features.9 | Conv | se_fc2 | 0.0241 | 1.5451 | 1.0000 | **0.6926** | `QAT CANDIDATE` |
| 6 | `features/features.1/block/block.1/f` | features.1 | Conv | se_fc2 | 0.2150 | 0.5815 | 1.0000 | **0.6202** | `QAT CANDIDATE` |
| 7 | `features/features.5/block/block.2/f` | features.5 | Conv | se_fc2 | 0.2458 | 1.9573 | 1.0000 | **0.6042** | `QAT CANDIDATE` |
| 8 | `classifier/classifier.1/Mul` | classifier | Mul | hardswish | 0.4075 | 0.6192 | 1.0000 | **0.5496** | `QAT CANDIDATE` |
| 9 | `features/features.5/block/block.1/b` | features.5 | Mul | hardswish | 0.4430 | 0.3214 | 1.0000 | **0.5490** | `QAT CANDIDATE` |
| 10 | `features/features.11/block/block.3/` | features.11 | Conv | project_conv | 0.3925 | 0.6308 | 1.0000 | **0.5430** | `QAT CANDIDATE` |

---

## 3. Top 5 Most Sensitive MobileNetV3 Blocks

| Block | Layer Count | Mean Cosine | Mean MAE | Max Sensitivity | Cosine Drop from Prev |
|---|---|---|---|---|---|
| **features.11** | 10 | 0.4785 | 0.6034 | **0.8798** | 0.0470 |
| **features.10** | 10 | 0.5255 | 0.5174 | **0.8423** | 0.1189 |
| **features.6** | 10 | 0.5728 | 0.5966 | **0.7805** | 0.0000 |
| **features.8** | 10 | 0.6056 | 0.4886 | **0.7500** | 0.1573 |
| **features.9** | 9 | 0.6444 | 0.4487 | **0.6926** | 0.0000 |

---

## 4. Error Accumulation Progression Across Graph

| Stage / Block | Layer Count | Mean Cosine | Mean MAE | Max Sensitivity | Cosine Drop |
|---|---|---|---|---|---|
| `features.0` | 2 | 0.9969 | 0.1838 | 0.2269 | 0.0031 |
| `features.1` | 6 | 0.8091 | 0.2933 | 0.6202 | 0.1877 |
| `features.2` | 3 | 0.8347 | 0.4237 | 0.4077 | 0.0000 |
| `features.3` | 4 | 0.8058 | 0.5152 | 0.4401 | 0.0289 |
| `features.4` | 9 | 0.8053 | 0.4890 | 0.4674 | 0.0005 |
| `features.5` | 10 | 0.5555 | 0.6014 | 0.6042 | 0.2498 |
| `features.6` | 10 | 0.5728 | 0.5966 | 0.7805 | 0.0000 |
| `features.7` | 9 | 0.7628 | 0.5221 | 0.4879 | 0.0000 |
| `features.8` | 10 | 0.6056 | 0.4886 | 0.7500 | 0.1573 |
| `features.9` | 9 | 0.6444 | 0.4487 | 0.6926 | 0.0000 |
| `features.10` | 10 | 0.5255 | 0.5174 | 0.8423 | 0.1189 |
| `features.11` | 10 | 0.4785 | 0.6034 | 0.8798 | 0.0470 |
| `features.12` | 2 | 0.5102 | 1.2467 | 0.5348 | 0.0000 |
| `avgpool` | 1 | 0.5561 | 0.4291 | 0.4776 | 0.0000 |
| `classifier` | 3 | 0.5063 | 1.0637 | 0.5496 | 0.0499 |

---

## 5. Class-Wise Accuracy Degradation (296 Samples)

| Class Index | Class Name | Samples | FP32 Accuracy (%) | INT8 Accuracy (%) | Accuracy Delta (pp) |
|---|---|---|---|---|---|
| 6 | **Particle** | 30 | 30.0% | 3.33% | **-26.67 pp** |
| 0 | **Clean** | 33 | 63.64% | 42.42% | **-21.21 pp** |
| 2 | **CMP** | 30 | 40.0% | 20.0% | **-20.0 pp** |
| 9 | **Other** | 50 | 40.0% | 22.0% | **-18.0 pp** |
| 8 | **VIA** | 30 | 26.67% | 10.0% | **-16.67 pp** |
| 3 | **Crack** | 31 | 38.71% | 22.58% | **-16.13 pp** |
| 5 | **Open** | 30 | 16.67% | 6.67% | **-10.0 pp** |
| 1 | **Bridge** | 32 | 43.75% | 37.5% | **-6.25 pp** |
| 4 | **LER** | 30 | 30.0% | 30.0% | **0.0 pp** |

---

## 6. Actionable Optimization Strategy Recommendations

Based on the empirical measurements:
- **Weight Quantization**: Weights are already per-channel quantized in TFLite FlatBuffer with minimal loss (>0.99 cosine).
- **Primary Problem**: Uniform per-tensor post-training quantization on **intermediate activations** collapses feature variance in depthwise convolutions and SE multiplier gates.
- **Recommended Next Steps**:
  1. **Selective Mixed Precision / FP16**: Keep depthwise convolution outputs and SE gating operations in FP16/FP32 while keeping pointwise projections and weights in INT8.
  2. **Quantization-Aware Training (QAT)**: Fine-tune the MobileNetV3 model with simulated quantization to allow depthwise layers and non-linearities to adapt to discrete INT8 representation.
