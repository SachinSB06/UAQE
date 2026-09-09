# UAQE Phase C.5: Independent Verification Report

Timestamp: 2026-09-04T13:47:12Z

---

## Summary

| Metric | C4 Claim | C5 Independent |
|:---|:---:|:---:|
| TFLite INT8 Accuracy | 97.97% | **97.9695%** |
| FP32 ONNX Accuracy | 97.46% | **97.4619%** |
| FP32/TFLite Agreement | 100.00% | **99.4924%** |
| Logit Cosine Similarity | N/A | **0.986559** |
| Data Leakage | N/A | **CONTAMINATED** |

## Final Verdict

**UNVERIFIED_CRITICAL_ISSUES**

### Critical Issues
- DATA_LEAKAGE: 1 test images in train set

### Warnings
- TFLite accuracy (97.97%) matches C4 claim exactly
- Near-perfect FP32/TFLite agreement (99.49%) despite logit divergence

## Per-Class Accuracy (TFLite INT8)

| Class | Support | Correct | Accuracy |
|:---|:---:|:---:|:---:|
| bridge | 25 | 22 | 88.0% |
| clean | 12 | 12 | 100.0% |
| cmp | 26 | 26 | 100.0% |
| crack | 25 | 24 | 96.0% |
| opens | 25 | 25 | 100.0% |
| other | 12 | 12 | 100.0% |
| particle | 23 | 23 | 100.0% |
| scratch | 24 | 24 | 100.0% |
| vias | 25 | 25 | 100.0% |

## V4 Logit Numerical Fidelity

| Metric | Value |
|:---|:---:|
| cosine_similarity | 0.986559 |
| mae | 0.406392 |
| rmse | 0.563188 |
| max_error | 3.995245 |
| top1_argmax_agree | 196 |
| top1_agree_rate | 0.994924 |
| fp32_l2 | 140.2583 |
| tflite_l2 | 144.4629 |
| scale_ratio | 1.03 |
