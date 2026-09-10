# UAQE Automated Optimization Plan

**Target Profile:** `Raspberry Pi 5 (Broadcom BCM2712, 4x Cortex-A76)`  
**Optimization Profile:** `balanced`  
**Status:** Read-Only Plan Generated (Awaiting User Approval)

---

## 1. Model & Dataset Specification
- **Model Architecture:** MobileNetV3-Small (ONNX)
- **Parameter Count:** 1,522,050
- **Source Checksum (SHA-256):** `616f53f5ef2a66e7028e075d2af204b20067ededffdf550a7696fdc476242711`
- **Dataset:** dataset (image_folder)
- **Classes:** 9
- **Dataset Partitions:** Train: 877 | Val: 184 | Test: 197

## 2. Adaptation Strategy
- **Adaptation Required:** YES
- **Original Output Classes:** 10
- **Target Dataset Classes:** 9

## 3. Quantization & Compression Plan
- **Planned Precision:** **INT8** (Sensitivity-Aware Post-Training Quantization / QAT)
- **Fallback Used:** NO
- **Pruning Allocation:** [0.1, 0.2]
- **Encoding & Packaging:** Sparse + RLE
- **Selective Clustering:** Disabled
- **Target Export Format:** `tflite`

## 4. Validation Policy
- **Optimization Search Split:** `validation`
- **Final Evaluation Split:** `test` (Frozen test set)
- **Accuracy Gating Tolerance:** $\le 2.0\%$ drop

> [!NOTE]
> Hardware latency figures represent target profile planning constraints and do not represent physical hardware measurements.
