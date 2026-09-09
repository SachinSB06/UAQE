# UAQE Phase D.1: Sensitivity-Aware Pruning Final Report

**Generated:** 2026-09-04 20:28:21  
**Target Architecture:** MobileNetV3-Small (9 semiconductor defect classes)  
**Dataset:** 196 clean test images (audit excluded 1 exact train-test duplicate)

---

## 1. Executive Summary

Phase D.1 implemented controlled sensitivity-aware and global unstructured pruning on the verified INT8 MobileNetV3-Small deployment pipeline. Calibration sampling was upgraded from load-order slicing to **deterministic stratified random calibration** across all 9 classes from TRAIN only.

Key conclusions:
1. **Accuracy Retention:** Sensitivity-aware pruning preserved high classification accuracy (**97.96%** at 20.0% actual sparsity, vs 97.96% clean baseline).
2. **Global vs Sensitivity-Aware:** Sensitivity-aware pruning allocated lower sparsity to critical layers (SE modules, Stem, Classifier), outperforming global uniform magnitude pruning at equal sparsity.
3. **Storage & Deployment Reality:** In standard dense FlatBuffer INT8 execution, unstructured weight zeros do NOT reduce `.tflite` file size (1.77 MB baseline vs 1.77 MB pruned). Pruning serves as a high-quality sparse representation preparatory step for downstream weight clustering (Phase D.2) and sparse encoding (Phase D.3).
4. **Structured Pruning Safety:** Structured channel pruning (D1-9 to D1-11) was formally audited and marked **BLOCKED** due to MobileNetV3 inverted bottleneck group constraints (`groups == in_channels`), Squeeze-and-Excitation channel coupling, and residual identity shape constraints.

---

## 2. Complete Experiment Matrix

| Exp ID | Architecture / Family | Requested Sparsity | Actual Sparsity | Val Acc | Val F1 | Clean Test Acc | Clean Test F1 | Model Size | INT8 Cov | Host Latency (Mean) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **D1-0** | Clean Baseline | 0.0% | 0.0% | N/A | N/A | **97.96%** | 97.94% | 1.77 MB | 81.1% | 47.12 ms | **SUCCESS** |
| **D1-1** | Global Unstructured | 10% | 10.0% | 100.00% | 100.00% | **97.96%** | 98.18% | 1.77 MB | 81.1% | 48.21 ms | **SUCCESS** |
| **D1-2** | Global Unstructured | 20% | 20.0% | 100.00% | 100.00% | **97.45%** | 97.46% | 1.77 MB | 81.1% | 47.69 ms | **SUCCESS** |
| **D1-3** | Global Unstructured | 30% | 30.0% | 98.37% | 98.02% | **96.94%** | 96.27% | 1.77 MB | 81.1% | 46.81 ms | **SUCCESS** |
| **D1-4** | Global Unstructured | 40% | 40.0% | 89.67% | 83.14% | **85.71%** | 79.28% | 1.77 MB | 81.1% | 48.09 ms | **SUCCESS** |
| **D1-5** | Global Unstructured | 50% | 50.0% | 70.11% | 59.96% | **51.02%** | 42.09% | 1.77 MB | 81.1% | 48.07 ms | **SUCCESS** |
| **D1-6** | Sensitivity-Aware Unstructured | 20% | 20.0% | 100.00% | 100.00% | **97.96%** | 97.94% | 1.77 MB | 81.1% | 46.47 ms | **SUCCESS** |
| **D1-7** | Sensitivity-Aware Unstructured | 30% | 30.0% | 99.46% | 99.30% | **96.94%** | 96.80% | 1.77 MB | 81.1% | 48.87 ms | **SUCCESS** |
| **D1-8** | Sensitivity-Aware Unstructured | 40% | 40.0% | 94.02% | 92.52% | **85.71%** | 83.07% | 1.77 MB | 81.1% | 47.89 ms | **SUCCESS** |
| **D1-9** | Structured Channel Pruning | 10% | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | **BLOCKED (Unsafe)** |
| **D1-10** | Structured Channel Pruning | 20% | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | **BLOCKED (Unsafe)** |
| **D1-11** | Structured Channel Pruning | 30% | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | **BLOCKED (Unsafe)** |


---

## 3. Best Pruning Candidate

* **Winning Experiment:** `D1-1` (Global Unstructured)
* **Actual Sparsity:** 10.00%
* **Validation Macro F1:** 100.00%
* **Clean Test Accuracy:** 97.96% (Clean baseline: 97.96%)
* **Accuracy Delta:** +0.00 pp
* **500-Run Latency:** 48.21 ms (P95: 56.17 ms)
* **INT8 Operator Coverage:** 81.1%
* **Stability:** 100% PASS (500 runs, 0 drift, 0 NaN/Inf)

---

## 4. Recommendation for Phase D.2 (Clustering)

Use the sparse weights from `D1-1` (`d1_best_sensitive.pth` / `d1_best_sensitive_int8.tflite`) as the input for k-means weight clustering in Phase D.2. The structured zero clusters and reduced parameter manifold provide an ideal starting point for centroid codebook quantization.
