# UAQE Phase D.1: Clean Baseline Evaluation Report

**Model:** `output/phase_c4/models/c4_best_int8.tflite`  
**Evaluation Scope:** 196 Clean Test Images (Excluding Duplicate `test/opens/open133.png`)  
**Execution Timestamp:** 2026-09-04 20:17:10

## 1. Summary Metrics

| Metric | Clean D1-0 Value |
| :--- | :--- |
| **Clean Test Accuracy** | **97.96%** (192/196) |
| **Macro Precision** | 97.86% |
| **Macro Recall** | 98.22% |
| **Macro F1** | 97.94% |
| **Model Size** | 1.77 MB (1,856,832 bytes) |
| **INT8 Tensors** | 284 |
| **INT32 Tensors** | 64 |
| **FP32 Tensors** | 2 (input/output boundaries) |
| **INT8 Tensor Coverage** | 81.1% |
| **Host Latency (Mean)** | 47.12 ms |
| **Host Latency (Median)** | 46.47 ms |
| **Host Latency (P95)** | 50.66 ms |
| **500-Run Stability** | PASS (0 drift, 0 NaN/Inf) |

## 2. Per-Class Accuracy Breakdown

| Class | Support | Correct | Accuracy |
| :--- | :---: | :---: | :---: |
| **bridge** | 25 | 22 | 88.00% |
| **clean** | 12 | 12 | 100.00% |
| **cmp** | 26 | 26 | 100.00% |
| **crack** | 25 | 24 | 96.00% |
| **opens** | 24 | 24 | 100.00% |
| **other** | 12 | 12 | 100.00% |
| **particle** | 23 | 23 | 100.00% |
| **scratch** | 24 | 24 | 100.00% |
| **vias** | 25 | 25 | 100.00% |
