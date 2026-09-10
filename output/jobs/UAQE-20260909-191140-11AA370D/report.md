# UAQE Autonomous Optimization Report — Job UAQE-20260909-191140-11AA370D

- **Model Architecture:** MobileNetV3-Small
- **Dataset:** dataset
- **Target Hardware:** Raspberry Pi 5 (Broadcom BCM2712, 4x Cortex-A76)
- **Optimization Profile:** balanced
- **Selected Strategy:** MobileNetV3 Static INT8 PTQ (mobilenet_adaptive)
- **Final Verdict:** **VERIFIED**

## Quantitative Benchmark Results

| Metric | Reference (FP32) | Optimized (MobileNetV3 Static INT8 PTQ) | Empirical Gain / Delta |
|:---|:---:|:---:|:---:|
| **Top-1 Accuracy** | 96.95% | **95.94%** | **-1.02%** (ACCEPTABLE) |
| **Macro F1 Score** | 96.55% | **95.29%** | -1.26% |
| **Model Disk Size** | 6,122,714 B (5.8 MB) | **1,855,816 B (1.8 MB)** | **-69.69% reduction** |
| **Host Latency (mean)** | 6.65 ms | **149.90 ms** | **+-2153.11% speedup** |
| **Throughput** | 150.31 img/s | **6.67 img/s** | +-95.6% |
| **Prediction Agreement** | 100.0% | **98.98%** | — |

*Note on Latency:* Measurements performed on host test environment. Hardware validation for physical target pending.

## Autonomous Candidate Search & Pareto Summary

- **Total Candidates Evaluated:** 4
- **Maximum Candidate Budget:** 10
- **Stopping Reason:** NO_LEGAL_CANDIDATES_REMAIN (Candidate search space fully explored. Best safe candidate selected.)
- **Best Candidate Objective Score:** 0.668989
- **Accuracy Constraint:** SATISFIED (Safety limit: 1.02 pp <= 4.00 pp)
- **Safe Fallback Retained:** NO

### Evaluated Candidates Summary

- **cand_001 (MobileNetV3 Static INT8 PTQ):** Acc=95.94% (+1.02 pp, ACCEPTABLE), Size=1.77 MB, Latency=149.90 ms, Score=0.6690 [SELECTED]
- **cand_002 (MobileNetV3 INT8 PTQ + Magnitude Pruning):** Acc=95.94% (+1.02 pp, ACCEPTABLE), Size=1.77 MB, Latency=142.96 ms, Score=0.6690 [REJECTED]
- **cand_003 (MobileNetV3 INT8 + Adaptive Sparse RLE Packaging):** Acc=95.94% (+1.02 pp, ACCEPTABLE), Size=1.77 MB, Latency=140.34 ms, Score=0.6690 [REJECTED]
- **cand_004 (MobileNetV3 XNNPACK-Compatible INT8):** Acc=95.43% (+1.52 pp, ACCEPTABLE), Size=1.77 MB, Latency=9.82 ms, Score=0.6664 [REJECTED]