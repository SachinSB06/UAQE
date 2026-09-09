# UAQE Phase C.1: 9-Class MobileNetV3 Model Reconstruction & FP32 Baseline Report

## Executive Summary
This report documents the successful execution of **Phase C.1 — Trainable 9-Class MobileNetV3 Reconstruction**. The original 10-class ONNX semiconductor defect model (`src/models/mobilenetv3_sem.onnx`) was forensically analyzed, proved to be structurally equivalent to `torchvision.models.mobilenet_v3_small`, and reconstructed in PyTorch. 

All compatible feature-extractor and backbone weights (52 Conv layers + SE blocks + `classifier.0`) were transferred and validated with numerical equivalence against ONNX Runtime. A deterministic 9-class classifier matching the full semiconductor dataset (`D:\semiconductor_dataset\dataset`) was initialized and fine-tuned, establishing a reproducible FP32 baseline ready for Quantization-Aware Training (QAT).

---

## 1. Source ONNX Model vs Target PyTorch Architecture
* **Source ONNX**: `src/models/mobilenetv3_sem.onnx` (Opset 13, IR 7, 10-class output, 52 Conv nodes with fused BatchNorm).
* **Target Architecture**: `torchvision.models.mobilenet_v3_small(num_classes=9)`
* **Input Signature**: `[batch, 3, 128, 128]` (Float32, Normalized `[0, 1]`, NCHW layout)
* **Total Parameters**: 1,527,081 parameters (~3.74 MB FP32 state dict)

---

## 2. 10-Class Numerical Equivalence Test
To verify the structural and numerical fidelity of the PyTorch reconstruction before modifying the classifier head, a 10-class PyTorch reference model was initialized with transferred ONNX weights and tested against ONNX Runtime across random and deterministic input tensors:

| Metric | Target Threshold | Measured Value | Status |
|:---|:---:|:---:|:---:|
| **Cosine Similarity** | $> 0.99999$ | **0.9999999881** | **VERIFIED** |
| **Mean Absolute Error (MAE)** | $< 1.0\times 10^{-5}$ | **6.4639e-06** | **VERIFIED** |
| **Root Mean Squared Error (RMSE)** | $< 1.0\times 10^{-5}$ | **8.6675e-06** | **VERIFIED** |
| **Max Absolute Error** | $< 1.0\times 10^{-4}$ | **6.6757e-05** | **VERIFIED** |

---

## 3. Deterministic 9-Class Mapping
The single source of truth mapping for all 9 physical defect categories in `D:\semiconductor_dataset\dataset` is saved in `output/phase_c1\class_mapping.json`:

```json
{
  "0": "bridge",
  "1": "clean",
  "2": "cmp",
  "3": "crack",
  "4": "opens",
  "5": "other",
  "6": "particle",
  "7": "scratch",
  "8": "vias"
}
```

---

## 4. ONNX $\rightarrow$ PyTorch Weight Transfer Summary
* **Transferred Tensors**: **106** (52 Conv weights, 52 folded BN biases/running parameters, `classifier.0.weight`, `classifier.0.bias`)
* **Newly Initialized Tensors**: **2** (`classifier.3.weight` `[9, 1024]`, `classifier.3.bias` `[9]`)
* **Unmatched Tensors**: **0**
* Detailed tensor-by-tensor mapping logged in `output/phase_c1\weight_transfer_report.csv`.

---

## 5. Training & Fine-Tuning Configuration
* **Dataset Root**: `D:\semiconductor_dataset\dataset`
* **Training Samples**: 877 images across 9 classes
* **Validation Samples**: 184 images across 9 classes
* **Test Samples**: 197 images across 9 classes
* **Optimizer**: `AdamW` (lr=3e-4, weight_decay=1e-4)
* **Scheduler**: `CosineAnnealingLR` (T_max=20, eta_min=1e-6)
* **Loss Function**: `CrossEntropyLoss`
* **Random Seed**: 42
* **Best Epoch**: Epoch 20

---

## 6. FP32 Model Evaluation Results

### Overview Across Splits
| Dataset Split | Total Samples | Correct | Accuracy | Macro Precision | Macro Recall | Macro F1 | Host Latency (ms/sample) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Training** | 877 | 876 | **99.89%** | 99.90% | 99.90% | 99.90% | 1.25 ms |
| **Validation** | 184 | 183 | **99.46%** | 99.15% | 99.49% | 99.30% | 1.96 ms |
| **Full Test** | 197 | 193 | **97.97%** | 97.86% | 98.22% | 97.94% | 2.07 ms |
| **296 Benchmark** | 296 | 88 | **29.73%** | 35.56% | 28.72% | 28.56% | 2.04 ms |

---

## 7. Per-Class Test Set Performance
| Class | Index | Support | Correct | Accuracy | Precision | Recall | F1 Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **bridge** | 0 | 25 | 22 | 88.00% | 100.00% | 88.00% | 93.62% |
| **clean** | 1 | 12 | 12 | 100.00% | 100.00% | 100.00% | 100.00% |
| **cmp** | 2 | 26 | 26 | 100.00% | 100.00% | 100.00% | 100.00% |
| **crack** | 3 | 25 | 24 | 96.00% | 100.00% | 96.00% | 97.96% |
| **opens** | 4 | 25 | 25 | 100.00% | 100.00% | 100.00% | 100.00% |
| **other** | 5 | 12 | 12 | 100.00% | 92.31% | 100.00% | 96.00% |
| **particle** | 6 | 23 | 23 | 100.00% | 88.46% | 100.00% | 93.88% |
| **scratch** | 7 | 24 | 24 | 100.00% | 100.00% | 100.00% | 100.00% |
| **vias** | 8 | 25 | 25 | 100.00% | 100.00% | 100.00% | 100.00% |

---

## 8. Benchmark Isolation & Label Handling
The 296-image benchmark dataset (`datasets/hackathon_test_dataset`) was strictly isolated from training, optimization, and checkpoint selection. The benchmark contains folder `LER`, which corresponds to `scratch` in the full dataset taxonomy.

---

## 9. Checkpoint Artifacts
1. **PyTorch Checkpoint**: `output/phase_c1\models\mobilenetv3_sem_9class_fp32.pth` (5.98 MB)
2. **Metadata JSON**: `output/phase_c1\models\mobilenetv3_sem_9class_fp32_metadata.json`
3. **Exported 9-Class ONNX**: `output/phase_c1\models\mobilenetv3_sem_9class_fp32.onnx` (5.84 MB)
4. **Weight Transfer CSV**: `output/phase_c1\weight_transfer_report.csv`
5. **Class Mapping JSON**: `output/phase_c1\class_mapping.json`

---

## 10. QAT Readiness Assessment
* **Status**: `READY`
* Trainable PyTorch model architecture verified and tested.
* Full weight transfer from legacy ONNX confirmed with $>0.99999$ cosine similarity.
* Reproducible training pipeline established with honest FP32 baseline.
* Clean checkpoint and ONNX export available for Phase C.2 (Quantization-Aware Training).
