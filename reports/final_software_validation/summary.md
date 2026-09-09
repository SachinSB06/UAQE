# Pre-Phase-6 UAQE Software Validation Report
        
This report provides the host-side software validation metrics for UAQE Phases 1 to 5.

- **Run ID**: `run_1788494449`
- **Start Time**: `2026-09-04T04:00:49.072459Z`
- **End Time**: `2026-09-04T04:01:28.822635Z`

## 1. Dataset Manifest
- **Dataset Path**: `D:\Quantization embedded\datasets\hackathon_test_dataset`
- **Total Labeled Samples**: 296
- **Corrupt Images Skipped**: 0
- **Unmapped Folders Skipped**: 0
- **Class Distribution**:
  | Class | Samples Count | Target Class Index |
  |---|---|---|
  | Bridge | 32 | 1 |
  | CMP | 30 | 2 |
  | Clean | 33 | 0 |
  | Crack | 31 | 3 |
  | LER | 30 | 4 |
  | Open | 30 | 5 |
  | Other | 50 | 9 |
  | Particle | 30 | 6 |
  | VIA | 30 | 8 |

## 2. Model Identification
| Model stage | File path | SHA-256 Hash |
|---|---|---|
| FP32 Reference | `D:\Quantization embedded\src\models\mobilenetv3_sem.onnx` | `616f53f5ef2a66e7028e075d2af204b20067ededffdf550a7696fdc476242711` |
| Optimized ONNX | `D:\Quantization embedded\src\outputs\exports\raspberrypi5\model.onnx` | `fb9adfc96f88a5b155380a57e8ea3659fa3ca81bb2e4b19376a1133ef36a3025` |
| TFLite compiled | `D:\Quantization embedded\src\outputs\exports\raspberrypi5\model.tflite` | `7dc82e7e5dbf1a84e42911a897dacce362cff55ab017dcbf394cf3665bd16f85` |

## 3. Model Preprocessing & Contract
- **Input shape (discovered)**: `[1, 3, 128, 128]`
- **Input dtype (discovered)**: `<class 'numpy.float32'>`
- **Input layout**: `NCHW`
- **Output shape (discovered)**: `[1, 10]`
- **Output dtype (discovered)**: `<class 'numpy.float32'>`
- **Output classes count**: `10`
- **Logical Preprocessing mode**: `rgb_0_1`

## 4. Real Accuracy
- **FP32 Reference Model Accuracy**: `37.16%`
- **Optimized ONNX Model Accuracy**: `36.82%`
- **Final TFLite Model Accuracy**: `21.96%`

**Accuracy Deltas**:
- **ONNX vs FP32 Delta**: `-0.34 percentage points`
- **TFLite vs FP32 Delta**: `-15.20 percentage points`
- **TFLite vs Optimized ONNX Delta**: `-14.86 percentage points`

## 5. Classification Report (TFLite)
### Performance Metrics per Class:
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Bridge | 0.324 | 0.375 | 0.348 | 32 |
| CMP | 0.462 | 0.200 | 0.279 | 30 |
| Class_7 | 0.000 | 0.000 | 0.000 | 0 |
| Clean | 0.219 | 0.424 | 0.289 | 33 |
| Crack | 0.200 | 0.226 | 0.212 | 31 |
| LER | 0.161 | 0.300 | 0.209 | 30 |
| Open | 0.667 | 0.067 | 0.121 | 30 |
| Other | 0.169 | 0.220 | 0.191 | 50 |
| Particle | 1.000 | 0.033 | 0.065 | 30 |
| VIA | 0.188 | 0.100 | 0.130 | 30 |
| **Macro Average** | 0.339 | 0.195 | 0.184 | 296 |
| **Weighted Average** | 0.360 | 0.220 | 0.206 | 296 |

### Confusion Matrix:
```
[[14, 1, 3, 2, 5, 0, 0, 1, 2, 5], [6, 12, 1, 2, 8, 0, 0, 0, 0, 3], [5, 1, 6, 4, 6, 0, 0, 0, 0, 8], [8, 2, 0, 7, 1, 0, 0, 0, 2, 11], [14, 4, 0, 1, 9, 0, 0, 0, 0, 2], [6, 6, 0, 0, 5, 2, 0, 3, 1, 7], [2, 5, 1, 4, 2, 1, 1, 1, 1, 12], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 1, 0, 6, 12, 0, 0, 0, 3, 6], [7, 5, 2, 9, 8, 0, 0, 1, 7, 11]]
```

## 6. Numerical Comparison (FP32 vs TFLite)
| Metric | Acceptance Threshold | Measured Value | Status |
|---|---|---|---|
| Cosine Similarity | 0.95 | 0.5310 | **FAIL** |
| Mean Absolute Error (MAE) | 0.15 | 1.1633 | **FAIL** |
| RMSE | 0.2 | 1.4837 | **FAIL** |
| Prediction Agreement | 0.9 | 0.3412 | **FAIL** |

- **Logits NaNs encountered**: `0`
- **Logits Infs encountered**: `0`

## 7. Structural Audit (model.tflite)
- **Total Tensors**: `350`
- **FLOAT32 Tensors**: `2`
- **FLOAT16 Tensors**: `0`
- **INT8 Tensors**: `284`
- **INT32 Tensors**: `64`
- **Quantized Tensors**: `230`
- **Total Operators**: `204`
- **Operator Types Count**:
  - `ADD`: 25
  - `CONV_2D`: 41
  - `DEPTHWISE_CONV_2D`: 11
  - `DEQUANTIZE`: 1
  - `FULLY_CONNECTED`: 2
  - `MEAN`: 10
  - `MINIMUM`: 28
  - `MUL`: 47
  - `PACK`: 1
  - `PAD`: 5
  - `QUANTIZE`: 1
  - `RELU`: 28
  - `RESHAPE`: 1
  - `SHAPE`: 1
  - `STRIDED_SLICE`: 1
  - `TRANSPOSE`: 1

## 8. Model File Size Audit
- **FP32 reference size**: `5.84 MB`
- **Optimized ONNX size**: `2.17 MB`
- **Compiled TFLite size**: `1.77 MB`
- **FP32 to TFLite reduction**: `69.69%`
- **Compression ratio**: `3.30x`

## 9. Performance Benchmark (Single-Threaded)
| Model | Mean Latency (ms) | Median Latency (ms) | p95 Latency (ms) | Throughput (FPS) |
|---|---|---|---|---|
| FP32 Reference | 1.35 | 1.15 | 2.37 | 739.0 |
| Optimized ONNX | 1.86 | 1.45 | 3.45 | 537.9 |
| Compiled TFLite | 2.31 | 2.11 | 3.32 | 432.9 |

- **Host Environment**: CPU: `Intel64 Family 6 Model 186 Stepping 3, GenuineIntel`, RAM: `15.69 GB`, OS: `Windows 11`.
- **Runtimes**: ONNX Runtime `1.28.0`, TensorFlow `2.21.0`.

## 10. Memory Usage
- **Memory Measurement Methodology**: RSS process memory is measured using psutil.Process().memory_info().rss before/after model loading and during inference runs.
- **Process RSS load delta (FP32 ONNX)**: `3880.0 KB`
- **Process RSS load delta (Optimized ONNX)**: `-2220.0 KB`
- **Process RSS load delta (Compiled TFLite)**: `1816.0 KB`
- **Process RSS inference delta (TFLite)**: `1816.0 KB`

## 11. 500-Run stability test
- **Iterations run**: `500`
- **Failures**: `0`
- **Prediction drift occurrences**: `0`
- **Inference time mean**: `2.08 ms`
- **Inference time max**: `6.37 ms`
- **Process RSS growth over 500 runs**: `0.0 KB`

## 12. Regression Test Suite
- **Passed**: 0
- **Failed**: 0
- **Skipped**: 0

## 13. Limitations & Remarks
- Benchmarks are conducted on the host CPU machine under single-threaded control. Execution on target Raspberry Pi 5 hardware (Phase 6) will differ in latency, throughput, and memory properties.
- Process RSS memory delta captures runtime process allocations and is subject to OS memory management/garbage collection; it serves as a proxy metric and does not represent hardware-isolated memory usage.
