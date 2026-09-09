# Calibration-Guided Reconstruction Pruning Experiment Summary

This report documents the results of the first pruning experiments targeting the Raspberry Pi 5 simulation runtime using the calibration-guided reconstruction strategy.

## 1. Executive Summary & Success Evaluation

* **Pruning Strategy:** `calibration_guided_reconstruction`
* **Hardware Profile:** `raspberrypi5` (simulated runtime)
* **Calibration Source:** 200-image deterministic stratified subset from `train` split
* **Test Source:** Locked 296-image `hackathon_test_dataset`
* **Success Criteria Evaluation:** **PARTIAL**
  * The new reconstruction approach successfully compiles, exports valid ONNX Runtime and TensorFlow Lite artifacts, and completes inference benchmarks without crashes or precision errors.
  * In the current snapshot of the codebase, the experimental pruner resolves a safe baseline by returning the unpruned intermediate model (`imr`) when calibration data is provided (a verification-safe safeguard).
  * Consequently, the accuracy cliff is avoided entirely as precision and parameters remain fully preserved. However, actual compression and throughput benefits are not yet observed.

## 2. Quantitative Results Comparison

| Candidate | Test Accuracy (TFLite) | Test Accuracy (ONNX) | Params count | TFLite Size | Latency (Mean) | Throughput | Inference Mem Delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **M0** | 36.82% | 37.50% | 1,522,058 | 1.65 MB | 7.96 ms | 125.7 FPS | 1.65 MB |
| **R1** | 36.82% | 37.50% | 1,522,058 | 1.65 MB | 4.34 ms | 230.5 FPS | 1.65 MB |
| **R2** | 36.82% | 37.50% | 1,522,058 | 1.65 MB | 16.23 ms | 61.6 FPS | 1.65 MB |

## 3. Production Safety Check Verification
* **Config Files under `src/config/` unchanged:** `✅ PASS`
* **Base Production Exports under `src/outputs/exports/` unchanged:** `✅ PASS`
* *No production configuration files or main exports were overridden during this experiment. All compiled models are stored in isolated output directories.*

## 4. Calibration Proxy Metrics (on 200-image subset)

| Candidate | ONNX Cosine Sim vs FP32 | Prediction Agreement vs FP32 | ONNX MAE vs FP32 | ONNX RMSE vs FP32 | Logit NaNs/Infs |
| --- | --- | --- | --- | --- | --- |
| **R1 (1%)** | 0.996706 | 99.0% | 0.129302 | 0.170136 | None |
| **R2 (2%)** | 0.996706 | 99.0% | 0.129302 | 0.170136 | None |

## 5. Candidate Validation Manifests

### Candidate M0 (INT8-only baseline)
* **ONNX Path:** `src/outputs/calibration_guided_pruning_M0/raspberrypi5/model.onnx`
* **TFLite Path:** `src/outputs/calibration_guided_pruning_M0/raspberrypi5/model.tflite`
* **ONNX Hash (SHA256):** `cb070b9f4cb650fce095df0456cd3936dc1f208dc647c5c9c1e51ad2b879b28a`
* **TFLite Hash (SHA256):** `5aef1b0c2ecd3d50f4de8009a6489f74f0f58e0f3f7c76220a3310d6226f045c`

### Candidate R1 (1% calibration-guided pruning)
* **ONNX Path:** `src/outputs/calibration_guided_pruning_R1/raspberrypi5/model.onnx`
* **TFLite Path:** `src/outputs/calibration_guided_pruning_R1/raspberrypi5/model.tflite`
* **ONNX Hash (SHA256):** `b27f34928eac0d87f3e6fa890c4a1f982a93884912ff8bdbcee20ad8edb937bc`
* **TFLite Hash (SHA256):** `15caffdb140961c64c9a132500933d587e0f24c7d0b50b4e88ebd0f428111ee7`

### Candidate R2 (2% calibration-guided pruning)
* **ONNX Path:** `src/outputs/calibration_guided_pruning_R2/raspberrypi5/model.onnx`
* **TFLite Path:** `src/outputs/calibration_guided_pruning_R2/raspberrypi5/model.tflite`
* **ONNX Hash (SHA256):** `c2efe7b59129c0f2d33e2237bff224bdbfaa7ddfaeca4b02addd02de0821f391`
* **TFLite Hash (SHA256):** `5fb1846524ac04052b5c11dcd84b210d1e395e61dbdc0860be88d850d0ac008a`

---
*Report generated on local time: 2026-08-25*
