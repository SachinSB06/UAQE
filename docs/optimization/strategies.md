# Optimization Strategies

UAQE generates multiple candidate models during autonomous optimization, evaluating each candidate along a multi-objective Pareto frontier balancing accuracy retention, model footprint, latency, and hardware acceleration suitability.

---

## Candidate Strategies Overview

| Candidate ID | Strategy Name | Description | Key Operators / Delegate |
| :--- | :--- | :--- | :--- |
| **Candidate 1** | Post-Training Quantization (PTQ INT8) | Full integer 8-bit quantization using representative calibration data. | Standard TFLite CPU Runtime |
| **Candidate 2** | Sensitivity-Guided Structured Pruning + INT8 | Identifies and prunes redundant channels via Fisher information approximation, followed by PTQ INT8. | Standard TFLite CPU Runtime |
| **Candidate 3** | Mixed Precision (INT8 / FP16) | Quantizes noise-sensitive layers to FP16 while maintaining INT8 for heavy compute layers. | TFLite Hybrid Runtime |
| **Candidate 4** | Experimental XNNPACK Strategy | Restructures quantization ranges and operator bindings for direct XNNPACK acceleration. | TFLite XNNPACK Delegate (204 delegated ops) |

---

## 1. Post-Training Quantization (PTQ INT8)
- Quantizes weights to signed `int8` and activations to asymmetric `uint8`/`int8`.
- Uses representative calibration datasets to calculate dynamic scale and zero-point parameters.
- Achieves approximately **4× compression** over FP32 baselines.

## 2. Sensitivity-Guided Pruning + Reconstruction
- Evaluates per-channel Hessian diagonal or gradient energy to identify prunable channels.
- Solves local least-squares reconstruction to minimize layer-wise output drift before global finetuning.
- Significantly reduces theoretical FLOPs.

## 3. Mixed Precision Allocation
- Analyzes per-layer signal-to-quantization-noise ratio (SQNR).
- Isolates high-vulnerability layers (e.g., initial stem convolutions or classification heads) in 16-bit float, preventing severe accuracy collapse.

## 4. Experimental XNNPACK Optimization
- Specifically addresses operator delegation bottlenecks in standard TFLite INT8 models.
- Enforces strict operator attribute alignment and bias vector scale constraints, enabling 100% of supported convolution and depthwise convolution operators (e.g., 204 operators on MobileNetV3) to delegate to the XNNPACK micro-kernel execution engine.
