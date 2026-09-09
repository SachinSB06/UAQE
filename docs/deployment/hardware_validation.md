# Hardware Validation Status

This document clarifies the exact hardware deployment state and measurement status for UAQE artifacts.

---

## 1. Capability-Driven "Universal" Orchestration

In UAQE, **"Universal"** denotes capability-driven orchestration: the engine dynamically inspects input models, operator sets, and target capabilities to orchestrate the most viable optimization paths (e.g., TFLite, ONNX, XNNPACK). It does **not** claim that every neural architecture in existence is supported out-of-the-box.

---

## 2. Hardware Validation State

| Target Platform | Runtime Engine | Execution Status | Validation Status |
| :--- | :--- | :--- | :--- |
| **Development Host CPU (AMD64 / x86_64)** | TensorFlow Lite / XNNPACK | Local Direct Execution | **VALIDATED & BENCHMARKED** |
| **Raspberry Pi 5 (Cortex-A76)** | TFLite ARM NEON / DotProd | Cross-compilation & profiling ready | **PENDING PHYSICAL HARDWARE RUN** |
| **Generic Embedded ARM Cortex-M** | TFLite for Microcontrollers (TFLM) | Supported operator subset | **THEORETICAL PROFILE READY** |

> [!NOTE]
> All latency, RAM, and throughput measurements presented in the current web dashboard reflect direct execution on the **Development Host CPU** (Windows 11 AMD64). Physical on-device execution on Raspberry Pi 5 hardware remains marked as **PENDING** until a live physical device test run is conducted.

---

## 3. Real-World Execution Considerations

- **INT8 vs FP32 Latency**: INT8 models reduce memory bandwidth and model storage by ~4×. However, whether INT8 runs faster than FP32 depends heavily on host CPU hardware instructions (such as AVX-512 VNNI, ARM NEON DotProd) and whether CPU cache effects dominate for small batch sizes. INT8 is **not** unconditionally faster than FP32 on every CPU architecture.
- **XNNPACK Micro-Kernels**: XNNPACK acceleration provides optimized GEMM and CONV micro-kernels, but relative speedups vary depending on matrix dimensions, cache alignment, and delegate invocation overhead.
