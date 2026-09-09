# Universal AI Quantization Engine (UAQE)

[![Tests Passing](https://img.shields.io/badge/Tests-430%2B%20Passing-brightgreen)](#21-testing)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue)](#16-installation--setup)
[![Frontend React 19](https://img.shields.io/badge/Frontend-React%2019%20%7C%20Vite-61dafb)](frontend/)
[![Zero Mocks](https://img.shields.io/badge/Metrics-100%25%20Empirical%20%7C%20No%20Mocks-orange)](#22-benchmark-methodology--provenance)
[![License Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## 1. Overview & Purpose

The **Universal AI Quantization Engine (UAQE)** is an autonomous, capability-driven model optimization and quantization framework engineered to transform edge computer vision neural networks into verified, deployment-ready artifacts.

UAQE replaces brittle manual trial-and-error quantization pipelines with an autonomous multi-candidate search engine. It balances **accuracy retention**, **model size reduction**, **execution latency**, and **memory footprint** against target hardware profiles. Every metric reported by UAQE is measured empirically through isolated hardware execution—synthetic, simulated, or hardcoded benchmark metrics are strictly forbidden across production execution paths.

> [!NOTE]
> In UAQE, **"Universal"** refers to **capability-driven orchestration**—the engine dynamically inspects layer topologies, tensor shapes, and target capabilities to orchestrate the most viable optimization paths. It does **not** claim that every neural architecture in existence is supported out-of-the-box.

---

## 2. Problem Statement: The Edge Deployment Gap

Deploying modern deep neural networks to resource-constrained edge platforms (e.g., Raspberry Pi, embedded ARM Cortex, MCUs, and edge FPGAs) presents severe bottlenecks:
1. **Memory Bandwidth Bottleneck**: 32-bit floating-point (FP32) weights place prohibitive demands on external DRAM bandwidth, causing compute units to stall.
2. **Strict SRAM/DRAM Limits**: Embedded micro-controllers and edge systems possess severely limited local storage (often <16 MB flash / <2 MB SRAM).
3. **Thermal & Energy Constraints**: FP32 arithmetic units consume significantly higher energy per multiply-accumulate (MAC) operation compared to integer vector instructions.
4. **Degradation Risk**: Uncalibrated post-training quantization frequently causes severe accuracy collapse without automated guardrails.

---

## 3. Why Quantization is Needed

Quantization maps continuous 32-bit floating-point parameters to discrete 8-bit signed integer representations:

$$q = \text{round}\left(\frac{x}{S}\right) + Z$$

where $S$ is the scale factor and $Z$ is the zero-point offset.

- **4× Storage Reduction**: Eliminates 75% of weight footprint immediately.
- **SIMD/Vector Acceleration**: Enables high-throughput integer vector execution (e.g., ARM NEON Dot-Product, Intel/AMD AVX-512 VNNI).
- **Reduced Memory Bandwidth**: Drastically lowers cache-line misses and power consumption during batch inference.

---

## 4. UAQE Architecture

UAQE is built on a decoupled, modular architecture:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                      UAQE Web Cockpit (React 19 + Vite)                     │
│               Interactive Dashboard • Telemetry Replay • Pareto             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ REST / SSE Events
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                    Master API Server (FastAPI + Uvicorn)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Model & Dataset Ingestors        │  Capability & Compatibility Probers     │
│  Autonomous Optimization Planner  │  Optimization Strategy Resolver         │
├─────────────────────────────────────────────────────────────────────────────┤
│                         Optimization Engine Core                            │
│  ├── Standard PTQ INT8            ├── Sensitivity Structured Pruner         │
│  ├── Reconstruction Engine        ├── Quantization-Aware Fine-Tuning (QAT)  │
│  └── XNNPACK Delegate Optimizer   └── FlatBuffer Compression Packager       │
├─────────────────────────────────────────────────────────────────────────────┤
│                Isolated Benchmark & Timing Provenance Layer                 │
│         Interpreters: TensorFlow Lite • ONNX Runtime • Native Host          │
├─────────────────────────────────────────────────────────────────────────────┤
│                 Safety Policy & Multi-Objective Pareto Selector             │
├─────────────────────────────────────────────────────────────────────────────┤
│               Artifact Exporter (Self-Contained .uaqe Bundles)              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. End-to-End Workflow

```text
[Input Model + Calibration Dataset]
                │
                ▼
   [Capability & Topology Inspection]
                │
                ▼
   [FP32 Baseline Benchmark (Empirical)]
                │
                ▼
  [Autonomous Multi-Candidate Generation]
   ├── Candidate 1: Standard PTQ INT8
   ├── Candidate 2: Reconstruction Pruning + INT8
   ├── Candidate 3: Mixed Precision (INT8 / FP16)
   └── Candidate 4: Experimental XNNPACK INT8
                │
                ▼
  [Canonical Benchmark & Timing Isolation]
                │
                ▼
    [Pareto Frontier & Winner Selection]
                │
                ▼
 [Artifact Packaging + Manifest & Provenance]
```

---

## 6. Capability-Driven Model & Dataset Support

### Supported Models
- **MobileNetV3 (Small / Large)**: Fully supported for Post-Training Quantization (PTQ INT8), Quantization-Aware Training (QAT), Structured Sensitivity Pruning, and XNNPACK CPU acceleration.
- **ResNet-50**: Supported for PTQ INT8, layer sensitivity profiling, and CIFAR-10 transfer learning workflows.
- **Vision Transformers (ViT-Large / ViT-Base)**: Ingested and topology-inspected. Dynamic attention operators are flagged as unsupported by edge TFLite micro-kernels and gracefully rejected via explicit fail-safe guardrails with HTTP 400 diagnostics rather than failing unpredictably.

### Supported Datasets
- **ImageFolder Structure**: Standard multi-class directories (`class_name/img.png`) or `train/`/`val/` splits.
- **CIFAR-10**: Binary pickle batches for baseline verification and transfer learning.
- **Manifest-Driven Sets**: JSON manifests detailing image relative paths and labels for reproducible calibration.

---

## 7. Autonomous Optimization & Pareto-Optimal Search

UAQE executes a structured multi-candidate search driven by optimization profiles:
- `balanced`: Balanced weighting between accuracy retention (40%), size reduction (30%), and latency speedup (30%).
- `accuracy_first`: Prioritizes minimum degradation (max 1.0% accuracy drop).
- `size_first`: Prioritizes extreme compression for storage-constrained microcontrollers.
- `latency_first`: Maximizes frames-per-second throughput and micro-kernel delegation.

Candidates are plotted along a multi-dimensional Pareto frontier evaluating accuracy versus latency and storage footprint.

---

## 8. Implemented Optimization Paths

1. **Post-Training Quantization (PTQ INT8)**: Per-channel symmetric weight quantization with asymmetric activation calibration using representative datasets.
2. **Sensitivity-Guided Structured Pruning**: Channel pruning guided by Fisher information approximations, followed by local least-squares weight reconstruction.
3. **Mixed Precision (INT8 / FP16)**: Preserves noise-vulnerable initial and terminal layers in 16-bit float while quantizing compute-heavy convolutional backbones to 8-bit.
4. **Quantization-Aware Training (QAT)**: Fine-tunes model weights with simulated rounding noise injected directly into the computational graph.
5. **Adaptive FlatBuffer Packaging**: Zero-copy FlatBuffer rearrangement with payload compression to produce portable `.uaqe` execution bundles.

---

## 9. Experimental XNNPACK Strategy

Standard TFLite INT8 models frequently encounter operator delegation fallbacks when targeting CPU micro-kernel engines like XNNPACK. UAQE includes an **experimental XNNPACK optimization strategy** (Candidate 4) that aligns tensor quantization parameters and bias offsets.

- **Verified Full Delegation**: Successfully delegates **204 out of 204 operators** on MobileNetV3 to the XNNPACK CPU delegate with **0 fallback operators**.
- **Execution**: Evaluated alongside reference candidates under dedicated hardware isolation.

---

## 10. Hardware Awareness & Target Registry

UAQE maintains an extensible hardware registry (`HardwareTargetRegistry`) modeling physical compute boundaries:

| Target ID | Target Device | Hardware Class | SRAM | Flash / DRAM | Default Format |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `raspberrypi5` | Raspberry Pi 5 (Cortex-A76) | SBC / Edge CPU | N/A | 4 GB / 8 GB | TFLite INT8 |
| `esp32` | Espressif ESP32-S3 | Embedded MCU | 512 KB | 8 MB | TFLite Micro |
| `stm32` | STMicroelectronics STM32H7 | Embedded MCU | 1 MB | 2 MB | TFLite Micro |
| `artix7` | Xilinx Artix-7 XC7A100T | Edge FPGA | 135 BRAM | 8 MB Flash | Custom UAQE |
| `zynq7000` | Xilinx Zynq-7020 SoC | FPGA SoC | 140 BRAM | 1 GB DRAM | Custom UAQE |

---

## 11. Telemetry, Observability & SSE Streaming

- **Process-Level Telemetry**: Continuous sampling of host CPU utilization (`psutil.cpu_percent()`) and process resident set memory (`psutil.Process().memory_info().rss`).
- **Real-Time Streaming**: Server-Sent Events (SSE) at `/api/jobs/{job_id}/events` stream live phase transitions, candidate progress, and benchmark timings directly to the React cockpit.
- **Timing Isolation**: Warmup passes (5–10 iterations) and explicit garbage collection barriers ensure latency measurements are untainted by cold-start compilation overhead.

---

## 12. Validation & Safety Guardrails

- **Degradation Threshold**: Candidates exceeding configurable accuracy degradation limits (default 3.0 percentage points) are flagged as critical safety violations and rejected from winning selection.
- **Fallback Protection**: If an advanced optimization candidate fails or degrades, the system automatically falls back to verified reference baselines.
- **Data-Flow Isolation**: Strict candidate identity binding prevents metric leakage across candidates.

---

## 13. Project Structure

```text
UAQE/
├── README.md                      # Comprehensive project documentation
├── LICENSE                        # Apache License 2.0
├── .gitignore                     # Git exclusion rules for clean version control
├── requirements.txt               # Full development, runtime, and testing dependencies
├── requirements-runtime.txt       # Minimal deployment runtime dependencies
├── start_uaqe.bat                 # One-click Windows CMD startup script
├── start_uaqe.ps1                 # Native Windows PowerShell startup script
├── pytest.py / pytest.bat / ps1   # Environment-aware test runners
├── uaqe.py                        # Master CLI entrypoint
├── src/
│   ├── uaqe/                      # Core Python engine package
│   │   ├── analyzer/              # Sensitivity and capability inspection
│   │   ├── api/                   # FastAPI routes (jobs, uploads, telemetry)
│   │   ├── compression/           # Structured pruners and packagers
│   │   ├── dataset/               # Dataset loaders and adapters
│   │   ├── exporter/              # Manifests and binary exporters
│   │   ├── orchestration/         # Autonomous planners and hardware targets
│   │   ├── runtime/               # Isolated benchmark execution engines
│   │   ├── telemetry/             # Hardware monitors and process metrics
│   │   └── server.py              # Master FastAPI backend server
│   └── models/                    # Reference model topologies
├── frontend/                      # React 19 + TypeScript + Vite web dashboard
│   ├── src/
│   │   ├── pages/                 # Cockpit, Jobs, Results, Inspection views
│   │   ├── charts/                # Pareto, Scatter, Telemetry, and Resource cards
│   │   ├── services/              # API client and SSE subscribers
│   │   └── utils/                 # Candidate metrics normalization and selectors
│   ├── package.json
│   └── vite.config.ts
├── tests/                         # Top-level regression suites
├── models/                        # Small verified demonstration artifacts & guide
│   ├── README.md
│   ├── mobilenetv3_sem_9class_qat_int8.tflite
│   └── mobilenetv3_sem_9class_xnnpack_int8.tflite
├── datasets/                      # Reference calibration sets & guidelines
│   ├── README.md
│   └── calibration/
├── docs/                          # Technical reports, architecture specifications
└── .github/
    └── workflows/
        └── tests.yml              # Automated GitHub Actions CI workflow
```

---

## 14. Installation & Setup

### Prerequisites
- **Python**: Version 3.11, 3.12, or 3.13 (Tested on Python 3.13.7 AMD64)
- **Node.js**: Version 18+ and npm (Tested on Node v20 & v24)
- **Git**: For repository cloning

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/your-username/UAQE.git
cd UAQE

# 2. Set up Python virtual environment (recommended)
python -m venv .venv
# Activate on Windows CMD:
.venv\Scripts\activate
# Activate on PowerShell:
.venv\Scripts\Activate.ps1

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install frontend dependencies
cd frontend
npm install
cd ..
```

---

## 15. Launching UAQE

UAQE provides native, repository-relative startup launchers that automatically discover their location, verify Python and Node prerequisites, recreate runtime directories, and launch both backend and frontend servers.

### Windows Command Prompt (CMD)
```cmd
start_uaqe.bat
```

### Windows PowerShell
```powershell
.\start_uaqe.ps1
```

Once launched:
- **Web Dashboard**: Open [http://localhost:3000](http://localhost:3000)
- **Backend API**: Accessible at [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive OpenAPI Documentation**: Available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Shutdown
To stop the application cleanly, close the spawned `UAQE Backend` and `UAQE Frontend` console windows, or press `Ctrl+C` inside those windows.

---

## 16. Command-Line Interface (CLI) Usage

UAQE can be executed entirely headless via `uaqe.py`:

```bash
# 1. Inspect model topology and parameter counts
python uaqe.py inspect-model --model "models/mobilenetv3_sem_9class_qat_int8.tflite"

# 2. Inspect dataset format, class distribution, and splits
python uaqe.py inspect-dataset --dataset "datasets/calibration/dataset"

# 3. Validate compatibility between model and dataset
python uaqe.py validate \
  --model "models/mobilenetv3_sem_9class_qat_int8.tflite" \
  --dataset "datasets/calibration/dataset"

# 4. Dry-run optimization plan generation without executing
python uaqe.py plan \
  --model "src/models/mobilenetv3_sem.onnx" \
  --dataset "datasets/calibration/dataset" \
  --target raspberrypi5 \
  --profile balanced

# 5. Execute full autonomous multi-candidate optimization
python uaqe.py optimize \
  --model "src/models/mobilenetv3_sem.onnx" \
  --dataset "datasets/calibration/dataset" \
  --target raspberrypi5 \
  --profile balanced \
  --auto-approve
```

---

## 17. Example End-to-End Workflow

1. **Ingestion**: Upload an ONNX or PyTorch vision model via the UI or CLI.
2. **Capability Check**: The engine parses tensor dimensions, layer operations, and input normalization requirements.
3. **Baseline Run**: Measures empirical FP32 latency, throughput, and accuracy on the host CPU.
4. **Candidate Generation**: The autonomous optimizer generates candidates:
   - Candidate 1: Standard Post-Training INT8 Quantization.
   - Candidate 2: Channel Pruning + Least-Squares Reconstruction + INT8.
   - Candidate 3: Mixed Precision INT8 backbone with FP16 sensitive layers.
   - Candidate 4: XNNPACK-aligned INT8 optimization.
5. **Evaluation**: Warmup passes and garbage collection barriers execute for each candidate.
6. **Pareto Winner**: The composite scoring function selects the optimal candidate adhering to safety constraints.
7. **Packaging**: The user downloads the candidate-specific `.tflite` model, diagnostic reports, and self-contained `.uaqe` archive.

---

## 18. Testing

Run tests using the project's environment-aware wrapper:

```bash
# Run candidate metric identity & data isolation suite
python pytest.py -q tests/test_candidate_metric_identity.py

# Run candidate latency provenance & timing verification suite
python pytest.py -q tests/test_candidate_latency_provenance.py

# Run candidate selection switch & artifact download suite
python pytest.py -q tests/test_candidate_selection_and_download.py

# Run XNNPACK strategy integration suite
python pytest.py -q tests/test_xnnpack_strategy_integration.py

# Build frontend production bundle
cd frontend && npm run build
```

---

## 19. Benchmark Methodology & Latency Provenance

- **Timing Isolation**: Every candidate is evaluated in a dedicated interpreter instance with multi-iteration warmup (5–10 runs), garbage collection barriers, and microsecond-resolution system timers (`time.perf_counter_ns()`).
- **Canonical Metrics**: Latencies and throughput values are normalized using canonical formulas (`speedup = (baseline_latency - candidate_latency) / baseline_latency * 100`, `throughput = 1000 / latency_ms`).
- **Provenance Integrity**: Each candidate's runtime backend (`Standard CPU` vs `XNNPACK Delegate`) and performance metrics are keyed strictly by unique `candidate_id`, preventing cross-candidate state contamination.

---

## 20. Known Limitations

- **INT8 Hardware Acceleration**: INT8 models reduce model size by ~4×. However, whether INT8 runs faster than FP32 depends on host CPU hardware instructions (e.g., AVX-512 VNNI or ARM NEON DotProd); INT8 is **not** unconditionally faster than FP32 across all CPUs.
- **Vision Transformers**: ViT models with dynamic attention shapes are recognized by capability probers but currently guarded as unsupported by edge TFLite micro-kernels.
- **Physical Edge Benchmarking**: Direct hardware measurements on physical Raspberry Pi 5 boards are pending scheduled physical lab execution.

---

## 21. Hardware Measurement Disclaimer

> [!IMPORTANT]
> All telemetry, latency figures, and resource utilization metrics reported in this repository reflect measurements performed on the **Development Host CPU (AMD64)**. On-device physical validation on Raspberry Pi 5 hardware remains marked as **PENDING Physical Run** in telemetry manifests until physical bench runs are executed.

---

## 22. Future Work

- [ ] Physical Raspberry Pi 5 automated benchmarking harness via serial/SSH telemetry.
- [ ] Support for 4-bit (INT4) weight-only quantization targeting edge LLMs.
- [ ] TensorRT and OpenVINO export delegates for industrial edge PCs.
- [ ] Custom FPGA bitstream synthesizers for AMD-Xilinx Kria and Zynq platforms.

---

## 23. License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for complete details.
