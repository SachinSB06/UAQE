# Architecture Overview

The **Universal AI Quantization Engine (UAQE)** is an autonomous, capability-driven model optimization and quantization framework designed to transform deep learning models into ultra-efficient edge deployment artifacts while preserving accuracy and safety boundaries.

---

## High-Level Architecture

```text
+-------------------------------------------------------------------------+
|                              User Interface                             |
|       React 19 + TypeScript + TailwindCSS / Lucide-React Dashboard      |
|    - Real-time SSE Pipeline Tracker      - Pareto Frontier Visualizer   |
|    - Per-Candidate Provenance Display    - Telemetry & System Gauges    |
+-------------------------------------------------------------------------+
                                    ▲
                                    │ HTTP / Server-Sent Events (SSE)
                                    ▼
+-------------------------------------------------------------------------+
|                           FastAPI Backend                               |
|   /api/jobs          - Lifecycle management & candidate persistence     |
|   /api/jobs/stream   - Live progress & log event streaming              |
|   /api/telemetry     - Hardware sampling (CPU, RAM, RSS, Peak RAM)      |
+-------------------------------------------------------------------------+
                                    ▲
                                    │ Direct In-Process Engine Invocation
                                    ▼
+-------------------------------------------------------------------------+
|                        Autonomous Engine Pipeline                       |
|                                                                         |
|  [Stage 1] Capability Inspection & Layer Profiling                      |
|  [Stage 2] Baseline Evaluation (FP32 Accuracy & Latency)                |
|  [Stage 3] Candidate Search (Multi-Strategy Autonomous Generation)       |
|            - Candidate 1: Standard PTQ INT8 (Per-Tensor / Channel)      |
|            - Candidate 2: Reconstruction Pruning + INT8                 |
|            - Candidate 3: Mixed Precision (INT8 / FP16 / INT4)          |
|            - Candidate 4: XNNPACK-Optimized INT8 Strategy               |
|  [Stage 4] Canonical Benchmark & Accuracy Validation                     |
|  [Stage 5] Multi-Objective Scoring & Winner Selection                   |
|  [Stage 6] Artifact Packaging & Manifest Emission                       |
+-------------------------------------------------------------------------+
```

---

## Core Components

### 1. Ingestion & Architecture Inspection (`src/uaqe/analyzer/`, `src/uaqe/model/`)
- Analyzes model input tensors, operator types, weight tensor shapes, and layer topology.
- Detects model family compatibility (MobileNetV3, ResNet-50, Vision Transformers) and unsupported structures (e.g., dynamic ViT attention heads) before committing optimization compute.

### 2. Multi-Candidate Generation (`src/uaqe/optimizer/`)
- Generates diverse candidate architectures rather than a single brute-force output.
- Employs calibration-guided structured pruning, Fisher information approximation, and QAT refinement.

### 3. Isolated Benchmarking (`src/uaqe/telemetry/`, `src/uaqe/runtime/`)
- Measures actual runtime latency, memory consumption, and throughput.
- Enforces strict inter-run timing isolation, garbage collection, and thread affinity.

### 4. Per-Candidate Provenance Propagation
- Candidate metrics are strictly isolated by unique `candidate_id` across backend serialization, API transmission, and frontend rendering, preventing cross-candidate state leakage.
