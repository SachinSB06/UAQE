# UAQE Repository Structure

This document outlines the organization and directory structure of the **Universal AI Quantization Engine (UAQE)** repository.

---

## Directory Hierarchy

```text
d:\Quantization embedded\
├── src/                        # Core Python engine containing models, optimizers, pipeline, runtime, telemetry, and API server
├── frontend/                   # React 19 + TypeScript + Vite web dashboard for interactive job monitoring, candidate visualization, and telemetry
├── tests/                      # Top-level regression suites (XNNPACK compatibility, candidate metric identity, latency provenance)
├── output/                     # Generated job executions, candidate models, and authoritative diagnostic reports
├── reports/                    # Historical phase benchmark logs, audit findings, and software validation records
├── datasets/                   # Reference datasets, validation manifests, and calibration image sets (e.g., SEM 9-class, CIFAR-10)
├── scratch/                    # Dedicated temporary test scratch workspace utilized during active unit and pipeline execution
├── archive/                    # Preserved historical temporary outputs, intermediate build records, and experiment scratch scripts
├── docs/                       # Architectural documentation, benchmarking protocols, optimization strategies, and operational guides
├── run_phase_*.py              # Authoritative phase execution and experiment reproducibility entrypoints
├── run_mobilenet_*.py          # Autonomous end-to-end multi-candidate optimization runner for MobileNetV3
├── uaqe.py                     # Command-line interface (CLI) entrypoint for headless engine execution
├── pytest.py / pytest.bat      # Environment-aware test runner wrappers ensuring clean Python 3.13 module resolution
├── start_uaqe.bat              # One-click startup script launching both FastAPI backend and Vite frontend
├── requirements-runtime.txt    # Runtime Python dependency specification
└── README.md                   # Primary project overview, operational instructions, and safety policy documentation
```

---

## Directory Descriptions

- **`src/`**: Contains the complete production Python engine, organized into sub-packages: `analyzer`, `dataset`, `exporter`, `model`, `optimizer`, `pipeline`, `runtime`, `server`, `telemetry`, `trainer`, and `utils`.
- **`frontend/`**: The modern web dashboard built with React 19 and Vite, providing real-time Server-Sent Events (SSE) streaming, per-candidate latency provenance cards, interactive Pareto frontier charts, and hardware telemetry displays.
- **`tests/`**: Authoritative root test suites verifying critical system guarantees, including candidate metric isolation (`test_candidate_metric_identity.py`), runtime delegate provenance (`test_candidate_latency_provenance.py`), and XNNPACK integration (`test_xnnpack_strategy_integration.py`).
- **`output/`**: The storage destination for live optimization jobs (`output/jobs/`), verified baseline models (`output/phase_c2/`), experimental delegates (`output/phase_xnnpack/`), and forensic diagnostic reports (`output/reports/`).
- **`reports/`**: Preserved historical audit logs, software validation results, phase progression reports (Phase B through Phase R1), and dependency mapping artifacts.
- **`datasets/`**: Reference image datasets and manifests used for calibration-guided pruning, post-training quantization calibration, and classification validation.
- **`scratch/`**: Standard test scratch directory used by test runners to store temporary model weights, logs, and evaluation dumps without polluting the root.
- **`archive/`**: Safely archived temporary runs, intermediate build notes (`file_order.txt`), and non-functional historical test scripts.
- **`docs/`**: Comprehensive project documentation covering system architecture, benchmarking methodology, candidate strategies, and hardware deployment states.
