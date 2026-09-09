# 02_Folder_Structure.md
## Universal AI Quantization Engine — Folder Structure

**Depends on:** `01_Project_Architecture.md`
**Status:** DRAFT — pending Architecture Lock

---

## 1. Top-Level Layout

```
uaqe/
├── src/
│   └── uaqe/
│       ├── common/
│       ├── domain/
│       ├── application/
│       ├── infrastructure/
│       └── interface/
├── config/
├── hardware_profiles/
├── datasets/
├── workdir/
├── outputs/
├── reports/
├── logs/
├── tests/
├── docs/
├── scripts/
├── plugins/
├── pyproject.toml
├── README.md
└── LICENSE
```

Every top-level folder below is owned by exactly one module or is explicitly shared read-only infrastructure. No folder may be written to by more than one module at runtime (enforced by `07_Coding_Standards.md`).

---

## 2. `src/uaqe/common/`

**Purpose:** Framework-agnostic primitives shared by every layer. Contains no business logic, no I/O.
**Owner:** Architecture team (shared)
**Depends on:** Nothing inside `uaqe`

```
common/
├── __init__.py
├── imr.py                  # Internal Model Representation: IMR, IMRLayer, IMRTensor classes
├── types.py                 # Shared enums: Precision, HardwareClass, CompressionType, ExportFormat
├── value_objects.py         # Immutable config value objects (QuantizationConfig, HardwareConfig, ...)
├── exceptions.py             # UAQEError hierarchy (see 01_Project_Architecture.md §12)
├── result_types.py           # StageResult, RunResult, CompatibilityReport, base dataclasses
├── constants.py               # Non-configurable literal constants (e.g., schema version strings)
└── interfaces/
    ├── __init__.py
    ├── i_logger.py            # ILogger
    ├── i_framework_adapter.py # IFrameworkAdapter
    ├── i_quantization_strategy.py
    ├── i_compression_strategy.py
    ├── i_exporter_backend.py
    ├── i_report_renderer.py
    ├── i_hardware_profile_repository.py
    └── i_config_repository.py
```

---

## 3. `src/uaqe/domain/`

**Purpose:** All ML/hardware decision logic. Zero third-party ML framework imports (no `torch`, `tensorflow`, `onnx` imports anywhere under this tree).
**Owner:** Domain module leads
**Depends on:** `uaqe.common` only

```
domain/
├── __init__.py
├── pipeline/
│   ├── __init__.py
│   ├── pipeline_orchestrator.py
│   ├── pipeline_context.py
│   ├── pipeline_stage.py          # abstract PipelineStage
│   └── pipeline_builder.py
├── model/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── model_validator.py
│   └── model_analyzer.py
├── compatibility/
│   ├── __init__.py
│   └── layer_compatibility_checker.py
├── hardware/
│   ├── __init__.py
│   └── hardware_manager.py
├── quantization/
│   ├── __init__.py
│   ├── quantization_advisor.py
│   ├── calibration_engine.py
│   ├── sensitivity_analyzer.py
│   └── quantization_engine.py
├── compression/
│   ├── __init__.py
│   ├── compression_advisor.py
│   └── compression_engine.py
├── optimization/
│   ├── __init__.py
│   ├── optimization_engine.py
│   └── memory_optimizer.py
├── export/
│   ├── __init__.py
│   └── exporter.py
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py
├── benchmark/
│   ├── __init__.py
│   └── benchmarker.py
├── advisory/
│   ├── __init__.py
│   ├── deployment_readiness_scorer.py
│   └── optimization_advisor.py
└── reporting/
    ├── __init__.py
    └── report_generator.py
```

---

## 4. `src/uaqe/application/`

**Purpose:** Orchestration and session/run lifecycle. No ML logic, no hardware logic.
**Owner:** Application layer lead
**Depends on:** `uaqe.domain`, `uaqe.common`

```
application/
├── __init__.py
├── model_ingestion_service.py
├── hardware_selection_service.py
├── session_manager.py
├── workflow_controller.py
└── run_request.py            # RunRequest DTO consumed from Interface Layer
```

---

## 5. `src/uaqe/infrastructure/`

**Purpose:** All concrete integrations: file I/O, ML framework parsing, hardware-native export, config loading, logging.
**Owner:** Infrastructure module leads (one per adapter family)
**Depends on:** `uaqe.common`; third-party libraries as needed per subfolder

```
infrastructure/
├── __init__.py
├── framework_adapters/
│   ├── __init__.py
│   ├── torch_adapter.py        # .pth, .pt
│   ├── onnx_adapter.py          # .onnx
│   ├── tensorflow_adapter.py    # .pb
│   ├── keras_adapter.py         # .h5, .keras
│   └── tflite_adapter.py        # .tflite
├── exporter_backends/
│   ├── __init__.py
│   ├── fpga/
│   │   ├── __init__.py
│   │   ├── artix7_backend.py
│   │   ├── zynq7000_backend.py
│   │   ├── kintex_backend.py
│   │   └── cyclonev_backend.py
│   ├── embedded/
│   │   ├── __init__.py
│   │   ├── esp32_backend.py
│   │   ├── esp32s3_backend.py
│   │   ├── stm32f4_backend.py
│   │   ├── stm32h7_backend.py
│   │   ├── rp2040_backend.py
│   │   └── portenta_h7_backend.py
│   └── raspberrypi/
│       ├── __init__.py
│       ├── raspberrypi4_backend.py
│       └── raspberrypi5_backend.py
├── repositories/
│   ├── __init__.py
│   ├── config_repository.py
│   ├── hardware_profile_repository.py
│   └── filesystem_repository.py
├── logging/
│   ├── __init__.py
│   └── structured_logger.py
├── metrics/
│   ├── __init__.py
│   └── metrics_collector.py
└── plugins/
    ├── __init__.py
    ├── plugin_registry.py
    └── builtin/
        ├── __init__.py
        ├── quantization/
        └── compression/
```

---

## 6. `src/uaqe/interface/`

**Purpose:** External entry points and the single Composition Root.
**Owner:** Interface layer lead
**Depends on:** All inner layers (only layer permitted to import concrete Infrastructure classes)

```
interface/
├── __init__.py
├── composition_root.py     # sole wiring point (see 01_Project_Architecture.md §15)
├── cli/
│   ├── __init__.py
│   └── cli_entry_point.py
├── api/
│   ├── __init__.py
│   ├── rest_entry_point.py
│   └── schemas/             # request/response Pydantic models
└── batch/
    ├── __init__.py
    └── batch_runner.py
```

---

## 7. `config/`

**Purpose:** All configuration files described in `06_Config_Spec.md`. Read-only at runtime by `ConfigRepository`.
**Owner:** `ConfigRepository`

```
config/
├── config.json
├── settings.yaml
├── hardware.json
├── compression.json
├── quantization.json
├── benchmark.json
└── reports.json
```

---

## 8. `hardware_profiles/`

**Purpose:** Hardware database JSON files consumed by `HardwareProfileRepository`. See `05_Hardware_Profile_Spec.md`.
**Owner:** `HardwareManager` (read-only)

```
hardware_profiles/
├── fpga/
│   ├── artix7.json
│   ├── zynq7000.json
│   ├── kintex.json
│   └── cyclonev.json
├── embedded/
│   ├── esp32.json
│   ├── esp32s3.json
│   ├── stm32f4.json
│   ├── stm32h7.json
│   ├── rp2040.json
│   └── portenta_h7.json
└── raspberrypi/
    ├── raspberrypi4.json
    └── raspberrypi5.json
```

---

## 9. `datasets/`

**Purpose:** Calibration and evaluation datasets, user-supplied or sample.
**Owner:** `CalibrationEngine` / `Evaluator` (read-only at runtime)

```
datasets/
├── calibration/
│   └── <dataset_name>/       # user-provided calibration samples
└── evaluation/
    └── <dataset_name>/       # user-provided evaluation/validation samples
```

---

## 10. `workdir/`

**Purpose:** Ephemeral per-run scratch space; safe to delete between runs.
**Owner:** `PipelineOrchestrator`

```
workdir/
└── <run_id>/
    ├── artifacts/            # streamed IMR snapshots (see 01_Project_Architecture.md §17)
    ├── calibration_cache/
    └── checkpoint.json       # resume state for PipelineContext
```

---

## 11. `outputs/`

**Purpose:** Final deployment-ready artifacts, one subfolder per run.
**Owner:** `Exporter`

```
outputs/
└── <run_id>/
    ├── fpga/          # .mem, .hex, .bin
    ├── embedded/       # .tflite, .bin, .h
    └── raspberrypi/    # .tflite, .onnx
```

---

## 12. `reports/`

**Purpose:** Generated report documents, one subfolder per run.
**Owner:** `ReportGenerator`

```
reports/
└── <run_id>/
    ├── accuracy_report.md
    ├── benchmark_report.md
    ├── compression_report.md
    ├── deployment_report.md
    └── summary_report.md
```

(Exact per-report schema and rendering format defined in `06_Config_Spec.md` and `10_Module_Development_Guide.md`.)

---

## 13. `logs/`

**Purpose:** Structured log output, one subfolder per run.
**Owner:** `StructuredLogger`

```
logs/
└── <run_id>/
    └── pipeline.log        # JSON-lines structured log (see 01_Project_Architecture.md §13)
```

---

## 14. `tests/`

**Purpose:** Mirrors `src/uaqe/` exactly, one test module per source module.
**Owner:** Each module's owning developer, reviewed per `07_Coding_Standards.md`

```
tests/
├── unit/
│   ├── common/
│   ├── domain/
│   ├── application/
│   └── infrastructure/
├── integration/
│   ├── pipeline_end_to_end/
│   └── exporter_backends/
└── fixtures/
    ├── sample_models/       # tiny .pth/.onnx/.h5 fixtures for CI
    └── sample_hardware_profiles/
```

---

## 15. `docs/`

**Purpose:** This document set plus any auto-generated API docs.

```
docs/
├── architecture/
│   ├── 01_Project_Architecture.md
│   ├── 02_Folder_Structure.md
│   ├── 03_API_Specification.md
│   ├── 04_Data_Flow.md
│   ├── 05_Hardware_Profile_Spec.md
│   ├── 06_Config_Spec.md
│   ├── 07_Coding_Standards.md
│   ├── 08_Project_Roadmap.md
│   ├── 09_Architecture_Lock.md
│   └── 10_Module_Development_Guide.md
└── generated/               # auto-generated API reference (out of scope for v1)
```

---

## 16. `scripts/`

**Purpose:** Developer utility scripts (not part of the shipped package): environment setup, dataset download helpers, lint/format wrappers.

```
scripts/
├── setup_dev_env.sh
├── run_lint.sh
├── run_tests.sh
└── generate_hardware_profile_template.py
```

---

## 17. `plugins/`

**Purpose:** External/custom plugin drop-in directory, scanned by `PluginRegistry` at startup (see `01_Project_Architecture.md` §11).

```
plugins/
├── quantization_strategies/
├── compression_strategies/
└── exporter_backends/
```

---

## 18. Root Files

| File | Purpose |
|---|---|
| `pyproject.toml` | Package metadata, dependency pins, tool config (black/isort/mypy/pytest) |
| `README.md` | Project overview and quickstart |
| `LICENSE` | Open-source license text |

---

## 19. Ownership Summary Table

| Folder | Owning Module | Write Access At Runtime |
|---|---|---|
| `config/` | `ConfigRepository` | Read-only |
| `hardware_profiles/` | `HardwareProfileRepository` | Read-only |
| `datasets/` | User-supplied | Read-only |
| `workdir/` | `PipelineOrchestrator` | Read/Write |
| `outputs/` | `Exporter` | Write |
| `reports/` | `ReportGenerator` | Write |
| `logs/` | `StructuredLogger` | Write |
| `plugins/` | `PluginRegistry` | Read-only |

---

**End of `02_Folder_Structure.md`.**
