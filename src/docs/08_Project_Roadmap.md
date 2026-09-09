# 08_Project_Roadmap.md
## Universal AI Quantization Engine — Project Roadmap

**Depends on:** `01_Project_Architecture.md` through `07_Coding_Standards.md`
**Status:** DRAFT — pending Architecture Lock
**Scope:** Breaks implementation into ordered, dependency-respecting phases. No implementation code. Every deliverable named here maps to a file in `02_Folder_Structure.md` and a contract in `03_API_Specification.md`. Phases are sequential; sub-items within a phase may proceed in parallel only where explicitly marked **[parallelizable]**.

---

## 0. Roadmap Principles

1. **Bottom-up dependency order.** A phase may only begin once every phase it depends on is complete and passing CI (`07_Coding_Standards.md` §11). This mirrors the layering in `01_Project_Architecture.md` §2: `common` → `domain` (leaf modules first) → `application` → `infrastructure` → `interface` → integration.
2. **Contracts before logic.** Every phase's first deliverable is the interface/abstract-class surface from `03_API_Specification.md` for the files in scope; concrete logic is filled in only after the contract is reviewed against `03_API_Specification.md` and `04_Data_Flow.md`.
3. **Testable increments.** Every phase ends in a state where `pytest` passes for everything built so far, per the 1:1 test-mirroring rule (`07_Coding_Standards.md` §10). No phase is "done" without its tests.
4. **No forward references.** A phase may not depend on a class, config file, or hardware profile that a later phase introduces. Where an early phase needs a stub (e.g. `ModelAnalyzer` needing a `HardwareProfile` before `Phase 6` exists), it depends on the **interface** only (`IHardwareProfileRepository`), satisfied by a test double, never a forward concrete dependency.
5. **Each phase names:** Objectives, Deliverables, Dependencies, Expected Outputs. "Expected Outputs" means the observable artifact (files that exist, tests that pass, CLI that runs) — not narrative description.

---

## Phase 1 — Architecture & Documentation Lock

**Objectives:** Freeze the design surface so every later phase has a single source of truth.

**Deliverables:**
- `01_Project_Architecture.md` through `10_Module_Development_Guide.md` (this document set), reviewed and approved.
- `09_Architecture_Lock.md` signed off — no further renames of packages, classes, methods, config keys, or JSON schemas without an RFC.
- Empty repository skeleton matching `02_Folder_Structure.md` §1–§19 exactly: every folder created, every package folder has an (empty) `__init__.py`, `pyproject.toml` scaffolded per `07_Coding_Standards.md` §1.

**Dependencies:** None.

**Expected Outputs:**
- A cloned repository whose `tree` output matches `02_Folder_Structure.md` byte-for-byte in structure (files may be empty).
- `pyproject.toml` with `black`, `isort`, `flake8`/`ruff`, `mypy --strict`, `pytest`, `pytest-cov`, `import-linter` configured but not yet enforced (no source to lint yet).

---

## Phase 2 — Core Framework: `uaqe.common`

**Objectives:** Build the framework-agnostic primitives every other layer depends on. Nothing in this phase imports anything else in `uaqe`.

**Deliverables** (all under `src/uaqe/common/`, all `[parallelizable]` once `types.py` exists since the others depend on it):
- `types.py` — `Precision`, `HardwareClass`, `CompressionType`, `ExportFormat`, `PipelineErrorPolicy` enums (`03_API_Specification.md` §1.2).
- `imr.py` — `IMR`, `IMRLayer`, `IMRTensor` classes (§1.1).
- `value_objects.py` — `QuantizationConfig`, `CompressionConfig`, `ExecutionConfig`, `HardwareConfig`, and other `@immutable` value objects (§1.3).
- `exceptions.py` — full `UAQEError` hierarchy (§1.4): `ModelLoadError`, `ModelValidationError`, `HardwareIncompatibilityError`, `QuantizationError`, `CompressionError`, `ExportError`, `BenchmarkError`, `ConfigurationError`, etc.
- `result_types.py` — `StageResult`, `RunResult`, `CompatibilityReport`, `AnalysisResult`, `SensitivityReport`, `EvaluationResult`, `BenchmarkResult`, `ReadinessScore`, `Recommendation`, `ReportDocument` (§1.5).
- `constants.py` — `HARDWARE_SCHEMA_VERSION` and other literal constants.
- `interfaces/` — all `I*` abstract classes: `ILogger`, `IFrameworkAdapter`, `IQuantizationStrategy`, `ICompressionStrategy`, `IExporterBackend`, `IReportRenderer`, `IHardwareProfileRepository`, `IConfigRepository` (§1.6–§1.13).

**Dependencies:** Phase 1.

**Expected Outputs:**
- `tests/unit/common/` fully populated, 1:1 with source files, ≥85% coverage (`07_Coding_Standards.md` §10).
- `mypy --strict` passes on `uaqe.common` with zero errors.
- `import-linter` contract confirms `uaqe.common` imports nothing else under `uaqe`.

---

## Phase 3 — Logging & Configuration Infrastructure

**Objectives:** Stand up the two cross-cutting services every later domain/application class is constructor-injected with: logging and configuration loading.

**Deliverables:**
- `infrastructure/logging/structured_logger.py` — `StructuredLogger implements ILogger` (JSON-lines, `run_id`/`stage_name`/`**fields`, per `07_Coding_Standards.md` §6).
- `infrastructure/repositories/config_repository.py` — `ConfigRepository implements IConfigRepository`, loading `config/*.json` and `settings.yaml` per `06_Config_Spec.md` §1–§9 (override precedence, schema validation).
- `infrastructure/repositories/filesystem_repository.py` — `FilesystemRepository`, the sole gateway to `workdir/`, `outputs/`, `reports/`, `logs/` (per `02_Folder_Structure.md` §19 ownership table).
- Populate `config/config.json`, `config/settings.yaml` with the default values shown in `06_Config_Spec.md` §1–§2 (other config files deferred to the phase that consumes them, to avoid speculative defaults for subsystems not yet built).

**Dependencies:** Phase 2.

**Expected Outputs:**
- `tests/unit/infrastructure/logging/`, `tests/unit/infrastructure/repositories/` passing.
- A throwaway script (not shipped — validated in CI only) that loads `config.json` via `ConfigRepository` and logs one structured line via `StructuredLogger`, proving the two integrate.
- `ConfigurationError` correctly raised for a missing/malformed `schema_version` (contract test per `06_Config_Spec.md` §9).

---

## Phase 4 — Hardware Profile Subsystem

**Objectives:** Build the hardware knowledge base and its repository, independent of any ML logic.

**Deliverables:**
- `infrastructure/repositories/hardware_profile_repository.py` — `HardwareProfileRepository implements IHardwareProfileRepository`.
- All 12 hardware profile JSON files under `hardware_profiles/fpga/`, `hardware_profiles/embedded/`, `hardware_profiles/raspberrypi/`, exactly as tabulated in `05_Hardware_Profile_Spec.md` §3–§5.
- `config/hardware.json` populated per `06_Config_Spec.md` §3.
- `domain/hardware/hardware_manager.py` — `HardwareManager implements PipelineStage`, `checkCompatibility()` stub sufficient for unit testing profile resolution (full compatibility logic depends on `LayerCompatibilityChecker` in Phase 6, so this phase covers profile *resolution* only, not layer-level compatibility).

**Dependencies:** Phase 2, Phase 3 (`ConfigRepository` for `hardware.json`; `FilesystemRepository`/`ILogger` injected into `HardwareProfileRepository`).

**Expected Outputs:**
- `HardwareProfileRepository.get("artix7")` (and all 11 other `profile_id`s) returns a correctly deserialized `HardwareProfile`, contract-tested against every JSON file per `07_Coding_Standards.md` §10 (interface contract test suite).
- `profile_cache_enabled` behavior verified (`06_Config_Spec.md` §3).
- Schema-version rejection test (`ConfigurationError` on unrecognized major version, `05_Hardware_Profile_Spec.md` §7.3).

---

## Phase 5 — Model Ingestion & Framework Adapters

**Objectives:** Get a model from disk into IMR, for every supported input format.

**Deliverables:**
- `infrastructure/framework_adapters/torch_adapter.py`, `onnx_adapter.py`, `tensorflow_adapter.py`, `keras_adapter.py`, `tflite_adapter.py` — each `implements IFrameworkAdapter`.
- `domain/model/model_loader.py` — `ModelLoader implements PipelineStage` (adapter selection logic per `04_Data_Flow.md` §2).
- `application/model_ingestion_service.py` — `ModelIngestionService`.
- `application/run_request.py` — `RunRequest` DTO.

**Dependencies:** Phase 2 (IMR, exceptions), Phase 3 (logging/config).

**Expected Outputs:**
- One passing fixture load per format in `tests/fixtures/sample_models/` (`.pth`, `.pt`, `.onnx`, `.pb`, `.h5`, `.keras`, `.tflite`) → valid `IMR`.
- `ModelLoadError` correctly raised and wraps the underlying framework exception (never leaks raw `torch`/`tf`/`onnx` exceptions past the adapter, per `07_Coding_Standards.md` §7).
- Contract test suite run against all five `IFrameworkAdapter` implementations (interchangeability guarantee, `07_Coding_Standards.md` §10).

---

## Phase 6 — Model Validation, Analysis & Compatibility

**Objectives:** Understand the model's structure and check it against a selected hardware target.

**Deliverables:**
- `domain/model/model_validator.py` — `ModelValidator implements PipelineStage`.
- `domain/model/model_analyzer.py` — `ModelAnalyzer implements PipelineStage`.
- `domain/compatibility/layer_compatibility_checker.py` — `LayerCompatibilityChecker implements PipelineStage`.
- `application/hardware_selection_service.py` — `HardwareSelectionService`.
- Completion of `HardwareManager.checkCompatibility()` deferred logic from Phase 4, now wired against `LayerCompatibilityChecker`.

**Dependencies:** Phase 4 (hardware profiles), Phase 5 (IMR from `ModelLoader`).

**Expected Outputs:**
- `AnalysisResult` correctly populated (`layer_graph_summary`, `parameter_count`, `estimated_flops`, `estimated_memory_bytes`, `op_type_histogram`) for every fixture model.
- `CompatibilityReport` correctly flags at least one known-unsupported case per hardware class (e.g. an LSTM layer against `artix7`, per its documented constraint in `05_Hardware_Profile_Spec.md` §2).
- `strict_compatibility_mode` toggle behavior verified end-to-end against `06_Config_Spec.md` §3.

---

## Phase 7 — Quantization Subsystem

**Objectives:** Implement the full quantization decision-and-execution chain.

**Deliverables** (in `04_Data_Flow.md` §5 order):
- `domain/quantization/quantization_advisor.py` — `QuantizationAdvisor implements PipelineStage`.
- `domain/quantization/calibration_engine.py` — `CalibrationEngine implements PipelineStage`.
- `domain/quantization/sensitivity_analyzer.py` — `SensitivityAnalyzer implements PipelineStage`.
- `domain/quantization/quantization_engine.py` — `QuantizationEngine implements PipelineStage`.
- At least one concrete `IQuantizationStrategy` implementation under `infrastructure/plugins/builtin/quantization/` (e.g. uniform INT8 strategy) to prove the interface end-to-end.
- `config/quantization.json` populated per `06_Config_Spec.md` §5.
- `datasets/calibration/<sample>/` fixture data.

**Dependencies:** Phase 6 (`AnalysisResult`, `HardwareProfile`, `CompatibilityReport`).

**Expected Outputs:**
- Full chain runs on a fixture model: `QuantizationConfig` → `CalibrationStats` → `SensitivityReport` → quantized `IMR`, matching the data-flow contract in `04_Data_Flow.md` §5 exactly (payload types, context keys).
- `QuantizationAdvisor` correctly rejects/overrides a `default_precision` not in `HardwareProfile.supported_precisions` (`05_Hardware_Profile_Spec.md` §6.2), with a logged warning.
- Contract test suite for `IQuantizationStrategy`.

---

## Phase 8 — Compression Subsystem

**Objectives:** Implement compression recommendation and execution, operating on the already-quantized IMR.

**Deliverables:**
- `domain/compression/compression_advisor.py` — `CompressionAdvisor implements PipelineStage`.
- `domain/compression/compression_engine.py` — `CompressionEngine implements PipelineStage`.
- At least one concrete `ICompressionStrategy` implementation (e.g. pruning) under `infrastructure/plugins/builtin/compression/`.
- `config/compression.json` populated per `06_Config_Spec.md` §4.

**Dependencies:** Phase 7 (quantized IMR).

**Expected Outputs:**
- `CompressionConfig` correctly derived from `config/compression.json` defaults plus `CompressionAdvisor` per-run recommendation.
- `min_acceptable_accuracy_delta` guardrail verified: advisor never recommends a combination that would (per its own projection) breach the configured delta.
- Contract test suite for `ICompressionStrategy`.

---

## Phase 9 — Optimization & Memory Planning

**Objectives:** Multi-objective optimization and target-specific memory layout, the last IMR-mutating stages before export.

**Deliverables:**
- `domain/optimization/optimization_engine.py` — `OptimizationEngine implements PipelineStage`.
- `domain/optimization/memory_optimizer.py` — `MemoryOptimizer implements PipelineStage`.

**Dependencies:** Phase 8 (compressed IMR), Phase 4 (`HardwareProfile.tensor_memory_bytes` ceiling per `05_Hardware_Profile_Spec.md` §6.4).

**Expected Outputs:**
- `MemoryOptimizer` correctly raises `HardwareIncompatibilityError` when a fixture model's activation arena projection exceeds a deliberately undersized test `HardwareProfile.tensor_memory_bytes`.
- Peak-memory budget check (`01_Project_Architecture.md` §17) verified before this stage executes.

---

## Phase 10 — Exporter & Backends

**Objectives:** Produce hardware-native deployment artifacts for all 12 supported profiles.

**Deliverables:**
- `domain/export/exporter.py` — `Exporter implements PipelineStage`.
- All 12 `IExporterBackend` implementations under `infrastructure/exporter_backends/{fpga,embedded,raspberrypi}/` (`04_Data_Flow.md` §8 table): `Artix7Backend`, `Zynq7000Backend`, `KintexBackend`, `CycloneVBackend`, `Esp32Backend`, `Esp32S3Backend`, `Stm32F4Backend`, `Stm32H7Backend`, `Rp2040Backend`, `PortentaH7Backend`, `RaspberryPi4Backend`, `RaspberryPi5Backend`.

**Dependencies:** Phase 9 (final IMR), Phase 4 (hardware profiles), Phase 3 (`FilesystemRepository` for `outputs/`).

**Expected Outputs:**
- One successful export per hardware profile against at least one fixture model, producing the correct file extensions per `01_Project_Architecture.md`/`02_Folder_Structure.md` §11 (`.mem`/`.hex`/`.bin` for FPGA, `.tflite`/`.bin`/`.h` for embedded, `.tflite`/`.onnx` for Raspberry Pi).
- `ExportError` correctly raised (never silent truncation) on a deliberately oversized fixture vs. `max_model_size_bytes` (`05_Hardware_Profile_Spec.md` §6.3).
- Every FPGA backend correctly raises `ExportError` on simulated `fpga_resources` overflow (§6.5).
- `tests/integration/exporter_backends/` passing for all 12 backends.

---

## Phase 11 — Evaluation & Benchmark

**Objectives:** Prove correctness and performance of exported artifacts.

**Deliverables:**
- `domain/evaluation/evaluator.py` — `Evaluator implements PipelineStage`.
- `domain/benchmark/benchmarker.py` — `Benchmarker implements PipelineStage`.
- `config/benchmark.json` populated per `06_Config_Spec.md` §6.
- `datasets/evaluation/<sample>/` fixture data.

**Dependencies:** Phase 10 (`DeploymentArtifact`).

**Expected Outputs:**
- `EvaluationResult.accuracy_delta` computed against the pre-quantization IMR baseline for at least one fixture per input format.
- `Benchmarker` simulation-fallback path verified (`simulation_fallback_enabled=true`) since CI has no physical hardware attached; `use_real_hardware_if_available` path documented as manually verified only (out of scope for automated CI, noted explicitly in test docstrings).
- `BenchmarkError` raised on simulated per-trial timeout.

---

## Phase 12 — Advisory & Reporting

**Objectives:** Turn raw results into human-facing scores, recommendations, and documents.

**Deliverables:**
- `domain/advisory/deployment_readiness_scorer.py` — `DeploymentReadinessScorer implements PipelineStage`.
- `domain/advisory/optimization_advisor.py` — `OptimizationAdvisor implements PipelineStage`.
- `domain/reporting/report_generator.py` — `ReportGenerator implements PipelineStage`.
- Five `IReportRenderer` implementations (markdown, per `06_Config_Spec.md` §7 v1 scope) — accuracy, benchmark, compression, deployment, summary — under `infrastructure/plugins/builtin/` or a dedicated `reporting` plugin namespace (exact location confirmed against `03_API_Specification.md` at implementation time; not a new top-level folder).
- `config/reports.json` populated per `06_Config_Spec.md` §7.

**Dependencies:** Phase 11 (`EvaluationResult`, `BenchmarkResult`).

**Expected Outputs:**
- All five report files generated under `reports/<run_id>/` for a full fixture run, matching the report-to-source mapping table in `04_Data_Flow.md` §12.
- Contract test suite for `IReportRenderer` across all five renderers.
- `include_raw_metrics_appendix` toggle verified.

---

## Phase 13 — Pipeline Orchestration & Application Layer

**Objectives:** Wire every Domain stage built in Phases 5–12 into a single ordered, resumable pipeline.

**Deliverables:**
- `domain/pipeline/pipeline_context.py`, `pipeline_stage.py` (abstract, already referenced by every prior phase but formally finalized here), `pipeline_builder.py`, `pipeline_orchestrator.py`.
- `application/session_manager.py`, `application/workflow_controller.py`.
- `infrastructure/plugins/plugin_registry.py` — `PluginRegistry` (enables the `IQuantizationStrategy`/`ICompressionStrategy`/`IExporterBackend`/`IReportRenderer` plugin points exercised in Phases 7, 8, 10, 12).
- `infrastructure/metrics/metrics_collector.py` — `MetricsCollector` (telemetry, reserved per `06_Config_Spec.md` §2, out of export scope for v1).

**Dependencies:** Phases 5–12 (every `PipelineStage` implementation must exist to be orchestrated).

**Expected Outputs:**
- `tests/integration/pipeline_end_to_end/` — the full 20-step workflow from `01_Project_Architecture.md` §5 runs start-to-finish for at least one model per input format and one hardware profile per `HardwareClass` (per `07_Coding_Standards.md` §10 integration test requirement).
- `PipelineOrchestrator.resume(run_id)` verified against a deliberately interrupted run (checkpoint/resume idempotency, `04_Data_Flow.md` §14 rule 5).
- `ExecutionConfig.on_error` policies (`ABORT`, `SKIP_STAGE`, `BEST_EFFORT`) each verified with a deliberately failing stage.
- `max_concurrent_runs` cap verified via `SessionManager`.

---

## Phase 14 — Interface Layer & Composition Root

**Objectives:** Expose the fully orchestrated pipeline to the outside world.

**Deliverables:**
- `interface/composition_root.py` — the sole wiring point (`01_Project_Architecture.md` §"Dependency Injection Strategy").
- `interface/cli/cli_entry_point.py`.
- `interface/api/rest_entry_point.py` and `interface/api/schemas/`.
- `interface/batch/batch_runner.py`.

**Dependencies:** Phase 13 (a fully working `PipelineOrchestrator`).

**Expected Outputs:**
- CLI: a full run triggerable end-to-end from the command line against a fixture model, producing `outputs/<run_id>/` and `reports/<run_id>/`.
- REST: the equivalent triggerable via HTTP request against a local test server, response schema validated against `interface/api/schemas/`.
- Batch: `BatchRunner` fans out ≥2 independent `PipelineOrchestrator` runs from one batch job file, verified against the horizontal-scalability contract in `01_Project_Architecture.md` §18.
- `import-linter` confirms only `interface.composition_root` imports concrete `infrastructure` classes (`07_Coding_Standards.md` §5).

---

## Phase 15 — Hardening, Full Test Suite & Coverage Gates

**Objectives:** Bring the whole codebase up to the coverage and quality bars defined in `07_Coding_Standards.md` before calling v1 complete.

**Deliverables:**
- Coverage backfill to reach ≥85% on `uaqe.domain`/`uaqe.application`, ≥70% on `uaqe.infrastructure` (§10).
- Full `mypy --strict` pass with zero un-justified `# type: ignore`.
- Full `import-linter` layering pass (no `domain`/`common` → `infrastructure`/`interface` violations anywhere).
- Negative-path test sweep: every documented `UAQEError` subclass has at least one test that triggers it via its documented trigger condition (cross-referenced against `03_API_Specification.md` §1.4 and every "raises" note in `04_Data_Flow.md`).

**Dependencies:** Phase 14 (everything exists to be hardened).

**Expected Outputs:**
- Green CI on `main`: lint, `mypy --strict`, `import-linter`, full `pytest` suite (unit + integration), coverage gate.
- A coverage report artifact attached to the CI run.

---

## Phase 16 — Documentation, Samples & v1 Release

**Objectives:** Make the platform usable and reviewable by someone outside the implementation team.

**Deliverables:**
- `README.md` quickstart (install, run a sample model against a sample hardware profile, inspect outputs/reports).
- `docs/generated/` API reference (auto-generated from docstrings per `07_Coding_Standards.md` §9 — mechanism TBD at implementation time, out of `09_Architecture_Lock.md` scope).
- `scripts/setup_dev_env.sh`, `run_lint.sh`, `run_tests.sh`, `generate_hardware_profile_template.py` finalized and tested.
- Tagged `v1.0.0` release.

**Dependencies:** Phase 15.

**Expected Outputs:**
- A fresh clone + `scripts/setup_dev_env.sh` + README quickstart steps reproduces a working run with no undocumented manual steps.
- `v1.0.0` tag with `09_Architecture_Lock.md` referenced as the frozen contract for all `v1.x` patch/minor work.

---

## Phase Dependency Summary

```
Phase 1  (Docs/Skeleton)
   │
Phase 2  (common)
   │
Phase 3  (logging/config infra) ──────────────┐
   │                                           │
Phase 4  (hardware profiles) ──────────────────┤
   │                                           │
Phase 5  (ingestion/adapters) ◄────────────────┘
   │
Phase 6  (validation/analysis/compatibility)
   │
Phase 7  (quantization)
   │
Phase 8  (compression)
   │
Phase 9  (optimization/memory)
   │
Phase 10 (export/backends)
   │
Phase 11 (evaluation/benchmark)
   │
Phase 12 (advisory/reporting)
   │
Phase 13 (orchestration/application)
   │
Phase 14 (interface/composition root)
   │
Phase 15 (hardening/coverage)
   │
Phase 16 (docs/release)
```

This is a strict topological order matching the dependency-inversion rule in `01_Project_Architecture.md` §2: no phase implements a class whose constructor-injected collaborators don't already exist as tested interfaces or concrete classes from an earlier phase.

---

## Document Control

| Field | Value |
|---|---|
| Depends on | `01` through `07` |
| Consumed by | `09_Architecture_Lock.md`, `10_Module_Development_Guide.md`, all future sprint/ticket planning |
| Change policy | Phase order and dependencies are structural and should not change post-lock; deliverable checklists within a phase may be refined via normal PR review (not an RFC) since they do not alter locked names/schemas |

**End of `08_Project_Roadmap.md`.**
Awaiting approval to proceed to `09_Architecture_Lock.md`.
