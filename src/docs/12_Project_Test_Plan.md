# 12_Project_Test_Plan.md
## Universal AI Quantization Engine — Verification & Validation (Test) Plan

**Depends on:** `01_Project_Architecture.md` through `11_Implementation_Rules.md`
**Status:** FINAL — Architecture Lock compliant
**Scope:** The complete testing strategy for the Universal AI Quantization Engine, covering every module in `02_Folder_Structure.md`, every contract in `03_API_Specification.md`, every hardware target in `05_Hardware_Profile_Spec.md`, and every rule in `11_Implementation_Rules.md`.

---

## 1. Testing Objectives

| Objective | What It Means Here | Primary Evidence |
|---|---|---|
| **Reliability** | The pipeline produces a deterministic, correctly-ordered `RunResult` for a given `RunRequest` and fixture model, run after run | Regression suite (§9), pipeline integration tests (§4) |
| **Accuracy** | Optimized models stay within documented accuracy tolerance of the original model | Accuracy Validation (§12) |
| **Performance** | Latency, memory, and export time stay within Quality Gate thresholds (§19) across supported hardware classes | Performance Testing (§7) |
| **Maintainability** | Every module has 1:1 mirrored tests (`07_Coding_Standards.md` §10) so a future change surfaces its blast radius immediately | Unit Testing Plan (§3) |
| **Robustness** | The system fails safely (raises a typed `UAQEError`, never crashes uncleanly or produces silently corrupt output) on malformed, oversized, or unsupported input | Stress Testing (§8), Security Testing (§13) |
| **Scalability** | Concurrent runs and parallel-eligible stages (`quantization`, `compression`, `benchmark`) do not degrade correctness under `execution.max_concurrent_runs` | Non-Functional Testing (§6), Stress Testing (§8) |

---

## 2. Testing Scope

Every module in `src/uaqe/` receives, at minimum, a unit test suite; modules that participate in the pipeline additionally receive integration coverage. Modules not in the pipeline (repositories, factories) receive contract and unit coverage.

| Module | Layer | Test Types Applied |
|---|---|---|
| Model Loader | domain/model | Unit, Integration (per input format) |
| Validator (`model_validator.py`) | domain/model | Unit, Integration |
| Analyzer (`model_analyzer.py`) | domain/model | Unit |
| Layer Compatibility Checker | domain/compatibility | Unit, Integration (per hardware profile) |
| Hardware Manager | domain/hardware | Unit, Integration |
| Quantization Advisor / Engine | domain/quantization | Unit, Integration, Accuracy Validation |
| Calibration Engine | domain/quantization | Unit, Stress (large calibration sets) |
| Sensitivity Analyzer | domain/quantization | Unit |
| Compression Advisor / Engine | domain/compression | Unit, Integration, Accuracy Validation |
| Optimization Engine | domain/optimization | Unit, Integration |
| Memory Optimizer | domain/optimization | Unit, Stress (memory ceilings) |
| Exporter (+ all backends) | domain/export, infrastructure/exporter_backends | Unit, Integration, Hardware Validation, Golden Output |
| Evaluator | domain/evaluation | Unit, Accuracy Validation |
| Benchmarker | domain/benchmark | Unit, Integration, Performance |
| Deployment Readiness Scorer / Optimization Advisor | domain/advisory | Unit |
| Report Generator (+ renderers) | domain/reporting | Unit, Integration, Golden Output |
| Framework Adapters | infrastructure/framework_adapters | Unit (mocked), Integration (per format), Contract |
| Exporter Backends | infrastructure/exporter_backends | Unit (mocked), Contract, Hardware Validation |
| Repositories (`Config`, `HardwareProfile`, `Filesystem`) | infrastructure/repositories | Unit, Contract, Security |
| Structured Logger | infrastructure/logging | Unit |
| Metrics Collector | infrastructure/metrics | Unit |
| Plugin Registry | infrastructure/plugins | Unit, Integration |
| Application services (`ModelIngestionService`, `HardwareSelectionService`, `SessionManager`, `WorkflowController`) | application | Unit, Integration |
| Interface entry points (CLI, REST, Batch, `CompositionRoot`) | interface | Integration, Acceptance |

---

## 3. Unit Testing Plan

Every unit test suite below follows the Arrange–Act–Assert structure, naming convention, and mocking rules from `11_Implementation_Rules.md` §14.

**Model Loader**
- Loads each supported extension via the correct `IFrameworkAdapter` (mocked adapter, verifying factory resolution logic only).
- Raises `ModelLoadError` for an unrecognized extension.
- Raises `ModelLoadError` (not a raw framework exception) when the mocked adapter raises.

**Validator**
- Accepts a well-formed IMR fixture.
- Rejects an IMR with a missing required field, raising `ValidationError` with `stage="validation"`.
- Confirms `validate()` never mutates the IMR it inspects.

**Analyzer**
- Produces correct layer-count/parameter-count/op-type summary statistics against a known fixture IMR.
- Handles a single-layer minimal IMR without division-by-zero or empty-collection errors.

**Hardware Manager**
- Resolves a `HardwareProfile` via the mocked `IHardwareProfileRepository` for every `profile_id` in §11.
- Raises `ConfigurationError` when asked to resolve an unregistered `profile_id`.

**Quantization**
- `QuantizationAdvisor` filters candidate precisions against a mocked `HardwareProfile.supported_precisions`, never recommending an unsupported precision.
- `QuantizationEngine` delegates entirely to the resolved `IQuantizationStrategy` (verified via a strategy test double asserting it — not the engine — performed the transform).
- Per-layer override application from `SensitivityReport` is verified against a fixture with at least one flagged layer.

**Calibration**
- Batches calibration samples at exactly `calibration_batch_size`.
- Iterates as a generator (verified via a fixture dataset larger than memory would reasonably allow if materialized, using a counting iterator test double).

**Sensitivity Analysis**
- Flags a layer as sensitive when its measured accuracy delta exceeds `sensitivity_threshold`; does not flag when within threshold.

**Compression**
- `CompressionAdvisor` respects `min_acceptable_accuracy_delta` guardrail, rejecting a strategy combination that would exceed it (mocked accuracy projection).
- `CompressionEngine` applies strategies in the exact order the advisor returns.

**Optimization**
- `OptimizationEngine` composes quantization + compression outputs without re-invoking either engine directly (collaborator test doubles verify no duplicate calls).

**Memory Optimization**
- `MemoryOptimizer` raises when a fixture activation-arena plan exceeds `tensor_memory_bytes` from a mocked `HardwareProfile`.
- Treats `tensor_memory_bytes` as distinct from `ram_bytes`/`flash_bytes` per `05_Hardware_Profile_Spec.md` §6.4 (verified with a fixture where the two values differ).

**Exporter**
- Selects the correct `IExporterBackend` via the mocked `ExporterFactory` for each `(HardwareClass, profile_id)` pair.
- Raises `ExportError` (not a silent truncation) when serialized IMR size exceeds `max_model_size_bytes`.

**Evaluation**
- Computes accuracy/precision/recall/F1 correctly against a small fixed-answer fixture dataset with a known confusion matrix.

**Benchmark**
- Excludes `warmup_trials` from aggregate statistics.
- Raises `BenchmarkError` on simulated per-trial timeout.
- Selects real-hardware vs. simulation path correctly based on mocked device-detection and config flags.

**Advisor (Deployment Readiness / Optimization Advisor)**
- Produces a readiness score that decreases monotonically as injected constraint-violation counts increase (property-style unit test).

**Reports**
- Each `IReportRenderer` produces output containing every field present in its input `StageResult`/`RunResult` payload (schema-completeness check), for the `enabled_reports` list.

**Visualization**
- (If present in `03_API_Specification.md`'s reporting contracts) chart/plot-generation helpers produce valid, non-empty output for a minimal fixture dataset and raise a typed error for an empty dataset rather than emitting a malformed artifact.

**Utilities**
- Shared helpers in `uaqe.common` (constants, type coercion helpers) are tested for every documented edge case (e.g., `None`-safe field handling per `05_Hardware_Profile_Spec.md` §2's "never omitted, only null" rule).

---

## 4. Integration Testing

### 4.1 Pipeline Tests
Full pipeline runs (`tests/integration/pipeline_end_to_end/`) against `tests/fixtures/sample_models/`, covering the full cross-product of supported input format × `HardwareClass`, per `11_Implementation_Rules.md` §15.1. Each run asserts: correct stage ordering, a `RunResult` with `status=SUCCESS`, and a non-empty output artifact under `outputs/<run_id>/`.

### 4.2 Framework Adapter Tests
Each `IFrameworkAdapter` is run against a real (tiny) fixture file of its format, verifying the produced IMR matches expected layer count/shape — not mocked at this level, since the adapter's entire job is the real parse.

### 4.3 Hardware Compatibility Tests
`LayerCompatibilityChecker` is run against fixture IMRs containing at least one deliberately unsupported op/layer per hardware class (e.g., an LSTM layer against `artix7`, per its documented HDL-backend constraint), asserting `CompatibilityReport.compatible=False` and the correct `constraint_violations` entry.

### 4.4 Exporter Tests
Every `IExporterBackend` is exercised end-to-end from a post-optimization IMR fixture through to a written output file, asserting the file exists, is non-empty, and (where feasible) round-trips through a format-appropriate parser.

### 4.5 Report Tests
`ReportGenerator` is run against a complete fixture `RunResult`, asserting every file listed in `enabled_reports` (`06_Config_Spec.md` §7) is written under `reports/<run_id>/` with the expected filename from `02_Folder_Structure.md` §12.

---

## 5. Functional Testing

Functional tests verify user-observable correctness for each pipeline capability, independent of internal structure:

- **Model Loading:** every documented input extension loads successfully; every undocumented extension is rejected with a clear error.
- **Validation:** a structurally invalid model (fixture with a corrupted layer graph) is rejected before any downstream stage runs.
- **Analysis:** analysis output is present and internally consistent (e.g., sum of per-layer parameter counts equals the reported total).
- **Quantization:** the requested/advised precision is actually applied to the output IMR's layers.
- **Compression:** the achieved compression ratio is measured and reported, and is consistent with the strategies actually applied.
- **Optimization:** memory-optimized output respects the target hardware's `tensor_memory_bytes` ceiling.
- **Export:** the exported artifact's file extension(s) match the hardware class's documented output formats (FPGA: `.mem`/`.hex`/`.bin`; Embedded: `.tflite`/`.bin`/`.h`; Raspberry Pi: `.tflite`/`.onnx`).
- **Reports:** every enabled report type is generated and is non-empty for a successful run.

---

## 6. Non-Functional Testing

| Category | What Is Verified |
|---|---|
| Performance | Latency and export time stay within Quality Gate thresholds (§19) |
| Memory | Peak process memory stays within a documented multiple of `HardwareProfile.tensor_memory_bytes`/model size for the largest supported fixture |
| CPU | No pipeline stage exhibits pathological (super-linear in unexpected ways) CPU scaling as fixture model size grows across the sample set |
| Scalability | `execution.max_concurrent_runs` concurrent runs complete without cross-run state leakage (verified via distinct `run_id` isolation checks) |
| Availability | REST/Batch entry points remain responsive to new run requests while an existing run is in progress, up to the configured concurrency limit |
| Reliability | Repeated identical runs against the same fixture produce identical `RunResult` status and equivalent output artifacts (deterministic-output check, accounting for any documented non-determinism such as trial-based benchmarking) |
| Maintainability | CI enforces the 1:1 source/test file mapping and coverage thresholds from `07_Coding_Standards.md` §10 on every PR |

---

## 7. Performance Testing

| Metric | Measured At | Method |
|---|---|---|
| Latency | Per-stage (`StageResult.duration_ms`) and end-to-end (`RunResult` total) | Wall-clock timing around `execute()`, aggregated across a fixed fixture set |
| Memory Usage | Peak resident memory during pipeline execution | Process-level memory sampling during integration test runs |
| Model Size | Pre- vs. post-optimization serialized IMR/output size | Direct byte-size comparison, recorded per run |
| Compression Ratio | `original_size / compressed_size` | Computed by `CompressionEngine`, cross-checked independently in the test assertion |
| Export Time | Time from `Exporter.export()` invocation to output file(s) fully written | Wall-clock timing, per hardware backend |
| Pipeline Execution Time | Total `RunResult` duration | End-to-end integration test timing, tracked over time for trend regression |

Performance test results are recorded per CI run and compared against the Quality Gates in §19; a run exceeding a gate fails the build rather than merely emitting a warning.

---

## 8. Stress Testing

| Condition | Fixture / Method | Expected Behavior |
|---|---|---|
| Very Large Models | Synthetic fixture at/above `large_model_threshold_mb` | Completes using the large-model threading strategy, or fails with a typed `UAQEError`, never an unhandled `MemoryError`/crash |
| Corrupted Models | Deliberately truncated/malformed fixture files per format | `ModelLoadError` raised at the adapter boundary, never a raw parser exception or silent partial load |
| Unsupported Operators | Fixture IMR containing an op type absent from every backend's supported set | `CompatibilityReport` correctly flags it; pipeline halts before export per `strict_compatibility_mode` |
| Huge Tensor Sizes | Synthetic fixture with an extreme single-layer tensor shape | Memory Optimizer / Exporter reject with `OptimizationError`/`ExportError` rather than attempting an allocation that could exhaust system memory |
| Memory Overflow | Constrained test-harness memory limit (e.g., cgroup/container limit in CI) against a large fixture | Graceful `UAQEError` termination with `cleanup()` still executed, no orphaned `workdir/` state |
| Hardware Overflow | Fixture IMR exceeding `max_model_size_bytes` / `fpga_resources` capacity for a given profile | `ExportError` raised before any output file is written, per `05_Hardware_Profile_Spec.md` §6.3/§6.5 |
| Large Calibration Datasets | Synthetic calibration set far larger than default batch expectations | `CalibrationEngine` streams without materializing the full set (verified via the generator-based unit test in §3, exercised at integration scale) |

---

## 9. Regression Testing

**Strategy:** Every merged defect fix includes a permanent regression test (`11_Implementation_Rules.md` §15.5), named after its tracking issue, placed in the appropriate unit or integration suite.

**Automation:** The full regression suite runs on every PR via CI (`pytest`, coverage, import-linter, mypy, lint — see §15); no regression test is ever marked skip/xfail without a linked follow-up ticket and an expiry review date.

**Baseline Comparison:** Performance and output-size metrics from §7 are compared against the last known-good baseline stored per release tag; a statistically significant regression (beyond a documented tolerance band) fails the build even if no functional test fails.

---

## 10. Compatibility Testing

| Dimension | Verified Against |
|---|---|
| Operating Systems | Windows, Linux, macOS — CI matrix runs the full unit + integration suite on all three |
| Python Versions | 3.11 and 3.12, per `07_Coding_Standards.md` §1 — CI matrix includes both |
| Supported Frameworks | All input formats: `.pth`, `.pt`, `.onnx`, `.pb`, `.h5`, `.keras`, `.tflite` |
| Supported Hardware | Every `profile_id` catalogued in `05_Hardware_Profile_Spec.md` §3–§5 |

---

## 11. Hardware Validation

Validation procedures for every catalogued profile, exercised through the Exporter and, where a real-hardware harness is available, the Benchmarker's real-hardware path:

| Profile | Validation Focus |
|---|---|
| Artix-7 | HDL export within `logic_cells`/`bram_kb`/`dsp_slices`/`lut_count`; no floating-point layers reach export (fully quantized, per constraint) |
| Zynq-7000 | HDL export at larger resource envelope; `MIXED` precision path exercised |
| Cyclone V | HDL export within Intel-specific resource profile |
| ESP32 | `.tflite`/`.bin`/`.h` export within `tensor_memory_bytes` (262,144 B) and `max_model_size_bytes` (1,048,576 B) |
| ESP32-S3 | As ESP32, plus `INT4` precision path |
| STM32F4 | Export within the tightest embedded envelope (131,072 B tensor memory) — primary stress case for Memory Optimizer |
| STM32H7 | Export with `INT4` precision at a larger envelope |
| RP2040 | `.tflite`/`.bin`/`.h` export within its documented envelope |
| Arduino Portenta H7 | Export including `FP16` precision path, the only embedded profile supporting it |
| Raspberry Pi 4 | `.tflite`/`.onnx` export; `tflite-runtime` benchmarking path |
| Raspberry Pi 5 | As Pi 4, at the larger documented envelope |

Each profile's validation run asserts: successful export within resource limits for a compliant fixture, and a correctly raised `ExportError` for a fixture deliberately exceeding that profile's limits (paired positive/negative case per profile).

---

## 12. Accuracy Validation

For every fixture model taken through quantization and/or compression, the Evaluator compares the original and optimized model on a fixed evaluation fixture dataset:

| Metric | Comparison |
|---|---|
| Accuracy | Original vs. optimized, delta computed |
| Precision | Per-class and macro-averaged, original vs. optimized |
| Recall | Per-class and macro-averaged, original vs. optimized |
| F1 Score | Per-class and macro-averaged, original vs. optimized |

**Tolerance Limits:** The measured accuracy delta must not exceed the negative bound configured in `compression.json`'s `advisor.min_acceptable_accuracy_delta` (default `-0.02`) for the compression path, and must not exceed the `sensitivity_threshold` (`quantization.json`, default `0.01`) per flagged layer for the quantization path.

**Acceptance Thresholds:** A fixture run whose measured delta exceeds its configured tolerance fails the accuracy validation suite; this is treated as a functional defect in the responsible `Advisor`, not an acceptable trade-off to silently document after the fact.

---

## 13. Security Testing

| Threat | Test Approach | Expected Behavior |
|---|---|---|
| Malformed Models | Same fixtures as Stress Testing §8 | Typed `UAQEError`, no crash, no partial/corrupt output file left in `outputs/` |
| Invalid Inputs | Fuzzed `RunRequest` fields (invalid `hardware_profile_id`, malformed `config_overrides` dotted paths) | `ConfigurationError`/`ValidationError` at the earliest possible boundary, never a downstream `KeyError`/`AttributeError` |
| Corrupted Files | Truncated/bit-flipped fixture files | Handled identically to Malformed Models above |
| Path Traversal | Crafted `run_id`/filename inputs attempting `../` escape from `workdir`/`outputs`/`reports`/`logs` roots | `FilesystemRepository`'s path-containment check (`11_Implementation_Rules.md` §5.3) rejects the path before any file operation occurs |
| Large File Attacks | Fixture file far exceeding any plausible model size, supplied as input | Rejected early (size pre-check before full parse) with a typed error, not an attempted full load |
| Configuration Validation | Malformed/incomplete `config/*.json` or `hardware_profiles/**/*.json` files | `ConfigurationError` raised at load time per `06_Config_Spec.md` §9 / `05_Hardware_Profile_Spec.md` §7.3, never deferred to point of use |

---

## 14. Acceptance Testing

A release candidate is accepted only when all of the following hold:

1. Full CI matrix (§10) is green: all unit, integration, contract, regression, and stress suites pass.
2. All Quality Gates in §19 are met on the latest measured run.
3. Hardware Validation (§11) positive and negative cases pass for every catalogued profile.
4. Accuracy Validation (§12) passes for every fixture/hardware combination in the pipeline integration matrix.
5. Security Testing (§13) shows no unhandled exception path and no path-traversal escape.
6. No open defect classified as build-blocking per `11_Implementation_Rules.md` §2.1.
7. Documentation set (`01`–`12`) is internally consistent — no acceptance-testing discovery contradicts a locked artifact without a corresponding filed RFC.

---

## 15. Continuous Integration Testing

CI (GitHub Actions, per `07_Coding_Standards.md` §11) runs on every PR and on `main`:

1. **Static Analysis / Linting:** `ruff`/`flake8` and `isort`/`black --check`.
2. **Type Checking:** `mypy --strict` across `src/uaqe/`.
3. **Import Layering:** `import-linter` verifying the dependency-direction rules in `07_Coding_Standards.md` §5.
4. **pytest:** full unit + integration + contract suite, matrixed across OS (§10) and Python version (§10).
5. **Coverage:** `pytest-cov`, enforcing the 85%/70% thresholds; PR fails if either drops below threshold.
6. **Automated Validation:** the 1:1 source/test file mapping check (`07_Coding_Standards.md` §10) and the shared parametrized contract tests for every `I*` interface implementation (built-in and plugin).

Any red CI stage blocks merge; no stage is optional or advisory.

---

## 16. Test Directory Structure

Mirrors and extends `02_Folder_Structure.md` §14:

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
├── performance/
│   └── benchmark_baselines/
├── stress/
│   ├── large_models/
│   ├── corrupted_inputs/
│   └── resource_overflow/
├── hardware/
│   ├── fpga/
│   ├── embedded/
│   └── raspberrypi/
└── fixtures/
    ├── sample_models/
    ├── sample_hardware_profiles/
    ├── calibration/
    ├── evaluation/
    └── golden_outputs/
```

`performance/`, `stress/`, and `hardware/` are additive to the structure already locked in `02_Folder_Structure.md` §14; they live under `tests/` (already an owned, writable folder for test authors) and introduce no new top-level project folder, so no Architecture Lock RFC is required.

---

## 17. Test Data Management

| Data Type | Location | Management Rule |
|---|---|---|
| Sample Models | `tests/fixtures/sample_models/` | Kept intentionally tiny (CI-fast); one per supported input format at minimum, plus deliberately malformed variants for stress/security suites |
| Calibration Data | `tests/fixtures/calibration/` (small) and synthetic-generated (large, for stress tests) | Small fixtures checked into the repo; large synthetic sets generated at test time, never checked in |
| Golden Outputs | `tests/fixtures/golden_outputs/` | One reference output per `IExporterBackend`/`IReportRenderer`; regenerated only via an explicit, reviewed script when an intentional output-format change is approved |
| Hardware Profiles | `tests/fixtures/sample_hardware_profiles/` | Minimal synthetic profiles for unit tests; real `hardware_profiles/**/*.json` files used directly for Hardware Validation (§11) integration tests |
| Expected Reports | Included within `golden_outputs/` | Report renderer output compared structurally (fields present/populated), not byte-for-byte, to avoid brittleness on cosmetic formatting changes |
| Benchmark Data | `tests/performance/benchmark_baselines/` | Stores the last accepted performance baseline per metric per release tag, used for the regression comparison in §9 |

---

## 18. Testing Checklist

Before a PR touching test code (or any code requiring new tests) is submitted:

- [ ] New/changed source file has a corresponding new/updated unit test file
- [ ] Arrange–Act–Assert structure and naming convention followed (`11_Implementation_Rules.md` §14)
- [ ] Mocks used only at interface boundaries, not concrete infrastructure classes
- [ ] New `I*` interface implementation has its shared contract test wired in
- [ ] Integration test added/updated if the change affects pipeline stage sequencing or a hardware backend
- [ ] Stress/security implications considered for any change touching file I/O, model parsing, or path handling
- [ ] Coverage thresholds still met locally before pushing
- [ ] No test marked skip/xfail without a linked ticket

---

## 19. Quality Gates

| Gate | Threshold |
|---|---|
| Minimum Code Coverage | 85% (`uaqe.domain`, `uaqe.application`), 70% (`uaqe.infrastructure`) |
| Maximum Accuracy Loss | Within `compression.json`'s `min_acceptable_accuracy_delta` (default −0.02) and `quantization.json`'s `sensitivity_threshold` (default 0.01) per §12 |
| Maximum Latency | End-to-end pipeline execution time must not regress beyond the documented tolerance band against the last accepted baseline (§9, §17) |
| Memory Threshold | Peak process memory for the largest fixture must not exceed the documented multiple of the target `HardwareProfile.tensor_memory_bytes` used for planning (Memory Optimizer's own ceiling, verified, not merely assumed) |
| Export Success Rate | 100% for all compliant fixtures across every catalogued hardware profile (§11); any failure on a compliant fixture is build-blocking |
| Pipeline Success Rate | 100% for the full pipeline integration matrix (§4.1) on every CI run |
| Benchmark Acceptance | `BenchmarkResult` aggregate statistics must be internally consistent (e.g., `warmup_trials` excluded, trial count matches config) for every profile exercised |

A release candidate failing any Quality Gate does not proceed to Acceptance Testing (§14) regardless of how many other gates pass.

---

## 20. Final Verification Plan

### 20.1 Release Certification
A release is certified only after: full CI green (§15), all Quality Gates met (§19), Acceptance Testing passed (§14), and documentation set consistency confirmed (no undocumented drift from `01`–`11`).

### 20.2 Verification Process
Verification (“was it built right?”) is satisfied by the complete automated suite: unit (§3), integration (§4), functional (§5), non-functional (§6), performance (§7), stress (§8), regression (§9), compatibility (§10), and security (§13) testing, all green in CI.

### 20.3 Validation Process
Validation (“was the right thing built?”) is satisfied by Hardware Validation (§11), Accuracy Validation (§12), and Acceptance Testing (§14) — confirming the system behaves correctly against real-world hardware constraints and real deployment expectations, not just internal contracts.

### 20.4 Production Readiness
A build is production-ready when Release Certification (§20.1) is complete and the Deployment Readiness Scorer's own self-assessment (domain/advisory) reports no unresolved blocking finding for the release's representative fixture matrix.

### 20.5 Regression Approval
Any change touching a previously-baselined performance or accuracy metric requires explicit sign-off from the owning module lead (per `02_Folder_Structure.md` ownership) referencing the specific baseline delta, before the regression baseline in `tests/performance/benchmark_baselines/` may be updated.

### 20.6 Deployment Approval
Final deployment approval requires: Release Certification (§20.1), Production Readiness (§20.4), and a second approving reviewer distinct from the release's primary implementer, per the two-approval rule already established for architecture-lock-adjacent changes in `07_Coding_Standards.md` §14.2.

---

**End of `12_Project_Test_Plan.md`.**
