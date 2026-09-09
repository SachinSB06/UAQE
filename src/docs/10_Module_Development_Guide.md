# 10_Module_Development_Guide.md
## Universal AI Quantization Engine — Module Development Guide

**Depends on:** `01_Project_Architecture.md` through `09_Architecture_Lock.md`
**Status:** DRAFT — pending Architecture Lock ratification (this document is normative once `09_Architecture_Lock.md` is approved)
**Audience:** Any developer, contractor, or Claude conversation implementing a module against this architecture, in isolation, without access to any other module's implementation.
**Purpose:** For every module defined in `02_Folder_Structure.md`, give an implementer everything needed to build it correctly on the first attempt: purpose, responsibilities, exact input/output contracts (cross-referenced to `03_API_Specification.md`), dependencies, expected classes/methods, future extension points, testing strategy, and a pre-merge integration checklist.

**How to use this document:** Find your assigned module below. Read only `01`, `02`, `03` (your module's section), `04` (your module's flow section), `05`/`06` if hardware/config-touching, `07` (always), `09` (always), and this document's entry for your module. You do not need to read any other module's implementation to build yours correctly — that is the entire point of the interface contracts in `03_API_Specification.md`.

---

## 0. Universal Rules for Every Module

Before any module-specific guidance, these rules apply to all modules without exception:

1. **Build against interfaces, not against other modules' implementations.** If your module needs `ILogger`, `IHardwareProfileRepository`, etc., you receive it via constructor injection (`01_Project_Architecture.md` §15). Never import a concrete class from another module to "peek" at its behavior.
2. **Never modify a locked name.** If `09_Architecture_Lock.md` lists your class/method/field, that name is final. If you believe it is wrong, file an RFC (`09_Architecture_Lock.md` §14) — do not rename unilaterally.
3. **Every public method signature must match `03_API_Specification.md` exactly**, including parameter names, types, return types, and declared exceptions.
4. **Every module ships with its mirrored test module** (`02_Folder_Structure.md` §14) in the same PR. A module without tests does not merge (`07_Coding_Standards.md` §10).
5. **Every module logs through `ILogger`, never `print()`.**
6. **Every module raises only `UAQEError` subclasses** across its public boundary (`07_Coding_Standards.md` §7).
7. **No module reads a config file directly.** Config Value Objects arrive via constructor injection, sourced from `ConfigRepository` by the `CompositionRoot`.

---

## 1. `uaqe.common`

### Purpose
Framework-agnostic primitives — the vocabulary every other module speaks. Contains no business logic and no I/O.

### Responsibilities
- Define `IMR` and its constituent types (`imr.py`).
- Define all shared enums (`types.py`).
- Define immutable config value objects (`value_objects.py`).
- Define the `UAQEError` exception hierarchy (`exceptions.py`).
- Define result/report dataclasses (`result_types.py`).
- Define every port (interface) consumed elsewhere (`interfaces/`).

### Input / Output
This module produces types consumed by every other layer; it has no runtime "input" of its own beyond being imported. See `03_API_Specification.md` §1 for the complete contract of every class listed below.

### Dependencies
None inside `uaqe`. Standard library only (`dataclasses`, `enum`, `typing`, `abc`).

### Expected Classes / Methods
`IMR`, `IMRLayer`, `IMRTensor`, `IMRMetadata`; enums `Precision`, `HardwareClass`, `CompressionType`, `ExportFormat`, `PipelineErrorPolicy`; value objects `QuantizationConfig`, `CompressionConfig`, `HardwareConfig`, `OptimizationConfig`, `ExecutionConfig`; `UAQEError` and its twelve subclasses; `StageResult`, `RunResult`, `CompatibilityReport`; interfaces `ILogger`, `IFrameworkAdapter`, `IQuantizationStrategy`, `ICompressionStrategy`, `IExporterBackend`, `IReportRenderer`, `IHardwareProfileRepository`, `IConfigRepository` — exact signatures per `03_API_Specification.md` §1.

### Future Extensions
- New `Precision`/`CompressionType`/`ExportFormat` members require an RFC (`09_Architecture_Lock.md` §14) since these enums are locked.
- New `UAQEError` subclasses may be added additively without an RFC as long as they inherit `UAQEError` and set `code`/`message`/`stage`.
- New interfaces may be added additively for new plugin extension points (`01_Project_Architecture.md` §11).

### Testing Strategy
- Pure unit tests: construct each value object/dataclass and assert immutability (`frozen=True` raises `FrozenInstanceError` on mutation attempt).
- `IMR.topological_order()` tested against hand-built graphs including diamond dependencies and single-node graphs.
- No mocks needed — this module has no collaborators.
- Coverage target: 85%+ (falls under `uaqe.domain`-tier coverage bar per `07_Coding_Standards.md` §10, since it is foundational to everything).

### Integration Checklist
- [ ] Every class name matches `03_API_Specification.md` §1 exactly.
- [ ] Every enum member matches `09_Architecture_Lock.md` §3/§9 exactly.
- [ ] No third-party imports beyond stdlib.
- [ ] `mypy --strict` passes with zero errors.
- [ ] 1:1 test file mirror exists for every file under `common/`.

---

## 2. `uaqe.domain.pipeline`

### Purpose
The orchestration backbone: defines what a pipeline stage is, how stages are sequenced, and how run state accumulates.

### Responsibilities
- `PipelineOrchestrator`: executes an ordered list of stages against a `PipelineContext`, handling `on_error` policy (`01_Project_Architecture.md` §12).
- `PipelineContext`: append-only run state container.
- `PipelineStage`: abstract base every domain stage implements.
- `PipelineBuilder`: constructs the correct ordered stage list per `HardwareClass` and config (Builder Pattern, `01_Project_Architecture.md` §9).

### Input / Output
- Input: `RunRequest` (from `uaqe.application`), `ExecutionConfig`.
- Output: `RunResult` containing all `StageResult`s, errors, and output paths.
- Full sequence: `01_Project_Architecture.md` §8; stage order is locked in `09_Architecture_Lock.md` §12.

### Dependencies
`uaqe.common` only. `PipelineOrchestrator` receives concrete `PipelineStage` instances via constructor injection from `CompositionRoot` — it never instantiates a stage itself.

### Expected Classes / Methods
```
PipelineOrchestrator
  + run() -> RunResult
  + resume(run_id: str) -> RunResult
PipelineContext
  + append(stage_name: str, result: StageResult) -> None
  + get(stage_name: str) -> StageResult
PipelineStage {abstract}
  + execute(context: PipelineContext) -> StageResult
PipelineBuilder
  + build(hardware_class: HardwareClass, config: ExecutionConfig) -> List[PipelineStage]
```

### Future Extensions
- Hardware-class-specific stage insertion (e.g., FPGA bitstream packaging) is done inside `PipelineBuilder` only — no change to `PipelineOrchestrator`.
- Remote/distributed stage dispatch (`01_Project_Architecture.md` §10.6) will extend `PipelineOrchestrator` behind the same `run()` contract; do not pre-build this now, but do not add anything that would block it (e.g., do not hold non-serializable state on `PipelineContext`).

### Testing Strategy
- Unit test `PipelineOrchestrator` against fake `PipelineStage` doubles that succeed, fail, and raise each `on_error` branch (`ABORT`, `SKIP_STAGE`, `BEST_EFFORT`).
- Unit test `resume()` against a `PipelineContext` checkpoint fixture with a partially completed stage list.
- Integration test (`tests/integration/pipeline_end_to_end/`) runs a real stage list built by `PipelineBuilder` against one fixture model per `HardwareClass`.

### Integration Checklist
- [ ] `PipelineOrchestrator` never imports a concrete stage class.
- [ ] Stage order produced by `PipelineBuilder` matches `09_Architecture_Lock.md` §12 exactly for every `HardwareClass`.
- [ ] `on_error` behavior matches `06_Config_Spec.md` §1 and `01_Project_Architecture.md` §12.
- [ ] Every stage transition is logged at `INFO`; every failure at `ERROR`.

---

## 3. `uaqe.domain.model`

### Purpose
Loading raw models into IMR and validating/analyzing them, independent of source framework.

### Responsibilities
- `ModelLoader`: delegates to the correct `IFrameworkAdapter` based on file extension, produces `IMR`.
- `ModelValidator`: structural sanity checks on `IMR` (no dangling edges, no cycles, valid tensor shapes).
- `ModelAnalyzer`: computes layer graph stats, parameter counts, FLOPs, memory footprint, op-type inventory.

### Input / Output
- Input: filesystem path (via `ModelIngestionService`), list of injected `IFrameworkAdapter` instances.
- Output: `IMR` (loader), `StageResult` wrapping validation errors or an `AnalysisResult` payload (analyzer). See `04_Data_Flow.md` §2–§4.

### Dependencies
`uaqe.common`. Receives `List[IFrameworkAdapter]` via constructor injection — never imports `torch_adapter.py` or any concrete adapter directly (those live in `uaqe.infrastructure`, which `uaqe.domain` may not import per `01_Project_Architecture.md` §7 prohibited edges).

### Expected Classes / Methods
```
ModelLoader implements PipelineStage
  - adapters: List[IFrameworkAdapter]
  + execute(context) -> StageResult
    raises ModelLoadError
ModelValidator implements PipelineStage
  + execute(context) -> StageResult
    raises ModelValidationError
ModelAnalyzer implements PipelineStage
  + execute(context) -> StageResult
```

### Future Extensions
- New input formats require only a new `IFrameworkAdapter` in `uaqe.infrastructure.framework_adapters`; `ModelLoader` requires no change (it iterates injected adapters and calls `.supports(extension)`).
- Analysis metrics (e.g., estimated MACs per layer) may be added additively to `AnalysisResult` payload fields without breaking downstream consumers, provided existing fields are not removed or retyped (`09_Architecture_Lock.md` §9).

### Testing Strategy
- `ModelLoader` tested against stub `IFrameworkAdapter`s (never real `torch`/`tf` — that belongs to infrastructure contract tests).
- `ModelValidator` tested against hand-crafted malformed IMR fixtures (dangling input, cyclic graph, zero-size tensor).
- `ModelAnalyzer` tested against `tests/fixtures/sample_models/` IMR snapshots with known expected parameter counts.

### Integration Checklist
- [ ] `ModelLoader` never catches a raw framework exception — that conversion happens inside the adapter (`07_Coding_Standards.md` §7).
- [ ] `ModelValidator` runs before `ModelAnalyzer` in every stage list (`09_Architecture_Lock.md` §12).
- [ ] No import of any class under `uaqe.infrastructure` anywhere in this package.

---

## 4. `uaqe.domain.compatibility`

### Purpose
Determine whether a model, as analyzed, can run on a candidate hardware target.

### Responsibilities
- `LayerCompatibilityChecker`: cross-references `AnalysisResult` op types against `HardwareProfile` constraints and produces `CompatibilityReport`.

### Input / Output
- Input: `IMR` (post-analysis), `HardwareProfile`.
- Output: `CompatibilityReport { compatible, unsupported_layers, constraint_violations }`.

### Dependencies
`uaqe.common` only.

### Expected Classes / Methods
```
LayerCompatibilityChecker implements PipelineStage
  + execute(context) -> StageResult
  + check(imr: IMR, profile: HardwareProfile) -> CompatibilityReport
```

### Future Extensions
- New constraint categories are added as code-level checks, not by parsing new free-text patterns from `HardwareProfile.constraints` (`05_Hardware_Profile_Spec.md` §6 rule 1 is binding — constraints are advisories surfaced verbatim, never parsed).
- `strict_compatibility_mode` toggle (`06_Config_Spec.md` §3) must be honored: when `true`, any `constraint_violations` entry forces `compatible=False`.

### Testing Strategy
- Table-driven tests: one fixture IMR × one fixture `HardwareProfile` → one expected `CompatibilityReport`, across every `HardwareClass`.
- Explicit test for `strict_compatibility_mode=True` vs `False` divergence.

### Integration Checklist
- [ ] Runs after `HardwareManager` loads the target profile, before `QuantizationAdvisor` (`09_Architecture_Lock.md` §12).
- [ ] Never raises for an incompatible model — returns `compatible=False` in the report; the *caller* (`PipelineOrchestrator`) decides whether to halt.

---

## 5. `uaqe.domain.hardware`

### Purpose
Single point of truth for "what does the target hardware support."

### Responsibilities
- `HardwareManager`: resolves a `profile_id` to a `HardwareProfile` via the injected repository, exposes capability queries to every downstream advisor/engine.

### Input / Output
- Input: `profile_id: str` (from `RunRequest` or `hardware.json` default).
- Output: `HardwareProfile` (§`05_Hardware_Profile_Spec.md` §1).

### Dependencies
`uaqe.common`. Receives `IHardwareProfileRepository` via constructor injection.

### Expected Classes / Methods
```
HardwareManager implements PipelineStage
  - hardware_repo: IHardwareProfileRepository
  + execute(context) -> StageResult
  + checkCompatibility(imr: IMR, profile: HardwareProfile) -> CompatibilityReport
```
Note: `checkCompatibility` here delegates to `LayerCompatibilityChecker`; `HardwareManager` does not duplicate that logic — it is the orchestration point, per `01_Project_Architecture.md` §19 UML diagram.

### Future Extensions
- New `HardwareClass` values require the process in `05_Hardware_Profile_Spec.md` §7.2 (RFC required, since `HardwareClass` is locked).
- New profile fields are additive-only per `05_Hardware_Profile_Spec.md` §7.3.

### Testing Strategy
- Unit test against a stub `IHardwareProfileRepository` returning fixture profiles for all 12 supported boards.
- Explicit test for `default_profile_id = null` behavior (must raise `ConfigurationError` if `RunRequest.hardware_profile_id` also absent, per `06_Config_Spec.md` §3).

### Integration Checklist
- [ ] Never reads `hardware_profiles/*.json` directly — always through `IHardwareProfileRepository`.
- [ ] `profile_cache_enabled` behavior is the repository's responsibility, not `HardwareManager`'s.

---

## 6. `uaqe.domain.quantization`

### Purpose
Decide and apply precision transforms to the model.

### Responsibilities
- `QuantizationAdvisor`: recommends a `QuantizationConfig` filtered against `HardwareProfile.supported_precisions` (`05_Hardware_Profile_Spec.md` §6 rule 2 — hard requirement, not overridable by user config).
- `CalibrationEngine`: runs calibration data through the model, collects activation statistics.
- `SensitivityAnalyzer`: measures per-layer accuracy impact of candidate precisions.
- `QuantizationEngine`: applies the final per-layer precision plan to IMR via an injected `IQuantizationStrategy`.

### Input / Output
See `04_Data_Flow.md` §5 for the complete flow diagram (Advisor → Calibration → Sensitivity → Engine).
- Output of `QuantizationEngine`: a new immutable `IMR` with `IMRLayer.precision` set per layer — never mutates the input IMR in place (`01_Project_Architecture.md` §5).

### Dependencies
`uaqe.common`. `QuantizationEngine` receives `IQuantizationStrategy` via constructor injection (Strategy Pattern) — it never hardcodes an INT8/INT4 algorithm itself.

### Expected Classes / Methods
```
QuantizationAdvisor
  + recommend(imr: IMR, profile: HardwareProfile, config: QuantizationConfig) -> QuantizationConfig
CalibrationEngine
  + calibrate(imr: IMR, dataset_path: str, batch_size: int) -> CalibrationStatistics
SensitivityAnalyzer
  + analyze(imr: IMR, stats: CalibrationStatistics, threshold: float) -> SensitivityReport
QuantizationEngine implements PipelineStage
  - strategy: IQuantizationStrategy
  + execute(context) -> StageResult
    raises QuantizationError
```

### Future Extensions
- New quantization algorithms (GPTQ-style, AWQ-style) are added as new `IQuantizationStrategy` implementations in `uaqe.infrastructure.plugins.builtin.quantization/` or a third-party plugin directory — zero changes to `QuantizationEngine` (`01_Project_Architecture.md` §10.3).
- `advisor.max_precision_levels_considered` (`06_Config_Spec.md` §5) bounds search cost; any new advisor search strategy must respect this config field rather than introducing a new one without an RFC.

### Testing Strategy
- `QuantizationAdvisor`: table-driven tests asserting it never recommends a precision outside `HardwareProfile.supported_precisions`, including the case where `default_precision` from config is itself unsupported (must override + log warning, per `06_Config_Spec.md` §5).
- `QuantizationEngine`: contract test shared across every `IQuantizationStrategy` implementation (`07_Coding_Standards.md` §10) confirming interchangeability.
- Golden-value test: known small IMR fixture quantized to INT8 produces byte-identical output across repeated runs (determinism check).

### Integration Checklist
- [ ] `QuantizationAdvisor`'s recommendation is enforced, not advisory, with respect to hardware precision support (`05_Hardware_Profile_Spec.md` §6 rule 2) — this must hold even under `RunRequest.config_overrides`.
- [ ] `QuantizationEngine` output IMR is a new object; original IMR reference remains valid and unmodified in `PipelineContext` history.

---

## 7. `uaqe.domain.compression`

### Purpose
Decide and apply size-reduction transforms.

### Responsibilities
- `CompressionAdvisor`: recommends `CompressionConfig`, preferring lossless (`HUFFMAN`/`RLE`) over lossy (`PRUNING`/`WEIGHT_CLUSTERING`) when size headroom allows (`06_Config_Spec.md` §4).
- `CompressionEngine`: applies the plan via injected `ICompressionStrategy`.

### Input / Output
See `04_Data_Flow.md` §6.

### Dependencies
`uaqe.common`. Strategy Pattern, same shape as quantization.

### Expected Classes / Methods
```
CompressionAdvisor
  + recommend(imr: IMR, profile: HardwareProfile, config: CompressionConfig) -> CompressionConfig
CompressionEngine implements PipelineStage
  - strategy: ICompressionStrategy
  + execute(context) -> StageResult
    raises CompressionError
```

### Future Extensions
New `CompressionType` values require an RFC (locked enum, `09_Architecture_Lock.md` §3/§9). New compression *algorithms* implementing an existing `CompressionType` are plugin additions, no RFC needed.

### Testing Strategy
- `CompressionAdvisor` tested against `advisor.min_acceptable_accuracy_delta` guardrail (`06_Config_Spec.md` §4) — must never recommend a combination projected to exceed it.
- Contract test across all `ICompressionStrategy` implementations.

### Integration Checklist
- [ ] Compression gating against hardware is `CompressionAdvisor`'s responsibility, not `HardwareProfile`'s (per explicit note in `06_Config_Spec.md` §4) — confirm no accidental coupling.

---

## 8. `uaqe.domain.optimization`

### Purpose
Multi-objective search across accuracy/latency/memory, and target-specific memory layout planning.

### Responsibilities
- `OptimizationEngine`: runs the search defined by `OptimizationConfig.objectives`/`objective_weights`.
- `MemoryOptimizer`: determines tensor layout, buffer reuse, arena sizing against `HardwareProfile.tensor_memory_bytes` as a hard ceiling (`05_Hardware_Profile_Spec.md` §6 rule 4).

### Input / Output
See `04_Data_Flow.md` §7.

### Dependencies
`uaqe.common`.

### Expected Classes / Methods
```
OptimizationEngine implements PipelineStage
  + execute(context) -> StageResult
    raises OptimizationError
MemoryOptimizer
  + plan(imr: IMR, profile: HardwareProfile) -> MemoryPlan
    raises HardwareIncompatibilityError
```

### Future Extensions
Parallel candidate evaluation is permitted per `01_Project_Architecture.md` §16 (embarrassingly parallel search) — implementers may add a `ProcessPoolExecutor` internally as long as `OptimizationEngine.execute()`'s external contract stays synchronous and side-effect-free on `PipelineContext` until the final result is ready.

### Testing Strategy
- `MemoryOptimizer` tested against profiles with deliberately tight `tensor_memory_bytes` to confirm `HardwareIncompatibilityError` fires before any downstream stage runs (peak memory budget check, `01_Project_Architecture.md` §17).
- `OptimizationEngine` tested for search convergence within `max_search_iterations` on a fixture search space with a known optimum.

### Integration Checklist
- [ ] `MemoryOptimizer` distinguishes `tensor_memory_bytes` from `ram_bytes`/`flash_bytes` per `05_Hardware_Profile_Spec.md` §6 rule 4 — verify no accidental conflation.

---

## 9. `uaqe.domain.export`

### Purpose
Select and delegate to the correct hardware-specific exporter backend.

### Responsibilities
- `Exporter`: chooses the `IExporterBackend` matching the target `HardwareProfile`, validates final IMR size against `max_model_size_bytes` before writing (`05_Hardware_Profile_Spec.md` §6 rule 3).

### Input / Output
See `04_Data_Flow.md` §8. Output: `DeploymentArtifact` written to `outputs/<run_id>/<hardware_class>/`.

### Dependencies
`uaqe.common`. Receives `List[IExporterBackend]` via constructor injection — never imports a concrete backend (those live in `uaqe.infrastructure.exporter_backends`, off-limits to `uaqe.domain` per the prohibited-edges rule).

### Expected Classes / Methods
```
Exporter implements PipelineStage
  - backends: List[IExporterBackend]
  + execute(context) -> StageResult
    raises ExportError
```

### Future Extensions
New hardware within an existing class needs no `Exporter` change if a backend for that runtime family already exists (`05_Hardware_Profile_Spec.md` §7.1). New hardware class requires a new `IExporterBackend` implementation and RFC (§7.2).

### Testing Strategy
- `Exporter` tested with a stub backend that reports overflow, confirming `ExportError` is raised and no partial file is written (never silently truncate, per rule 3).
- Contract test shared across every `IExporterBackend` implementation.

### Integration Checklist
- [ ] Size validation happens before any write call, not after.
- [ ] `Exporter` selects backend by `HardwareProfile.hardware_class` + `profile_id`, never by string-matching a filename.

---

## 10. `uaqe.domain.evaluation` and `uaqe.domain.benchmark`

### Purpose
Prove the exported artifact is correct (`Evaluator`) and performant (`Benchmarker`).

### Responsibilities
- `Evaluator`: computes accuracy/error metrics pre- vs. post-optimization against `datasets/evaluation/`.
- `Benchmarker`: measures latency/throughput/memory, real hardware if available and `use_real_hardware_if_available=true`, else analytical simulation if `simulation_fallback_enabled=true`, else `BenchmarkError` (`06_Config_Spec.md` §6).

### Input / Output
See `04_Data_Flow.md` §9–§10.

### Dependencies
`uaqe.common`. `Benchmarker` device-detection mechanism is infrastructure-specific per backend (`06_Config_Spec.md` §6 note) — the domain-level `Benchmarker` calls an injected capability, it does not itself probe hardware.

### Expected Classes / Methods
```
Evaluator implements PipelineStage
  + execute(context) -> StageResult
    raises EvaluationError
Benchmarker implements PipelineStage
  + execute(context) -> StageResult
  + run_benchmark(artifact: DeploymentArtifact, trials: int) -> BenchmarkResult
    raises BenchmarkError
```

### Future Extensions
Additional metrics (e.g., energy consumption per inference) are additive fields on `BenchmarkResult`, never a breaking rename.

### Testing Strategy
- `Evaluator` tested against fixture pre/post IMR pairs with known accuracy deltas.
- `Benchmarker` tested for both branches: `simulation_fallback_enabled=True/False` with no device attached.
- Warmup trials excluded from aggregate statistics — explicit test for this.

### Integration Checklist
- [ ] `Benchmarker` never computes accuracy; `Evaluator` never runs on-device timing (strict responsibility split, `01_Project_Architecture.md` §6).
- [ ] Timeout handling (`timeout_seconds_per_trial`) raises `BenchmarkError`, never hangs the pipeline.

---

## 11. `uaqe.domain.advisory`

### Purpose
Aggregate everything learned into a human-actionable score and recommendation set.

### Responsibilities
- `DeploymentReadinessScorer`: produces a `ReadinessScore` from `CompatibilityReport`, evaluation, and benchmark results.
- `OptimizationAdvisor`: aggregates all prior `StageResult`s into recommendations, without re-running any prior stage (`01_Project_Architecture.md` §6).

### Input / Output
See `04_Data_Flow.md` §11.

### Dependencies
`uaqe.common` only — this module reads exclusively from `PipelineContext`, never triggers new computation.

### Expected Classes / Methods
```
DeploymentReadinessScorer
  + score(context: PipelineContext) -> ReadinessScore
OptimizationAdvisor implements PipelineStage
  + execute(context) -> StageResult
```

### Future Extensions
New scoring dimensions are additive fields on `ReadinessScore`; scoring weights should be config-driven (extend `06_Config_Spec.md` if a new tunable is needed, with a doc update).

### Testing Strategy
- Pure function tests: given a fixed `PipelineContext` fixture, `score()` and `execute()` must be deterministic and side-effect-free.

### Integration Checklist
- [ ] Confirm zero calls into any other domain module's `execute()` — this stage only *reads* `PipelineContext`.

---

## 12. `uaqe.domain.reporting`

### Purpose
Render the five report types from final `PipelineContext` state.

### Responsibilities
- `ReportGenerator`: dispatches to injected `IReportRenderer` implementations per `reports.json` → `enabled_reports` (`06_Config_Spec.md` §7).

### Input / Output
See `04_Data_Flow.md` §12. Output: files under `reports/<run_id>/`.

### Dependencies
`uaqe.common`. Receives `List[IReportRenderer]` via constructor injection.

### Expected Classes / Methods
```
ReportGenerator implements PipelineStage
  - renderers: List[IReportRenderer]
  + execute(context) -> StageResult
```

### Future Extensions
New report types (`html`/`json` output formats reserved per `06_Config_Spec.md` §7) are new `IReportRenderer` plugins; `ReportGenerator` requires no change.

### Testing Strategy
- Contract test shared across every `IReportRenderer` implementation.
- Snapshot tests per report type against a fixed `PipelineContext` fixture.
- Explicit test that a renderer not listed in `enabled_reports` is skipped even if registered (`06_Config_Spec.md` §7 note).

### Integration Checklist
- [ ] `ReportGenerator` computes no new metrics — every value rendered must already exist in `PipelineContext` (`01_Project_Architecture.md` §6, "Must NOT Do" column).

---

## 13. `uaqe.application`

### Purpose
Session/run lifecycle orchestration above the pipeline; the seam between Interface and Domain.

### Responsibilities
- `ModelIngestionService`: handles the upload step, invokes `ModelLoader` indirectly via the pipeline.
- `HardwareSelectionService`: resolves target hardware selection (explicit or `default_profile_id`).
- `SessionManager`: tracks `active_runs`, enforces `max_concurrent_runs` (`06_Config_Spec.md` §2).
- `WorkflowController`: top-level entry point called by the Interface Layer; builds a `RunRequest` and hands it to `PipelineOrchestrator`.

### Input / Output
- Input: raw request from Interface Layer (file path/bytes, hardware selection, config overrides).
- Output: `RunResult`.

### Dependencies
`uaqe.domain`, `uaqe.common`. **No ML logic, no hardware logic** — this layer only knows *order*, never *content* (`01_Project_Architecture.md` §3.2).

### Expected Classes / Methods
```
ModelIngestionService
  + ingest(file_path: str) -> IMR
HardwareSelectionService
  + resolve(requested_profile_id: Optional[str], default_profile_id: Optional[str]) -> str
    raises ConfigurationError
SessionManager
  - active_runs: Dict[str, RunResult]
  + start_run(request: RunRequest) -> str
  + get_status(run_id: str) -> RunResult
WorkflowController
  + execute(request: RunRequest) -> RunResult
RunRequest (DTO)
  + model_path: str
  + hardware_profile_id: Optional[str]
  + config_overrides: Dict[str, Any]
```

### Future Extensions
Async/queued run execution (beyond in-memory `active_runs` dict) is a natural extension point but out of scope for v1 — do not introduce a message queue dependency without an RFC, since it would touch `SessionManager`'s locked shape.

### Testing Strategy
- `SessionManager` tested for the `max_concurrent_runs` boundary (request N+1 queues or rejects per implementation choice — document whichever is chosen, since `06_Config_Spec.md` does not specify overflow behavior explicitly; raise this as a documentation gap if ambiguous rather than guessing silently).
- `WorkflowController` tested end-to-end against a stubbed `PipelineOrchestrator`.

### Integration Checklist
- [ ] Zero imports from `uaqe.infrastructure` anywhere in this layer.
- [ ] `RunRequest.config_overrides` precedence matches `06_Config_Spec.md` §8 exactly.

---

## 14. `uaqe.infrastructure.framework_adapters`

### Purpose
Convert real framework model files into `IMR`.

### Responsibilities
One adapter per format: `torch_adapter.py` (`.pth`/`.pt`), `onnx_adapter.py` (`.onnx`), `tensorflow_adapter.py` (`.pb`), `keras_adapter.py` (`.h5`/`.keras`), `tflite_adapter.py` (`.tflite`). Each implements `IFrameworkAdapter`.

### Input / Output
Input: filesystem path. Output: `IMR`. Must catch every third-party exception and re-raise as `ModelLoadError` with context (`07_Coding_Standards.md` §7) — no raw `torch`/`tf`/`onnx` exception may cross this boundary.

### Dependencies
`uaqe.common`. Third-party ML libraries are scoped exclusively to their own adapter file (`07_Coding_Standards.md` §14 rule 4) — `torch` may only appear in `torch_adapter.py`'s dependency group.

### Expected Classes / Methods
```
TorchAdapter implements IFrameworkAdapter
  + load(path: str) -> IMR
    raises ModelLoadError
  + supports(extension: str) -> bool
```
(Same shape for each of the other four adapters, per `03_API_Specification.md` §5.)

### Future Extensions
New frameworks (JAX, PaddlePaddle) are new files in this folder implementing `IFrameworkAdapter` — zero changes elsewhere (`01_Project_Architecture.md` §10.1).

### Testing Strategy
- Contract test suite (shared base, `07_Coding_Standards.md` §10) run against all five adapters using `tests/fixtures/sample_models/`.
- Explicit negative test: a corrupted/truncated file of each format must raise `ModelLoadError`, never a raw framework exception, never a silent partial `IMR`.
- Coverage bar: 70% (infrastructure tier, `07_Coding_Standards.md` §10).

### Integration Checklist
- [ ] Confirm the adapter's third-party dependency appears only in its own `pyproject.toml` dependency group.
- [ ] `supports()` matches purely on extension string, case-insensitively.

---

## 15. `uaqe.infrastructure.exporter_backends`

### Purpose
Hardware-native artifact writers, one per board/family, grouped by `hardware_class`.

### Responsibilities
FPGA backends (`artix7_backend.py`, `zynq7000_backend.py`, `kintex_backend.py`, `cyclonev_backend.py`) MUST read `HardwareProfile.fpga_resources` and raise `ExportError` on projected overflow rather than hard-coding resource limits (`05_Hardware_Profile_Spec.md` §6 rule 5). Embedded backends (`esp32_backend.py`, `esp32s3_backend.py`, `stm32f4_backend.py`, `stm32h7_backend.py`, `rp2040_backend.py`, `portenta_h7_backend.py`) write `.tflite`/`.bin`/`.h`. Raspberry Pi backends (`raspberrypi4_backend.py`, `raspberrypi5_backend.py`) write `.tflite`/`.onnx`.

### Input / Output
Input: quantized+compressed+optimized `IMR`, `HardwareProfile`. Output: `DeploymentArtifact` written to `outputs/<run_id>/<hardware_class>/`.

### Dependencies
`uaqe.common`; hardware-vendor toolchains/SDKs as needed, scoped to the individual backend file.

### Expected Classes / Methods
```
Artix7Backend implements IExporterBackend
  + export(imr: IMR, target: HardwareProfile) -> DeploymentArtifact
    raises ExportError
  + supported_targets() -> List[str]
```
(Same shape for every other backend file, per `03_API_Specification.md` §5.)

### Future Extensions
A new board within an existing family is a new backend file implementing `IExporterBackend`; the `Exporter` domain class discovers it via injection at `CompositionRoot` wiring time, no domain change required.

### Testing Strategy
- Contract test suite across all twelve backends confirming interchangeable `IExporterBackend` behavior.
- FPGA-specific test: fixture IMR sized to deliberately exceed `fpga_resources.bram_kb` must raise `ExportError`, never emit a partial `.mem`/`.hex`/`.bin` file.
- Embedded/RPi-specific test: output file is valid for its declared `ExportFormat` (e.g., `.tflite` output is parseable by a TFLite reader in the test harness).

### Integration Checklist
- [ ] No backend hard-codes a resource ceiling — every limit is read from the injected `HardwareProfile`.
- [ ] Every backend's `supported_targets()` return value matches its file's `profile_id` exactly as listed in `05_Hardware_Profile_Spec.md` §3–§5.

---

## 16. `uaqe.infrastructure.repositories`

### Purpose
Concrete, file-backed implementations of the repository interfaces.

### Responsibilities
- `ConfigRepository` implements `IConfigRepository`: loads/validates every file in `06_Config_Spec.md`, enforces override precedence (§8) and `schema_version` checks (§9).
- `HardwareProfileRepository` implements `IHardwareProfileRepository`: loads/caches `hardware_profiles/**/*.json` per `05_Hardware_Profile_Spec.md` §2, enforces `schema_version` major-version rejection (§7.3).
- `FilesystemRepository`: generic read/write helper for `workdir/`, `outputs/`, `reports/`, `logs/` paths, enforcing the single-writer-per-folder rule (`02_Folder_Structure.md` §1, §19).

### Input / Output
Input: file paths under `config/` and `hardware_profiles/`. Output: strongly-typed Config Value Objects and `HardwareProfile` instances — never raw dicts crossing into `uaqe.domain` (`01_Project_Architecture.md` §14).

### Dependencies
`uaqe.common`; `json`/`yaml` parsing libraries.

### Expected Classes / Methods
```
ConfigRepository implements IConfigRepository
  - cache: Dict[str, Any]
  + load_quantization_config(overrides: Dict[str, Any]) -> QuantizationConfig
  + load_compression_config(overrides: Dict[str, Any]) -> CompressionConfig
  + load_hardware_config(overrides: Dict[str, Any]) -> HardwareConfig
  + load_execution_config(overrides: Dict[str, Any]) -> ExecutionConfig
    raises ConfigurationError
HardwareProfileRepository implements IHardwareProfileRepository
  - cache: Dict[str, HardwareProfile]
  + get(profile_id: str) -> HardwareProfile
  + list_all() -> List[HardwareProfile]
    raises ConfigurationError
```

### Future Extensions
Swapping the backing store (e.g., a database-backed `HardwareProfileRepository` instead of JSON files) is a drop-in replacement implementing the same interface — no change to any Domain consumer, by design.

### Testing Strategy
- `ConfigRepository`: test every precedence tier in `06_Config_Spec.md` §8 explicitly, including the "missing required file raises `ConfigurationError`" case.
- `HardwareProfileRepository`: test `schema_version` major-mismatch rejection, and `profile_cache_enabled=False` bypassing the cache.
- Round-trip test: every JSON file physically present under `config/` and `hardware_profiles/` in the repo must parse without error in CI (schema drift guard).

### Integration Checklist
- [ ] Every enum-valued config field is validated against its `uaqe.common.types` enum at load time (`06_Config_Spec.md` §9 rule 2), not deferred.
- [ ] `null`-valued fields in hardware profile JSON deserialize correctly to `Optional[...]=None`, never a sentinel string.

---

## 17. `uaqe.infrastructure.logging` and `uaqe.infrastructure.metrics`

### Purpose
Concrete `ILogger` implementation and metrics collection.

### Responsibilities
- `StructuredLogger` implements `ILogger`: JSON-lines output to `logs/<run_id>/pipeline.log` and/or stdout per `settings.yaml` → `logging.sink` (`06_Config_Spec.md` §2).
- `MetricsCollector`: reserved for telemetry (`06_Config_Spec.md` §2, `telemetry.*`) — out of scope for v1 export, but the interface/class shell must exist per folder structure.

### Input / Output
Input: log calls (`msg`, `**fields`) from every module. Output: JSON-lines file + optional stdout stream.

### Dependencies
`uaqe.common`; stdlib `json`, `pathlib`.

### Expected Classes / Methods
```
StructuredLogger implements ILogger
  - min_level: str
  - sink_paths: List[str]
  + debug(msg: str, **fields) -> None
  + info(msg: str, **fields) -> None
  + warning(msg: str, **fields) -> None
  + error(msg: str, **fields) -> None
  + critical(msg: str, **fields) -> None
```

### Future Extensions
Per-module log-level overrides (`01_Project_Architecture.md` §13, e.g. `QuantizationEngine: DEBUG`) are a `settings.yaml` schema extension — update `06_Config_Spec.md` before implementing.

### Testing Strategy
- Assert every log call includes `run_id` and `stage_name` where applicable (`07_Coding_Standards.md` §6).
- Assert `min_level` filtering suppresses lower-severity calls without writing them.

### Integration Checklist
- [ ] No log call anywhere in the codebase bypasses `ILogger` (grep-based CI check recommended).
- [ ] JSON-lines output is valid JSON per line, parseable by `json.loads` without pre-processing.

---

## 18. `uaqe.infrastructure.plugins`

### Purpose
Discover and register third-party/custom strategy, adapter, backend, and renderer plugins.

### Responsibilities
- `PluginRegistry`: scans the configured `plugins_path` (`06_Config_Spec.md` §1) at startup and registers discovered implementations of the four/five extension-point interfaces (`01_Project_Architecture.md` §11).
- `builtin/quantization/`, `builtin/compression/`: ship UAQE's own default strategies, registered the same way as third-party plugins (dogfooding the plugin mechanism, per Open/Closed Principle).

### Input / Output
Input: `plugins_path` directory tree. Output: populated strategy/backend/renderer registries consumed by `CompositionRoot`.

### Dependencies
`uaqe.common`.

### Expected Classes / Methods
```
PluginRegistry
  + discover(plugins_path: str) -> None
  + get_quantization_strategies() -> List[IQuantizationStrategy]
  + get_compression_strategies() -> List[ICompressionStrategy]
  + get_exporter_backends() -> List[IExporterBackend]
  + get_report_renderers() -> List[IReportRenderer]
    raises PluginLoadError
```

### Future Extensions
This is itself the primary future-extension mechanism (`01_Project_Architecture.md` §10) — no further seam needed beyond keeping `PluginRegistry`'s discovery mechanism stable.

### Testing Strategy
- Test discovery against a fixture plugin directory containing one valid and one deliberately malformed plugin (missing required interface method) — malformed plugin must raise `PluginLoadError`, not crash the whole registry.
- Confirm built-in strategies register identically to third-party ones (no special-casing).

### Integration Checklist
- [ ] `PluginRegistry` is the one intentional global/singleton permitted by `01_Project_Architecture.md` §15 — confirm no *other* module introduces a second global.

---

## 19. `uaqe.interface`

### Purpose
External entry points and the sole Composition Root.

### Responsibilities
- `composition_root.py`: the only file in the entire codebase permitted to instantiate concrete `uaqe.infrastructure` classes and wire them into `uaqe.domain`/`uaqe.application` interfaces (`01_Project_Architecture.md` §15, `07_Coding_Standards.md` §5).
- `cli/cli_entry_point.py`: CLI invocation → `WorkflowController.execute()` → stdout/exit code.
- `api/rest_entry_point.py` + `api/schemas/`: HTTP request → `WorkflowController.execute()` → JSON response.
- `batch/batch_runner.py`: fans out independent `PipelineOrchestrator` runs for batch jobs (`01_Project_Architecture.md` §18).

### Input / Output
Input: CLI args, HTTP request bodies, or batch job files. Output: stdout/exit code, HTTP JSON, or batch summary — translated from `RunResult`, never a raw stack trace (`01_Project_Architecture.md` §12).

### Dependencies
All inner layers. This is the only layer permitted to import concrete `uaqe.infrastructure` classes.

### Expected Classes / Methods
```
CompositionRoot
  + build_workflow_controller(settings: Settings) -> WorkflowController
CliEntryPoint
  + main(argv: List[str]) -> int
RestEntryPoint
  + handle_run_request(payload: dict) -> dict
BatchRunner
  + run_batch(job_file_path: str) -> List[RunResult]
```

### Future Extensions
New entry points (e.g., a gRPC server) are new files under `interface/`, each still routing through the single `CompositionRoot`-built `WorkflowController` — never a second wiring point.

### Testing Strategy
- `CompositionRoot` tested for successful wiring against real config fixtures (this is the one place an integration-style test is appropriate at "unit" granularity, since its whole job is wiring).
- Each entry point tested against a stubbed `WorkflowController`, confirming correct translation of user-facing errors (structured `code`/`message`/`remediation_hint`, never a raw exception).

### Integration Checklist
- [ ] Grep confirms no `import uaqe.infrastructure` anywhere outside `interface/composition_root.py`.
- [ ] Every entry point converts `UAQEError` into its channel-appropriate representation (exit code, HTTP status, batch summary line) without leaking a raw traceback to the end user.

---

## 20. Cross-Module Integration Checklist (Run Before Any Merge to `main`)

This supplements the per-module checklists above and mirrors the reviewer checklist in `07_Coding_Standards.md` §14:

1. [ ] Signature matches `03_API_Specification.md` exactly.
2. [ ] No Data Flow Rule from `04_Data_Flow.md` §14 is violated.
3. [ ] Test coverage meets the tier bar in `07_Coding_Standards.md` §10 (85% domain/application, 70% infrastructure).
4. [ ] No name, folder, schema, or interface listed in `09_Architecture_Lock.md` was altered without an approved RFC.
5. [ ] `import-linter` passes (no prohibited layer edges per `01_Project_Architecture.md` §7).
6. [ ] `mypy --strict` passes with no unexplained `# type: ignore`.
7. [ ] Every new exception raised is a `UAQEError` subclass with `code`, `message`, `stage` set.
8. [ ] Every new log call goes through injected `ILogger`.
9. [ ] Every new config field, if any, is documented in `06_Config_Spec.md` in the same PR.
10. [ ] Every new hardware-affecting behavior respects `05_Hardware_Profile_Spec.md` §6's five enforcement-contract rules.

---

## 21. Document Control

| Field | Value |
|---|---|
| Depends on | `01` through `09` (all prior documents) |
| Consumed by | Every implementing developer/Claude conversation, per-module |
| Change policy | This document may be extended (new modules, clarified guidance) freely; it may not contradict `09_Architecture_Lock.md`. Any apparent conflict between this document and `09_Architecture_Lock.md` is resolved in favor of `09_Architecture_Lock.md`, and this document must be corrected. |

**End of `10_Module_Development_Guide.md`.**

**This completes the full `01`–`10` architecture document set for the Universal AI Quantization Engine.**
