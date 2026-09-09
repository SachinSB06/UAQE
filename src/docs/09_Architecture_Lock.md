# 09_Architecture_Lock.md
## Universal AI Quantization Engine — Architecture Lock

**Depends on:** `01_Project_Architecture.md` through `08_Project_Roadmap.md`
**Status:** **LOCKED** as of approval of this document. This is the single most important document in the set.
**Scope:** Freezes every structural decision made in `01`–`08` into a binding contract. From this point forward, nothing listed in §1–§13 below may change without a formal RFC (§14). Every future module, every future Claude conversation, every future contributor implements *against* this document, not around it.

---

## 0. What "Locked" Means

1. **Locked** = the exact name, signature, schema, or structural rule as written below is permanent for the `v1.x` line. Bug fixes and implementations may not silently rename, retype, relocate, or restructure anything locked.
2. **Additive change is allowed without an RFC** where explicitly marked "open for addition" (e.g. new hardware profile JSON files, new plugin implementations, new `HardwareClass`-scoped export backends). Additive means: a new file/class/enum member/JSON entry that follows an existing pattern, with zero modification to existing names.
3. **Everything else requires an RFC** filed against this document, reviewed and approved by two owners (per `07_Coding_Standards.md` §14 rule 2), before any PR touching it may merge.
4. If a locked item and an earlier document (`01`–`08`) ever disagree, **this document wins** — it is the canonical, deduplicated summary of everything those documents established.

---

## 1. Locked Folder Structure

The full tree in `02_Folder_Structure.md` §1 is locked exactly as written: every top-level folder (`src/`, `config/`, `hardware_profiles/`, `datasets/`, `workdir/`, `outputs/`, `reports/`, `logs/`, `tests/`, `docs/`, `scripts/`, `plugins/`), every `src/uaqe/{common,domain,application,infrastructure,interface}/` subfolder, and every named `.py` file within them.

**Locked, no renaming/moving/merging/splitting without RFC:**
- All folder names (`snake_case`, per `07_Coding_Standards.md` §4).
- All module (file) names listed in `02_Folder_Structure.md` §2–§6.
- The ownership table in `02_Folder_Structure.md` §19 (which module has write access to which runtime folder).

**Open for addition (no RFC needed):**
- New files under `hardware_profiles/<class>/*.json` (new boards within an existing `HardwareClass`).
- New files under `infrastructure/exporter_backends/<class>/` (new backend for a new board within an existing class).
- New files under `plugins/{quantization_strategies,compression_strategies,exporter_backends}/` (external plugin drop-ins).
- New test files under `tests/` mirroring any of the above.

---

## 2. Locked Package & Module Names

Every package path is locked exactly as it appears in `02_Folder_Structure.md` and is used verbatim throughout `03_API_Specification.md` and `04_Data_Flow.md`:

```
uaqe.common
uaqe.common.interfaces
uaqe.domain.pipeline
uaqe.domain.model
uaqe.domain.compatibility
uaqe.domain.hardware
uaqe.domain.quantization
uaqe.domain.compression
uaqe.domain.optimization
uaqe.domain.export
uaqe.domain.evaluation
uaqe.domain.benchmark
uaqe.domain.advisory
uaqe.domain.reporting
uaqe.application
uaqe.infrastructure.framework_adapters
uaqe.infrastructure.exporter_backends.{fpga,embedded,raspberrypi}
uaqe.infrastructure.repositories
uaqe.infrastructure.logging
uaqe.infrastructure.metrics
uaqe.infrastructure.plugins
uaqe.interface
uaqe.interface.cli
uaqe.interface.api
uaqe.interface.batch
```

No package may be renamed. No package may be collapsed into another. No file may move to a different package than the one listed in `02_Folder_Structure.md` without an RFC.

---

## 3. Locked Class Names

The complete class inventory, grouped by file, as defined in `03_API_Specification.md`. This is the canonical list — any class not on this list that a contributor believes is needed requires an RFC before creation.

**`uaqe.common`:** `IMR`, `IMRLayer`, `IMRTensor`, `IMRMetadata`, `QuantizationConfig`, `CompressionConfig`, `HardwareConfig`, `OptimizationConfig`, `ExecutionConfig`, `UAQEError` and its 12 subclasses (`ModelLoadError`, `ModelValidationError`, `UnsupportedLayerError`, `HardwareIncompatibilityError`, `QuantizationError`, `CompressionError`, `OptimizationError`, `ExportError`, `EvaluationError`, `BenchmarkError`, `ConfigurationError`, `PluginLoadError`), `StageResult`, `RunResult`, `CompatibilityReport`, `ILogger`, `IFrameworkAdapter`, `IQuantizationStrategy`, `ICompressionStrategy`, `IExporterBackend`, `IReportRenderer`, `IHardwareProfileRepository`, `IConfigRepository`.

**`uaqe.domain.pipeline`:** `PipelineContext`, `PipelineStage`, `PipelineBuilder`, `PipelineOrchestrator`.

**`uaqe.domain.model`:** `ModelLoader`, `ModelValidator`, `AnalysisResult`, `ModelAnalyzer`.

**`uaqe.domain.compatibility`:** `LayerCompatibilityChecker`.

**`uaqe.domain.hardware`:** `HardwareProfile`, `HardwareManager`.

**`uaqe.domain.quantization`:** `QuantizationPlan`, `QuantizationAdvisor`, `CalibrationStats`, `CalibrationEngine`, `SensitivityReport`, `SensitivityAnalyzer`, `QuantizationEngine`.

**`uaqe.domain.compression`:** `CompressionPlan`, `CompressionAdvisor`, `CompressionEngine`.

**`uaqe.domain.optimization`:** `OptimizationResult`, `OptimizationEngine`, `MemoryPlan`, `MemoryOptimizer`.

**`uaqe.domain.export`:** `DeploymentArtifact`, `Exporter`.

**`uaqe.domain.evaluation`:** `EvaluationResult`, `Evaluator`.

**`uaqe.domain.benchmark`:** `BenchmarkResult`, `Benchmarker`.

**`uaqe.domain.advisory`:** `ReadinessScore`, `DeploymentReadinessScorer`, `Recommendation`, `OptimizationAdvisor`.

**`uaqe.domain.reporting`:** `ReportDocument`, `ReportGenerator`.

**`uaqe.application`:** `RunRequest`, `ModelIngestionService`, `HardwareSelectionService`, `SessionManager`, `WorkflowController`.

**`uaqe.infrastructure.framework_adapters`:** `TorchAdapter`, `OnnxAdapter`, `TensorFlowAdapter`, `KerasAdapter`, `TFLiteAdapter`.

**`uaqe.infrastructure.exporter_backends`:** `Artix7Backend`, `Zynq7000Backend`, `KintexBackend`, `CycloneVBackend`, `Esp32Backend`, `Esp32S3Backend`, `Stm32F4Backend`, `Stm32H7Backend`, `Rp2040Backend`, `PortentaH7Backend`, `RaspberryPi4Backend`, `RaspberryPi5Backend`.

**`uaqe.infrastructure.repositories`:** `ConfigRepository`, `HardwareProfileRepository`, `FilesystemRepository`.

**`uaqe.infrastructure.logging`:** `StructuredLogger`.

**`uaqe.infrastructure.metrics`:** `MetricsCollector`.

**`uaqe.infrastructure.plugins`:** `PluginRegistry`.

**`uaqe.interface`:** `CompositionRoot`, `CliEntryPoint`, `RestEntryPoint`, `BatchRunner`.

**Open for addition (no RFC needed):** concrete `IQuantizationStrategy` / `ICompressionStrategy` / `IReportRenderer` implementations registered via `PluginRegistry` — these are plugin-space classes, not core architecture classes, provided they implement an existing locked interface without modifying it.

---

## 4. Locked Method Signatures

Every method signature in `03_API_Specification.md` §1–§21 is locked verbatim: name, parameter names, parameter types, return type, and declared `raises` clause. This includes:

- All `{abstract}` methods on every `I*` interface (§1.6–§1.13).
- `PipelineStage.execute(context: PipelineContext) -> StageResult` — this exact signature on **every** concrete stage class; no stage may add required constructor-independent parameters to `execute()`.
- `PipelineOrchestrator.run() -> RunResult` and `.resume(run_id: str) -> RunResult`.
- Every domain-logic method beyond `execute()` (e.g. `HardwareManager.check_compatibility()`, `QuantizationAdvisor.recommend()`, `Exporter.select_backend()`, `Benchmarker.run_benchmark()`) exactly as typed in `03_API_Specification.md`.
- `CompositionRoot.build_workflow_controller(config_path: str) -> WorkflowController` as the **sole** static entry point for wiring the system (§21.1).

**Locked contract rules** (restated from `03_API_Specification.md` §22, binding permanently):
1. Interface-typed constructor parameters MUST be injected by `CompositionRoot`, never instantiated internally by the receiving class.
2. Every failable method MUST raise a subclass of `UAQEError`; no bare `Exception`.
3. `PipelineStage.execute()` MUST read inputs only via `PipelineContext.get(...)` and MUST return a `StageResult`, never `None`.
4. Stage payload dataclasses are immutable; no stage mutates a prior stage's payload.
5. Additional private/protected helper methods MAY be added to any class during implementation (they are not part of the public contract) provided they do not change any locked public signature.

---

## 5. Locked Configuration Schema

Every file under `config/` and its exact field set, type, and default is locked per `06_Config_Spec.md` §1–§7:

| File | Locked top-level fields |
|---|---|
| `config.json` | `schema_version`, `app_name`, `default_run_id_prefix`, `workdir_path`, `outputs_path`, `reports_path`, `logs_path`, `hardware_profiles_path`, `plugins_path`, `large_model_threshold_mb`, `on_error` |
| `settings.yaml` | `schema_version`, `environment`, `logging.{min_level,sink,json_lines}`, `execution.parallelism.{quantization,compression,benchmark}`, `execution.max_concurrent_runs`, `telemetry.{enabled,endpoint}` |
| `hardware.json` | `schema_version`, `default_profile_id`, `profile_cache_enabled`, `strict_compatibility_mode`, `allow_partial_export_on_warning` |
| `compression.json` | `schema_version`, `enabled_types`, `target_ratio`, `pruning_sparsity`, `advisor.{prefer_lossless_when_hardware_allows,min_acceptable_accuracy_delta}` |
| `quantization.json` | `schema_version`, `default_precision`, `per_layer_overrides`, `calibration_batch_size`, `sensitivity_threshold`, `advisor.{allow_mixed_precision,max_precision_levels_considered}` |
| `benchmark.json` | `schema_version`, `trials`, `warmup_trials`, `timeout_seconds_per_trial`, `use_real_hardware_if_available`, `simulation_fallback_enabled` |
| `reports.json` | `schema_version`, `enabled_reports`, `output_format`, `include_raw_metrics_appendix` |

**Locked precedence order** (`06_Config_Spec.md` §8): `RunRequest.config_overrides` > subsystem `advisor` block > file top-level defaults > hard-coded fail-safe (v1: unused — missing file always raises `ConfigurationError`).

**Additive rule:** a new field may be added to any config file only alongside a corresponding update to `06_Config_Spec.md` and, if it maps to a locked value object in `uaqe.common.value_objects`, an RFC against this document (per `06_Config_Spec.md` §9 rule 4 — restated here as binding).

---

## 6. Locked Hardware Profile JSON Schema

The `HardwareProfile` on-disk schema (`05_Hardware_Profile_Spec.md` §2) is locked field-for-field:

```
schema_version, profile_id, display_name, hardware_class, ram_bytes, flash_bytes,
storage_bytes, tensor_memory_bytes, runtime, supported_precisions, max_model_size_bytes,
preferred_export_formats, clock_speed_hz, fpga_resources {logic_cells, bram_kb, dsp_slices, lut_count},
constraints
```

**Locked rule:** fields not applicable to a given `hardware_class` are always present and set to `null` — never omitted (`05_Hardware_Profile_Spec.md` §2). This is a parsing-safety guarantee for `HardwareProfileRepository` and may not be relaxed.

**Locked profile inventory (12 profiles, 3 classes):** `artix7`, `zynq7000`, `kintex`, `cyclonev` (FPGA); `esp32`, `esp32s3`, `stm32f4`, `stm32h7`, `rp2040`, `portenta_h7` (EMBEDDED); `raspberrypi4`, `raspberrypi5` (RASPBERRY_PI) — with the exact field values tabulated in `05_Hardware_Profile_Spec.md` §3–§5.

**Note on value-object field drift:** `03_API_Specification.md` §5.1's in-memory `HardwareProfile` dataclass lists a subset of fields (`profile_id`, `hardware_class`, `ram_bytes`, `flash_bytes`, `tensor_memory_bytes`, `runtime`, `supported_precisions`, `max_model_size_bytes`, `preferred_export_format` [singular]) compared to the full on-disk schema in `05_Hardware_Profile_Spec.md` §1 (`display_name`, `storage_bytes`, `preferred_export_formats` [plural, list], `clock_speed_hz`, `fpga_resources`, `constraints`, `schema_version` also present). **This is a known open discrepancy, not a silent inconsistency** — `05_Hardware_Profile_Spec.md` §1 is the authoritative full field set (it is the value object description, explicitly stated to belong in `uaqe.common.value_objects`); `03_API_Specification.md` §5.1 is treated as a partial/abbreviated listing. **Resolution, locked as of this document:** `HardwareProfile` carries the complete field set from `05_Hardware_Profile_Spec.md` §1. Any implementation must reconcile `03_API_Specification.md` §5.1 to match this document, not the reverse. This reconciliation itself does not require an RFC (it is a correction of an already-identified drift, not a new architectural decision) but must be applied consistently everywhere `HardwareProfile` is referenced before any code implementing `HardwareManager` or `HardwareProfileRepository` is merged.

**Open for addition (no RFC needed):** new hardware profile JSON files within an existing `hardware_class`, per `05_Hardware_Profile_Spec.md` §7.1.

**Requires RFC:** any new `HardwareClass` enum member (§7.2), any new top-level field on the schema itself (§7.3 — additive fields are allowed but MUST be `null`-defaulted across all existing files and documented in `06_Config_Spec.md` first).

---

## 7. Locked Interfaces (Ports)

The eight interfaces below are the complete, locked set of extension points in the system. No ninth core interface may be added without an RFC; new capabilities are added via new *implementations* of these interfaces, not new interfaces, wherever possible.

| Interface | Locked abstract methods |
|---|---|
| `ILogger` | `debug`, `info`, `warning`, `error`, `critical` (all `(msg: str, **fields) -> None`) |
| `IFrameworkAdapter` | `load(path: str) -> IMR`, `supports(extension: str) -> bool` |
| `IQuantizationStrategy` | `apply(imr: IMR, plan: QuantizationConfig) -> IMR`, `name() -> str` |
| `ICompressionStrategy` | `apply(imr: IMR, plan: CompressionConfig) -> IMR`, `name() -> str` |
| `IExporterBackend` | `export(imr: IMR, target: HardwareProfile) -> DeploymentArtifact`, `supported_targets() -> List[str]` |
| `IReportRenderer` | `render(context: PipelineContext) -> ReportDocument`, `report_type() -> str` |
| `IHardwareProfileRepository` | `get(profile_id: str) -> HardwareProfile`, `list_all() -> List[HardwareProfile]` |
| `IConfigRepository` | `load_quantization_config`, `load_compression_config`, `load_execution_config`, `load_optimization_config` (all `() -> <Config>`) |

---

## 8. Locked Dependency & Import Structure

The layering rule from `01_Project_Architecture.md` §2 and `07_Coding_Standards.md` §5 is locked and mechanically enforced via `import-linter`:

```
interface       ──▶ application, domain, common, infrastructure (composition_root only)
application     ──▶ domain, common
domain          ──▶ common                    [MUST NOT import infrastructure or interface]
infrastructure  ──▶ common                    [MUST NOT import domain or application]
common          ──▶ (nothing inside uaqe)
```

**Locked rules:**
1. `uaqe.domain` and `uaqe.common` MUST NOT import anything from `uaqe.infrastructure` or `uaqe.interface`.
2. Only `uaqe.interface.composition_root` may import concrete classes from `uaqe.infrastructure`.
3. Absolute imports only (`from uaqe.domain.x.y import Z`); no relative imports; no wildcard imports.
4. No circular imports; `import-linter` runs in CI as a hard gate, not a review suggestion.
5. Third-party ML/framework libraries (`torch`, `tensorflow`, `onnx`, etc.) may only appear as dependencies of files under `infrastructure/framework_adapters/`; they may never appear as an import in `uaqe.domain` or `uaqe.common`, enforced by `07_Coding_Standards.md` §14 rule 4.

---

## 9. Locked Return Types

Every stage-producing method returns exactly one of the dataclasses defined in `03_API_Specification.md`; the mapping below is locked (also cross-referenced against `04_Data_Flow.md`'s context-key table):

| Producing stage | Context key | Payload type |
|---|---|---|
| `ModelLoader` | `model_loader` | `IMR` |
| `ModelValidator` | `model_validator` | `None` |
| `ModelAnalyzer` | `model_analyzer` | `AnalysisResult` |
| `HardwareManager` | `hardware_manager` | `HardwareProfile` |
| `LayerCompatibilityChecker` | `layer_compatibility_checker` | `CompatibilityReport` |
| `QuantizationAdvisor` | `quantization_advisor` | `QuantizationConfig` / `QuantizationPlan`* |
| `CalibrationEngine` | `calibration_engine` | `CalibrationStats` |
| `SensitivityAnalyzer` | `sensitivity_analyzer` | `SensitivityReport` |
| `QuantizationEngine` | `quantization_engine` | `IMR` (quantized) |
| `CompressionAdvisor` | `compression_advisor` | `CompressionConfig` / `CompressionPlan`* |
| `CompressionEngine` | `compression_engine` | `IMR` (compressed) |
| `OptimizationEngine` | `optimization_engine` | `IMR` (optimized) / `OptimizationResult`* |
| `MemoryOptimizer` | `memory_optimizer` | `IMR` (final) / `MemoryPlan`* |
| `Exporter` | `exporter` | `DeploymentArtifact` |
| `Evaluator` | `evaluator` | `EvaluationResult` |
| `Benchmarker` | `benchmarker` | `BenchmarkResult` |
| `DeploymentReadinessScorer` | `deployment_readiness_scorer` | `ReadinessScore` |
| `OptimizationAdvisor` | `optimization_advisor` | `List[Recommendation]` |
| `ReportGenerator` | `report_generator` | `List[ReportDocument]` |

\* **Known open item, resolved here:** four stages (`QuantizationAdvisor`, `CompressionAdvisor`, `OptimizationEngine`, `MemoryOptimizer`) have a domain-level "plan/result" dataclass (`QuantizationPlan`, `CompressionPlan`, `OptimizationResult`, `MemoryPlan` — §3 above) distinct from the "config" value object consumed as input. `04_Data_Flow.md` describes the advisor stages' `StageResult.payload` as the `*Config` object; `03_API_Specification.md` defines a `recommend()`/`search()`/`plan()` method returning the richer `*Plan`/`*Result` dataclass. **Locked resolution:** the `StageResult.payload` for these four stages is the richer domain dataclass (`QuantizationPlan`, `CompressionPlan`, `OptimizationResult`, `MemoryPlan` respectively) since it is a strict superset carrying rationale/scores the downstream `Exporter` and `ReportGenerator` need; the corresponding `*Config` value object is treated as an input parameter to these stages (sourced from `IConfigRepository`), not their output. This clarification binds all future implementation and does not require further RFC to adopt, but any change to it after adoption does.

---

## 10. Locked Naming Standards

Restated as binding from `07_Coding_Standards.md` §3 — this table is now permanent:

| Element | Convention | 
|---|---|
| Module (file) | `snake_case.py` |
| Package (folder) | `snake_case` |
| Class | `PascalCase` |
| Interface (abstract class) | `IPascalCase` |
| Method / function | `snake_case()` |
| Constant | `UPPER_SNAKE_CASE` |
| Private attribute | `_leading_underscore`, documented `-` |
| Protected attribute | `_single_underscore`, documented `#` |
| Enum member | `UPPER_SNAKE_CASE` |
| Dataclass | `PascalCase`, noun phrase |

No abbreviation may be invented beyond those already present in `02_Folder_Structure.md` / `03_API_Specification.md` (e.g. `IMR` stays; no new acronyms without RFC).

---

## 11. Locked Public vs. Internal API Boundary

**Public API** (stable surface other modules and future contributors code against):
- Every method marked `+` in `03_API_Specification.md`.
- Every class listed in §3 above.
- Every interface listed in §7 above.
- `CompositionRoot.build_workflow_controller()` as the only sanctioned system bootstrap entry point.

**Internal API** (implementation detail, may change freely without RFC as long as public behavior/signatures are preserved):
- Every attribute marked `-` (private) or `#` (protected) in `03_API_Specification.md`.
- Any helper method not listed in `03_API_Specification.md`, added during implementation.
- Internal caching, batching, or algorithmic strategy inside a method body.

**Rule:** a change is "internal" only if no public signature, return type, exception type, or `PipelineContext` key changes as a result. Anything else is a public API change and requires an RFC.

---

## 12. Locked Pipeline Stage Order

The canonical 19-stage execution order (`01_Project_Architecture.md` §5, `04_Data_Flow.md` §13) is locked:

```
model_loader → model_validator → model_analyzer → hardware_manager →
layer_compatibility_checker → quantization_advisor → calibration_engine →
sensitivity_analyzer → quantization_engine → compression_advisor →
compression_engine → optimization_engine → memory_optimizer → exporter →
{evaluator, benchmarker} → deployment_readiness_scorer → optimization_advisor →
report_generator
```

`evaluator` and `benchmarker` both depend only on `exporter`'s output and `deployment_readiness_scorer` depends on both — they are the one point in the pipeline where two stages may execute independently before the next join point, per `04_Data_Flow.md` §13's diagram. `PipelineBuilder.build()` is the sole authority for materializing this order into a concrete `List[PipelineStage]`; no stage may be reordered by any other component. Stage order is `HardwareClass`-dependent only insofar as `PipelineBuilder` may select different concrete `IExporterBackend`/strategy implementations per class — the stage *sequence* itself does not vary by hardware class.

---

## 13. Locked Data Flow Rules

Restated as permanent from `04_Data_Flow.md` §14:

1. `PipelineContext` is append-only; a stage name is never overwritten once written.
2. The "current" IMR is always the latest in the chain `model_loader → quantization_engine → compression_engine → optimization_engine → memory_optimizer`; there is no fixed "the IMR" key.
3. `HardwareProfile` is resolved exactly once by `HardwareManager` and read (never re-fetched) thereafter.
4. Only `Exporter` and `ReportGenerator` (via `FilesystemRepository`) write to `outputs/` or `reports/`.
5. `PipelineOrchestrator.resume()` is idempotent per already-populated context entries; a stage never re-executes if `context.has(stage.name())` is `True`.

---

## 14. RFC Process (governs all changes to this document)

1. An RFC is a PR against this file (`09_Architecture_Lock.md`) plus the corresponding source document(s) it amends (`01`–`08`), submitted together.
2. Requires **two approvals** from module owners not both from the same module, per `07_Coding_Standards.md` §14 rule 2.
3. An RFC PR description MUST state: (a) what locked item changes, (b) why the current lock is insufficient, (c) every downstream document/file that must be updated in the same PR to stay consistent (no partial updates — this document, the amended source doc, and any code already implementing the changed contract must land together).
4. Additive changes explicitly marked "open for addition" throughout this document do **not** require an RFC — only a normal PR review.
5. This RFC process itself is locked and may only be changed by unanimous sign-off of all current module owners (not a standard RFC).

---

## 15. Document Control

| Field | Value |
|---|---|
| Depends on | `01` through `08` |
| Consumed by | `10_Module_Development_Guide.md` and all implementation work, permanently |
| Change policy | Locked; amendable only via §14 RFC process |
| Effective date | Upon approval of this document |

**End of `09_Architecture_Lock.md`.**
Awaiting approval to proceed to `10_Module_Development_Guide.md`.
