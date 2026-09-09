# 11_Implementation_Rules.md
## Universal AI Quantization Engine — Implementation Rules

**Depends on:** `01_Project_Architecture.md`, `02_Folder_Structure.md`, `03_API_Specification.md`, `04_Data_Flow.md`, `05_Hardware_Profile_Spec.md`, `06_Config_Spec.md`, `07_Coding_Standards.md`, `09_Architecture_Lock.md`, `10_Module_Development_Guide.md`
**Status:** FINAL — Architecture Lock compliant
**Scope:** Binding implementation rules for every developer, reviewer, and Claude conversation that writes code against the locked architecture. This document does not introduce new architecture; it removes ambiguity from applying the existing architecture.

---

## 1. Purpose

### 1.1 Why This Document Exists

`01_Project_Architecture.md` defines *what* the system is. `03_API_Specification.md` defines *what the contracts are*. `07_Coding_Standards.md` defines *how code looks*. None of these tell an implementer, module-by-module, *how to write the body of a method without silently drifting from the architecture*. This document closes that gap. It is the last document consulted before a single line of implementation code is written, and the first document re-consulted whenever a design question arises mid-implementation.

### 1.2 How Developers Should Use It

Before touching any module:

1. Read the module's entry in `02_Folder_Structure.md` (ownership, dependency direction).
2. Read the module's contract in `03_API_Specification.md` (exact signatures).
3. Read the module's data-flow position in `04_Data_Flow.md` (upstream/downstream stages).
4. Read this document's relevant section (pattern-level implementation guidance).
5. Read `10_Module_Development_Guide.md` for the module-specific worked example, if one exists.

If, during implementation, a rule in this document appears to conflict with a locked artifact, the locked artifact wins and an RFC is filed against `09_Architecture_Lock.md` — this document is never itself authoritative over a locked document; it is subordinate to all ten preceding documents.

### 1.3 Relationship With Previous Documents

| Document | Relationship to this document |
|---|---|
| `01_Project_Architecture.md` | Source of truth for structural patterns (layering, pipeline, plugin system) this document operationalizes |
| `02_Folder_Structure.md` | Defines *where* code implementing these rules lives |
| `03_API_Specification.md` | Defines the exact signatures these rules must be implemented against, verbatim |
| `04_Data_Flow.md` | Defines the sequencing this document's stage-lifecycle rules must respect |
| `05_Hardware_Profile_Spec.md` | Source of the hardware constraint contract referenced in §5, §6 |
| `06_Config_Spec.md` | Source of the configuration injection contract referenced in §3, §5 |
| `07_Coding_Standards.md` | Defines *style*; this document defines *behavior* |
| `09_Architecture_Lock.md` | Defines what may never change without RFC; this document never contradicts it |
| `10_Module_Development_Guide.md` | Provides worked, module-specific examples of the rules in this document |

---

## 2. Universal Implementation Rules

These rules are absolute. They apply to every module, every layer, every contributor, with zero exceptions.

1. **Never modify `09_Architecture_Lock.md`** or any artifact it locks, without a merged RFC.
2. **Never modify `03_API_Specification.md`** signatures. If a method appears to need a different signature to be implementable, the implementation approach is wrong, not the specification — escalate before changing the contract.
3. **Never modify `02_Folder_Structure.md`.** No file may be created outside its documented location; no folder may be renamed, merged, or split.
4. **Never modify the configuration schema** documented in `06_Config_Spec.md` (field names, types, file names) without a corresponding spec update and, for locked value objects, an RFC.
5. **Never rename a class** documented in `03_API_Specification.md` or `02_Folder_Structure.md`.
6. **Never rename a method** documented in `03_API_Specification.md`, including parameter names (`03_API_Specification.md` signatures are matched positionally *and* by keyword in tests).
7. **Never change a documented return type.** If a method's documented return type is `StageResult`, no implementation may return `None`, a raw `dict`, or a subclass not itself documented.
8. **Never change a package name** under `src/uaqe/` as laid out in `02_Folder_Structure.md`.
9. **Never violate a dependency-direction rule.** `uaqe.domain` and `uaqe.common` never import from `uaqe.infrastructure` or `uaqe.interface` (`07_Coding_Standards.md` §5). This is enforced by `import-linter` in CI, but code review must catch it before CI does.

### 2.1 Consequences of Violating These Rules

A violation of any rule in this section is treated as a **build-blocking defect**, not a style preference:

- A PR that violates §2.1–§2.9 is rejected in review regardless of functional correctness or test coverage — correctness against the wrong contract is not correctness.
- A violation discovered post-merge is reverted, not patched forward, unless the revert itself would break a dependent module already merged on top of it, in which case a same-day corrective PR is required and flagged in the next release's `09_Architecture_Lock.md` change log.
- Three or more architecture-lock violations from the same contributor within a rolling 90-day window trigger a mandatory review of that contributor's PR process with their module lead, per `07_Coding_Standards.md` §14.
- Any exporter or plugin that bypasses the Hardware Constraint Enforcement Contract (`05_Hardware_Profile_Spec.md` §6) is treated as a **safety defect**, not a functional one, because it can produce artifacts that silently fail or corrupt state on physical hardware.

---

## 3. Constructor Injection Rules

### 3.1 Dependency Injection

Every class in `uaqe.domain` and `uaqe.application` receives its collaborators through its constructor — never through module-level globals, service locators, or lazy internal instantiation of concrete infrastructure classes. A domain class depends only on interfaces from `uaqe.common.interfaces`; concrete infrastructure implementations are wired exclusively at `uaqe.interface.composition_root`.

### 3.2 Constructor Injection

All dependencies are passed as constructor parameters, typed against the interface, never the concrete class:

```
class QuantizationEngine
  - _logger: ILogger
  - _hardware_repo: IHardwareProfileRepository
  - _strategy_factory: IQuantizationStrategyFactory

  + __init__(logger: ILogger,
              hardware_repo: IHardwareProfileRepository,
              strategy_factory: IQuantizationStrategyFactory) -> None
```

No class performs conditional construction of its own dependencies (`if x is None: x = ConcreteThing()`). If a default is genuinely needed, the default is wired at `CompositionRoot`, never inside the consuming class.

### 3.3 Configuration Injection

Configuration value objects (`QuantizationConfig`, `CompressionConfig`, `HardwareConfig`, `ExecutionConfig`, per `03_API_Specification.md` §1.3) are **not** injected at construction time for stage classes; they are passed per-call through `PipelineContext`, because a single long-lived stage instance may process multiple runs with different configs (see §4.4). Long-lived infrastructure objects (`ConfigRepository` itself) *are* constructor-injected, since the repository, not the config values, is the stable dependency.

### 3.4 Logger Injection

Every class that logs receives an `ILogger` at construction. No class instantiates `StructuredLogger` directly except `CompositionRoot`. This guarantees every log line is routed through the single structured sink and carries consistent `run_id`/`stage_name` context (`07_Coding_Standards.md` §6).

### 3.5 Repository Injection

Repositories (`IConfigRepository`, `IHardwareProfileRepository`) are injected wherever a class needs read access to configuration or hardware data. No class constructs a repository internally; repositories are constructed once at `CompositionRoot` and shared (see §5.6 for lifecycle).

### 3.6 Factory Injection

Factories (`IFrameworkAdapterFactory`, `IExporterFactory`, `ICompressionFactory`, `IQuantizationFactory`, `IPluginFactory`) are injected into any class that needs to resolve a strategy/backend at runtime by key (framework name, hardware class, compression type). No class embeds `if/elif` chains that duplicate factory resolution logic — resolution logic lives in exactly one place per factory.

### 3.7 UML-Style Construction Example

```
┌─────────────────────────────┐
│  CompositionRoot             │
│  ─────────────────────────── │
│  + build_pipeline_orchestrator()
└───────────────┬───────────────┘
                │ constructs, in dependency order
                ▼
┌─────────────────────────────┐        implements        ┌───────────────────┐
│  StructuredLogger             │◄────────────────────────│  ILogger            │
└───────────────┬───────────────┘                          └───────────────────┘
                │ injected into
                ▼
┌─────────────────────────────┐        implements        ┌───────────────────────────────┐
│  HardwareProfileRepository    │◄────────────────────────│  IHardwareProfileRepository      │
└───────────────┬───────────────┘                          └───────────────────────────────┘
                │ injected into
                ▼
┌─────────────────────────────┐
│  QuantizationEngine (domain)  │  ← constructed with ILogger + IHardwareProfileRepository only,
└─────────────────────────────┘     never with the concrete StructuredLogger/HardwareProfileRepository types
```

---

## 4. Pipeline Stage Implementation Template

Every class in `domain/*` that participates in the pipeline extends the abstract `PipelineStage` (`domain/pipeline/pipeline_stage.py`). Every concrete stage implements the following lifecycle methods in this exact order of responsibility.

### 4.1 `initialize()`

- Validates that everything the stage needs from `PipelineContext` is present (upstream `StageResult`s, required config keys). Raises the appropriate `UAQEError` subclass immediately if a precondition is missing — never defers the check into `execute()`.
- Performs no side effects beyond validation and internal state setup (no file writes, no model mutation).
- Idempotent: calling `initialize()` twice on the same context must not corrupt stage state.

### 4.2 `execute()`

- Contains the stage's actual work. Reads only from `PipelineContext` and injected collaborators; never reaches into global state.
- Produces exactly one `StageResult` as its return value. Never returns `None`, even on a no-op stage (return a `StageResult` with `status=SKIPPED` and an explanatory note).
- Never calls another stage's `execute()` directly — inter-stage sequencing is `PipelineOrchestrator`'s responsibility only.
- All logging inside `execute()` brackets the work with an `info`-level "stage start" log immediately on entry and a "stage end" log immediately before return, per `07_Coding_Standards.md` §6.

### 4.3 `validate()`

- Runs *after* `execute()` produces a result, checking the result's internal consistency (e.g., `QuantizationEngine.validate()` confirms every layer in the IMR has an assigned precision before the stage is considered complete).
- `validate()` failures raise, they do not silently downgrade `StageResult.status`.
- `validate()` never re-does work already done in `execute()`; it inspects the produced artifact, it does not recompute it.

### 4.4 `cleanup()`

- Releases any per-run resources the stage acquired (temp buffers, file handles under `workdir/<run_id>/`). Always called by `PipelineOrchestrator`, including on the failure path, via a `finally` block at the orchestration level — individual stages never assume `cleanup()` runs only on success.
- `cleanup()` never raises; a `cleanup()`-time error is logged at `warning` level and swallowed, since it must never mask the original stage outcome.

### 4.5 `rollback()`

- Invoked only by `PipelineOrchestrator` when `ExecutionConfig.on_error == ABORT` (or `BEST_EFFORT` with an unrecoverable stage) and the stage has partially mutated shared state (e.g., partially written `workdir/<run_id>/artifacts/`).
- Reverses exactly what the stage itself wrote; a stage never rolls back another stage's output.
- Stages that are purely read/compute with no persisted side effects (e.g., `SensitivityAnalyzer`) implement `rollback()` as a documented no-op, not an omitted method — every stage has an explicit rollback story.

### 4.6 `StageResult` Creation

Every `execute()` implementation constructs its `StageResult` (`common/result_types.py`) with:

- `status` — one of the documented enum values (`SUCCESS`, `SKIPPED`, `FAILED`) — never inferred by the orchestrator from side channels.
- `stage_name` — matches the stage's registered name in `PipelineBuilder`, not a free-text label.
- `payload` — the stage's actual output object (typed per `03_API_Specification.md`), never a loosely-typed `dict` grab-bag.
- `warnings` — a list populated for every recoverable anomaly the stage encountered; an empty list, not `None`, when there are none.
- `duration_ms` — measured by the stage itself around its own `execute()` body, not estimated by the orchestrator.

### 4.7 Logging Within a Stage

Minimum required log calls per stage invocation: one `info` at start, one `info` at end (including `duration_ms`), one `warning` per `StageResult.warnings` entry, one `error` immediately before any raised exception leaves the stage. No `debug` logging is required but is encouraged for expensive internal branches.

### 4.8 Exception Handling Within a Stage

A stage never catches an exception only to suppress it. Every caught exception is either (a) converted to a `UAQEError` subclass and re-raised with `raise ... from err`, or (b) recorded as a `StageResult.warnings` entry when the stage's contract explicitly allows partial success for that condition (documented per-stage in `10_Module_Development_Guide.md`).

### 4.9 State Management

Stages are stateless between runs. Any field set in `initialize()` is scoped to the current `PipelineContext` and is reset (or the stage instance is discarded and rebuilt) before the next run — verified by the shared parametrized contract test described in `07_Coding_Standards.md` §10.

---

## 5. Repository Implementation Rules

### 5.1 `ConfigRepository`

- Loads each `config/*.json` / `settings.yaml` file exactly once per process lifetime unless an explicit reload is requested; parses and validates per `06_Config_Spec.md` §9 at load time, not at point of use.
- Exposes one typed `load_*_config()` method per subsystem (`load_quantization_config()`, `load_compression_config()`, etc.) — never a single generic `get(key: str)` that bypasses type validation.
- Applies the override precedence chain from `06_Config_Spec.md` §8 internally; callers never manually merge `config_overrides` themselves.

### 5.2 `HardwareProfileRepository`

- Loads `HardwareProfile` objects from `hardware_profiles/**/*.json` on demand (see Lazy Loading, §5.4), keyed by `profile_id`.
- Rejects any profile file with an unrecognized `schema_version` major component with `ConfigurationError`, per `05_Hardware_Profile_Spec.md` §7.3.
- Never mutates a `HardwareProfile` after construction — all `HardwareProfile` and `FpgaResourceProfile` instances are `@dataclass(frozen=True)`.

### 5.3 `FilesystemRepository`

- The sole component permitted to perform raw path construction under `workdir/`, `outputs/`, `reports/`, `logs/` (per `02_Folder_Structure.md` §19 ownership table). No other class calls `os.path` / `pathlib` directly against these roots.
- Enforces path containment: every path it returns is validated to resolve inside the configured root (`config.json`'s `workdir_path`/`outputs_path`/etc.), rejecting any input that would traverse outside it — this is the primary defense referenced in the Security Testing section of `12_Project_Test_Plan.md` §13.

### 5.4 Caching

- `HardwareProfileRepository` caches parsed `HardwareProfile` objects in-memory when `hardware.json`'s `profile_cache_enabled` is `true` (`06_Config_Spec.md` §3); cache key is `profile_id`, cache is never invalidated mid-process (profiles are read-only for the process lifetime).
- `ConfigRepository` caches parsed config file contents identically; a reload is only ever triggered by an explicit administrative action (e.g., CLI `--reload-config`), never implicitly by a run request.

### 5.5 Lazy Loading

- Hardware profiles are loaded on first request for a given `profile_id`, not eagerly at startup — startup time must not scale with the size of the hardware profile database.
- Calibration/evaluation datasets under `datasets/` are streamed by `CalibrationEngine`/`Evaluator`, never fully materialized in memory by a repository (see §12).

### 5.6 Read-Only Data

- `config/`, `hardware_profiles/`, `datasets/`, `plugins/` are read-only at runtime for every repository (`02_Folder_Structure.md` §19). No repository ever writes back to these folders. Any workflow that appears to require writing to `config/` (e.g., "save this run's settings as new defaults") writes to a new file under `outputs/<run_id>/` instead, never mutates the source.

### 5.7 Thread Safety

- All repositories are safe for concurrent read access from multiple stages running under `execution.parallelism` (`06_Config_Spec.md` §2). Caches use a read-mostly structure (e.g., populate-once-then-read dict) rather than locking on every read; a single mutex (or equivalent) guards first-population only.
- No repository exposes a mutable reference to its internal cache; every `get()` call returns either an immutable value object or a defensive copy.

### 5.8 Repository Lifecycle

- All repositories are constructed once at `CompositionRoot` and live for the process lifetime (or for the life of the hosting `SessionManager` in a long-running API server deployment) — never re-constructed per run, per stage, or per request.

---

## 6. Factory Pattern Rules

### 6.1 Framework Factory

Resolves an `IFrameworkAdapter` from the input file's extension (`.pth`/`.pt` → `TorchAdapter`, `.onnx` → `OnnxAdapter`, `.pb` → `TensorflowAdapter`, `.h5`/`.keras` → `KerasAdapter`, `.tflite` → `TfliteAdapter`, per `02_Folder_Structure.md` §5). Adding a new input format never modifies `ModelLoader`; it registers a new adapter with the factory.

### 6.2 Exporter Factory

Resolves an `IExporterBackend` from `(HardwareClass, profile_id)`. FPGA family resolves to one of `artix7_backend`/`zynq7000_backend`/`kintex_backend`/`cyclonev_backend`; Embedded family resolves per MCU backend; Raspberry Pi family resolves per board backend (`02_Folder_Structure.md` §5).

### 6.3 Compression Factory

Resolves one or more `ICompressionStrategy` implementations from `CompressionConfig.enabled_types`, in the order `CompressionAdvisor` recommends.

### 6.4 Quantization Factory

Resolves an `IQuantizationStrategy` from `QuantizationConfig.default_precision` (and any `per_layer_overrides`), filtered against `HardwareProfile.supported_precisions` per the Hardware Constraint Enforcement Contract.

### 6.5 Plugin Factory

Resolves strategies/backends registered by `PluginRegistry` at startup (see §8) using the same interface types as the built-in factories — a plugin-supplied `IQuantizationStrategy` is indistinguishable, from the factory's perspective, from a built-in one.

### 6.6 Factory Interaction Diagram

```
RunRequest ──► WorkflowController ──► PipelineBuilder
                                            │
                                            ▼
                                 PipelineOrchestrator
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       ▼
          FrameworkAdapterFactory   QuantizationFactory     ExporterFactory
                    │                       │                       │
          resolves IFrameworkAdapter  resolves IQuantizationStrategy resolves IExporterBackend
          (built-in or plugin)        (built-in or plugin)           (built-in or plugin)
                    │                       │                       │
                    └───────────► PluginRegistry (consulted first for overrides,
                                   falls through to built-in registrations)
```

Every factory consults `PluginRegistry` before its built-in registry, so a plugin may override a built-in strategy for the same key only if explicitly marked `allow_override=True` in its plugin metadata (§8.4); otherwise a key collision raises `PluginValidationError` at registration time, never at resolution time.

---

## 7. Strategy Pattern Rules

### 7.1 Quantization Strategy

`IQuantizationStrategy` implementations (one per `Precision` value or precision family) encapsulate the actual weight/activation transformation. `QuantizationEngine` never contains precision-specific branching logic itself — it delegates entirely to the resolved strategy.

### 7.2 Compression Strategy

`ICompressionStrategy` implementations (`PRUNING`, `WEIGHT_CLUSTERING`, `HUFFMAN`, `RLE`, per `06_Config_Spec.md` §4) each operate on the IMR independently and are composable — `CompressionEngine` applies the ordered list `CompressionAdvisor` returns, treating each strategy as a pure IMR-to-IMR transform.

### 7.3 Exporter Strategy

`IExporterBackend` implementations translate the final IMR into hardware-native output files. Each backend owns its target format exclusively (e.g., only `artix7_backend.py` writes `.mem`/`.hex`/`.bin` for Artix-7); no backend writes another hardware class's output format.

### 7.4 Benchmark Strategy

`Benchmarker` delegates to a real-hardware execution path or a `HardwareProfile`-derived analytical simulation path per `benchmark.json`'s `use_real_hardware_if_available`/`simulation_fallback_enabled` flags (`06_Config_Spec.md` §6) — both paths implement the same internal strategy interface so `Benchmarker`'s aggregation logic is identical regardless of which path ran.

### 7.5 Selection Process

Strategy selection always flows: `*Advisor` (domain-level decision, hardware- and accuracy-aware) → `*Factory` (resolves the interface implementation for the advisor's chosen key) → `*Engine` (applies it). An `Engine` never selects its own strategy by inspecting the model directly — that decision belongs to the `Advisor`.

### 7.6 Extension Process

To add a new quantization/compression/exporter/benchmark strategy:

1. Implement the appropriate interface from `uaqe.common.interfaces`.
2. Register it with the relevant factory (built-in: direct registration in the factory's static registry; external: via `PluginRegistry`, see §8).
3. Add the shared parametrized contract test (`07_Coding_Standards.md` §10) so the new implementation is automatically verified for interchangeability.
4. No existing `Advisor`, `Engine`, or `Factory` class requires modification — this is the concrete test of correct strategy-pattern application; if adding a strategy requires editing `QuantizationEngine`, the extension was done wrong.

---

## 8. Plugin Development Rules

### 8.1 Plugin Registration

Plugins under `plugins/{quantization_strategies,compression_strategies,exporter_backends}/` (`02_Folder_Structure.md` §17) are discovered and registered by `PluginRegistry` at `CompositionRoot` startup, before any `RunRequest` is accepted.

### 8.2 Plugin Discovery

`PluginRegistry` scans each `plugins/<category>/` subfolder for modules exposing a documented entry-point convention (a single class per module implementing the relevant `I*` interface, with a module-level `PLUGIN_METADATA` object). Discovery never executes arbitrary top-level code beyond import; side-effecting setup happens only inside the plugin's own `initialize()`-equivalent, invoked explicitly by the registry.

### 8.3 Plugin Validation

Every discovered plugin is validated before registration:

- Implements the declared interface completely (no missing abstract methods).
- `PLUGIN_METADATA` is present and well-formed (see §8.4).
- Passes the same shared contract test suite as built-in strategies (§7.6/§10), run in an isolated validation pass at startup; a plugin failing its contract test is logged at `error` and excluded from the registry, never silently partially registered.

### 8.4 Plugin Metadata

```
class PluginMetadata
  + plugin_id: str
  + display_name: str
  + version: str                # semver
  + category: PluginCategory     # QUANTIZATION | COMPRESSION | EXPORTER
  + interface_version: str       # matches the I* interface's own version tag
  + allow_override: bool         # default False
```

### 8.5 Plugin Versioning

`interface_version` is checked against the currently supported interface version range at registration time; a plugin built against an incompatible `interface_version` is rejected with a clear `PluginValidationError` message naming the mismatch, not a downstream `AttributeError` at resolution time.

### 8.6 Backward Compatibility

Once an `I*` interface is part of a tagged release, its existing abstract method signatures are never changed — only additive (new optional methods with default implementations, or a new interface version) — so that plugins built against an older `interface_version` continue to load unless they are explicitly incompatible per §8.5.

---

## 9. Exception Handling Rules

### 9.1 `UAQEError` Hierarchy

Every domain/application exception is a subclass of `UAQEError`, mirroring the pipeline stages it can originate from (e.g., `ModelLoadError`, `ValidationError`, `CompatibilityError`, `QuantizationError`, `CompressionError`, `OptimizationError`, `ExportError`, `BenchmarkError`, `ConfigurationError`, `PluginValidationError`). No new exception type is introduced without adding it to this hierarchy and documenting it alongside its originating stage.

### 9.2 Exception Conversion

Any exception raised by a third-party library (`torch`, `onnx`, `tensorflow`, `h5py`, etc.) is caught at the narrowest possible boundary — inside the specific `infrastructure/framework_adapters/*` or `infrastructure/exporter_backends/*` module — and re-raised as the matching `UAQEError` subclass with the original exception preserved via `raise ... from err`. No third-party exception type ever crosses into `uaqe.domain`.

### 9.3 Exception Propagation

Stages never catch a `UAQEError` raised by a collaborator just to re-wrap it in a different `UAQEError` subclass — propagate it unchanged unless the stage is genuinely translating a lower-level condition into a higher-level one it is responsible for reporting.

### 9.4 Logging on Exception

Every raised `UAQEError` is logged at `error` (or `critical` if pipeline-halting) at the point of raise, including `code`, `stage`, and `remediation_hint` if present — not re-logged again at each layer it passes through, to avoid duplicate log noise for the same failure.

### 9.5 Recovery

Recovery is governed exclusively by `PipelineOrchestrator` reading `ExecutionConfig.on_error`: `ABORT` halts the run immediately; `SKIP_STAGE` marks the stage `FAILED` in its `StageResult`, records the error, and proceeds to the next independent stage where the data-flow graph allows it; `BEST_EFFORT` proceeds past non-fatal stage failures and surfaces every accumulated error in the final `RunResult`.

### 9.6 Rollback

On `ABORT`, `PipelineOrchestrator` invokes `rollback()` (§4.5) on every stage that completed or partially completed in the current run, in reverse execution order, before propagating the terminal exception to the caller.

---

## 10. Logging Rules

### 10.1 Stage Start / Stage End

Every stage logs `info` on entry to `execute()` (fields: `run_id`, `stage_name`, `status="STARTED"`) and `info` on successful exit (fields: `run_id`, `stage_name`, `status`, `duration_ms`).

### 10.2 Warnings

Every `StageResult.warnings` entry is mirrored as a `warning`-level log line at the moment it is appended, not batched and logged only at stage end.

### 10.3 Errors

Every raised `UAQEError` is logged `error` (or `critical`) per §9.4, at the raise site.

### 10.4 Performance Logs

`Benchmarker` and `PipelineOrchestrator` emit `info`-level performance log entries (`duration_ms`, `peak_memory_bytes` where measurable) distinct from correctness logs, tagged `log_type="performance"` for downstream filtering.

### 10.5 Benchmark Logs

`Benchmarker` logs each individual trial at `debug` and the aggregate `BenchmarkResult` at `info`, always including `hardware_profile_id` and `trials`/`warmup_trials` counts actually used.

### 10.6 Run IDs

Every log line carries `run_id`, generated once by `SessionManager` at run start (`default_run_id_prefix` from `config.json`) and threaded through `PipelineContext` unchanged for the run's entire lifetime.

### 10.7 Correlation IDs

Where a single `run_id` spans multiple concurrent stage executions (`execution.parallelism`), each concurrent stage additionally carries a `correlation_id` unique to its own execution, so interleaved JSON-lines log output remains attributable per-stage-instance during log analysis.

### 10.8 Structured Logging

All log calls go through `ILogger` and emit JSON-lines (`06_Config_Spec.md` §2, `json_lines: true`) — never free-text string interpolation into the log message body; structured fields are passed as `**fields`, per `07_Coding_Standards.md` §6.

---

## 11. Thread Safety Rules

### 11.1 Immutable Objects

All value objects (`QuantizationConfig`, `CompressionConfig`, `HardwareProfile`, `FpgaResourceProfile`, etc.) are `@dataclass(frozen=True)` and therefore inherently thread-safe to share across concurrent stage executions.

### 11.2 Repositories

Safe for concurrent reads per §5.7; no repository is ever written to during a run (all repository-owned folders are read-only at runtime).

### 11.3 Logger

`StructuredLogger`'s underlying sink write is synchronized internally (single writer per `run_id` log file) so concurrent stages logging simultaneously never interleave partial JSON lines.

### 11.4 Configuration

`PipelineContext`'s resolved configuration for a given run is computed once at run start and treated as read-only for the remainder of that run; concurrent stages within the same run never mutate shared config state.

### 11.5 Pipeline

`PipelineOrchestrator` only parallelizes stages explicitly marked safe in `execution.parallelism` (`quantization`, `compression`, `benchmark` — per `06_Config_Spec.md` §2 and `01_Project_Architecture.md`'s Threading Strategy). No other stage pair is ever executed concurrently, because the data-flow graph in `04_Data_Flow.md` establishes sequential dependency for all other stage pairs.

### 11.6 Shared Objects

Any object shared across concurrently executing stages (a repository, the logger, a read-only `HardwareProfile`) must satisfy §11.1–§11.3; any object that does not (a mutable `PipelineContext` field, a per-stage scratch buffer) is never shared and is instead constructed per-stage-instance.

### 11.7 Synchronization

Where synchronization is unavoidable (repository first-population, log file writes), the minimal-scope primitive is used (e.g., a lock held only around the cache-population branch, not around every `get()` call) to avoid turning read-heavy paths into a contention bottleneck.

---

## 12. Memory Management Rules

### 12.1 Large Model Loading

Models exceeding `config.json`'s `large_model_threshold_mb` trigger `PipelineOrchestrator`'s large-model threading strategy (documented in `01_Project_Architecture.md`); `ModelLoader` itself always streams from disk via the resolved `IFrameworkAdapter` rather than reading the entire file into a Python `bytes` buffer before parsing, wherever the underlying framework library supports streaming/mapped loading.

### 12.2 Streaming

`CalibrationEngine` and `Evaluator` iterate calibration/evaluation datasets under `datasets/` as generators, never materializing the full dataset list in memory (§5.5).

### 12.3 Lazy Evaluation

IMR transformations (quantization, compression, optimization) are applied layer-by-layer where the underlying framework allows it, rather than requiring the full model resident in a secondary transformed copy simultaneously with the original.

### 12.4 Garbage Collection

Stages release references to large intermediate buffers (raw framework model objects, uncompressed calibration batches) as soon as the IMR conversion or transformation step consuming them completes — explicit `del` plus scope exit is preferred over relying on end-of-function garbage collection for multi-gigabyte objects.

### 12.5 Temporary Files

All temporary artifacts live under `workdir/<run_id>/` (never the system temp directory, never `outputs/`) so `PipelineOrchestrator`'s `cleanup()`/`rollback()` machinery has a single, predictable location to reclaim (§4.4, §4.5).

### 12.6 Memory Pools

Where a stage repeatedly allocates same-shaped buffers (e.g., `CalibrationEngine` batching), it reuses a pre-allocated buffer across batches rather than allocating fresh per batch, subject to profiling evidence that this materially helps — this is a recommended optimization, not a correctness requirement, and must never obscure per-batch isolation needed for correctness.

### 12.7 Cleanup

Every stage's `cleanup()` (§4.4) explicitly removes files it wrote under `workdir/<run_id>/` that are not needed by a downstream stage; artifacts genuinely needed downstream are documented as such in `04_Data_Flow.md` and are the responsibility of `PipelineOrchestrator`'s final run cleanup, not any individual stage.

---

## 13. Performance Rules

### 13.1 Caching

Repository-level caching per §5.4 is the primary caching mechanism; ad hoc caching inside domain classes (e.g., memoizing an `Advisor` decision) is permitted only within the scope of a single run and must be discarded with the stage instance.

### 13.2 Avoid Recomputation

`SensitivityAnalyzer`'s output (`SensitivityReport`) is computed once per run and consumed by both `QuantizationEngine` (per-layer overrides) and `CompressionAdvisor` (accuracy-delta guardrail) rather than being recomputed independently by each consumer.

### 13.3 Parallel Execution

Only the three stages enumerated in `execution.parallelism` (`06_Config_Spec.md` §2) are eligible for concurrent execution, and only when the data-flow graph in `04_Data_Flow.md` shows no dependency edge between the specific stage instances being parallelized for a given run.

### 13.4 Generators

Any per-layer or per-sample iteration (IMR layer walks, calibration batches, benchmark trials) is implemented as a generator/iterator rather than a fully materialized list, wherever the consumer only needs to process one item at a time.

### 13.5 Streaming

Export writers (`IExporterBackend` implementations) stream output bytes to disk incrementally for large artifacts rather than building the complete output buffer in memory before a single `write()` call, where the target format supports incremental writing (e.g., `.hex`/`.mem` line-oriented formats).

### 13.6 Batch Processing

`CalibrationEngine` and `Benchmarker` process data in the configured batch size (`calibration_batch_size`, `benchmark.json` trial batching where applicable) rather than per-sample, to amortize framework call overhead — batch size is always sourced from config, never hard-coded in the engine.

---

## 14. Unit Test Rules

### 14.1 Arrange / Act / Assert

Every unit test follows strict Arrange–Act–Assert structure with the three phases visually separated (blank line or comment); a test that interleaves setup and assertions is rejected in review as harder to diagnose on failure.

### 14.2 Fixtures

Shared setup (a minimal `HardwareProfile`, a tiny fixture model) lives in `tests/fixtures/` (`02_Folder_Structure.md` §14) as reusable `pytest` fixtures, never duplicated inline across multiple test files.

### 14.3 Mocking

Domain/application unit tests mock at the interface boundary (`ILogger`, `IHardwareProfileRepository`, `IFrameworkAdapter`) using simple hand-written fakes or `unittest.mock`, never instantiate concrete infrastructure classes — this is what keeps `uaqe.domain` unit tests fast and independent of real files/frameworks.

### 14.4 Coverage

Minimum coverage thresholds from `07_Coding_Standards.md` §10 (85% `domain`/`application`, 70% `infrastructure`) are enforced per-PR in CI, not just tracked as a trailing metric.

### 14.5 Naming

Test function names follow `test_<method_under_test>__<condition>__<expected_outcome>`, e.g. `test_execute__unsupported_precision__raises_quantization_error`.

---

## 15. Integration Test Rules

### 15.1 Pipeline Tests

`tests/integration/pipeline_end_to_end/` runs the full pipeline against `tests/fixtures/sample_models/`, covering at minimum one model per supported input format (`.pth`, `.pt`, `.onnx`, `.pb`, `.h5`, `.keras`, `.tflite`) crossed with one hardware profile per `HardwareClass` (`07_Coding_Standards.md` §10).

### 15.2 Hardware Tests

Hardware-class-specific integration tests verify the Hardware Constraint Enforcement Contract end-to-end (`05_Hardware_Profile_Spec.md` §6) — e.g., an intentionally oversized fixture model against `artix7`'s `max_model_size_bytes` must produce `ExportError`, not a truncated `.mem` file.

### 15.3 Export Tests

`tests/integration/exporter_backends/` runs every `IExporterBackend` against the shared contract test suite (§7.6) plus a backend-specific golden-output comparison against `tests/fixtures/golden_outputs/` (mirrors `12_Project_Test_Plan.md` §17).

### 15.4 Benchmark Tests

Benchmark integration tests exercise both the real-hardware and simulation-fallback code paths (mocking the "device attached" detection), verifying `Benchmarker` produces a structurally valid `BenchmarkResult` from either path.

### 15.5 Regression Tests

Every fixed defect gets a permanent regression test named after the defect's ticket/issue ID, added in the same PR as the fix, placed alongside the relevant unit or integration test suite — never deleted or "temporarily skipped" without a linked follow-up ticket.

---

## 16. Documentation Rules

### 16.1 Google Docstrings

Every public class and method carries a Google-style docstring with `Args:`, `Returns:`, `Raises:` sections wherever applicable, per `07_Coding_Standards.md` §9.

### 16.2 Module Docstrings

Every file opens with a one-sentence module docstring stating its single responsibility, matching its `02_Folder_Structure.md` entry.

### 16.3 Class Docstrings

Every class docstring states its role in the architecture (which interface it implements, which layer it lives in) in addition to its behavior.

### 16.4 Method Docstrings

Every public method docstring explains *why*, not just restates the signature (`07_Coding_Standards.md` §9) — the type hints already say *what*.

### 16.5 Examples

Complex methods (advisor decision logic, factory resolution) include a short illustrative example in the docstring `Example:` section showing typical input/output shape, without embedding a runnable code sample that could drift out of sync (illustrative pseudocode/comments are preferred over live doctest blocks for this project).

### 16.6 Documentation Quality

A docstring that only repeats the method name in prose ("Executes the stage.") is treated as a missing docstring in review.

---

## 17. Code Review Checklist

Every reviewer verifies, in this order, before considering style:

**Architecture**
- [ ] No folder, class, method, or config field renamed or relocated
- [ ] Dependency direction respected (`domain`/`common` never import `infrastructure`/`interface`)
- [ ] No new top-level dependency without `pyproject.toml` update + justification

**API**
- [ ] Signature matches `03_API_Specification.md` exactly (names, types, order)
- [ ] Return type matches documented type, including `Optional`/`List`/`Dict` wrapping

**Imports**
- [ ] Absolute imports only, correctly ordered (`isort`/`black` clean)
- [ ] No wildcard imports, no new circular import introduced

**Performance**
- [ ] No unnecessary full-materialization of large iterables (§13.4)
- [ ] No recomputation of already-available upstream results (§13.2)

**Logging**
- [ ] Stage start/end logged; warnings mirrored to log lines; errors logged at raise site

**Memory**
- [ ] Temp files scoped to `workdir/<run_id>/`; large buffers released promptly

**Exceptions**
- [ ] All raised exceptions are `UAQEError` subclasses with `code`/`message`/`stage` set
- [ ] No bare `except:`/`except Exception:` swallow

**Testing**
- [ ] Matching unit test file exists (1:1 mapping)
- [ ] Coverage threshold met for the touched layer
- [ ] New defect fixes include a regression test

**Documentation**
- [ ] Module/class/method docstrings present and meaningful
- [ ] Any new config field or hardware field documented in the relevant spec file, not left implicit

---

## 18. Common Mistakes

| Mistake | Why It Happens | How To Avoid It |
|---|---|---|
| Instantiating a concrete infrastructure class inside a domain class | Feels faster than wiring through `CompositionRoot` during a quick fix | Always inject the interface; if construction is inconvenient, that is a signal the dependency graph needs `CompositionRoot` attention, not a shortcut |
| Catching a third-party exception outside the adapter boundary | Third-party error surfaces deep in a call stack that spans layers | Wrap the narrowest possible third-party call site, in the adapter module only |
| Returning `None` from a stage on a no-op path | "Nothing happened" feels like it needs no result object | Always return a `StageResult` with `status=SKIPPED` and a warning explaining why |
| Hard-coding a hardware limit instead of reading `HardwareProfile` | The number is "obviously" fixed for a known board during initial implementation | Every limit used in an exporter/optimizer must be read from the injected `HardwareProfile`, even if it happens to match a known constant today |
| Adding a new `if/elif` branch to an `Engine` for a new strategy | Feels like the smallest diff for a one-off strategy addition | Implement the strategy interface and register it with the factory instead (§7.6) — the `Engine` must not grow branches per strategy |
| Logging via `print()` or stdlib `logging` in domain code during debugging, and forgetting to remove it | Convenient during local iteration | Use `ILogger` from the start; CI's linter step should catch stray `print`/`logging` imports in `uaqe.domain` |
| Mutating a `HardwareProfile` or config value object in place | Feels convenient to "patch" one field for a special case | Value objects are frozen; construct a new instance (or use `config_overrides` at the `RunRequest` layer) instead |
| Skipping `rollback()` implementation for a "read-only" stage | Assumed unnecessary since nothing is written | Implement an explicit no-op `rollback()` so the contract is visibly satisfied, not silently absent |

---

## 19. Best Practices

1. Favor composition over inheritance beyond the single `PipelineStage` base class — strategies, factories, and repositories are composed into a stage, not subclassed into stage variants.
2. Keep `Advisor` classes purely decision-making (no I/O, no mutation) so they are trivially unit-testable in isolation from `Engine` classes.
3. Treat every `HardwareProfile` field as the single source of truth for hardware limits — never re-derive or cache a hardware number outside the repository layer.
4. Write the contract test for a new interface implementation before or alongside the implementation itself, not after, so interchangeability is verified from the first commit.
5. Prefer explicit `StageResult.warnings` entries over silent best-effort behavior — a downstream report should never be the first place a data quality issue surfaces.
6. Keep `PipelineContext` narrow: only the data genuinely needed by downstream stages belongs in it; scratch state stays local to the stage that created it.
7. When in doubt about whether something is "architectural" (requiring an RFC) or "implementation detail" (this document's domain), default to treating it as architectural and ask — reversing an RFC decision is cheaper than reversing a merged architecture drift.
8. Optimize only after profiling; §12/§13's rules describe the *shape* of correct memory/performance behavior, not a license to prematurely hand-optimize a stage that is not yet a measured bottleneck.

---

## 20. Final Implementation Checklist

Before committing any code, every developer verifies:

- [ ] Read the relevant sections of `01`–`10` for the module being touched
- [ ] No locked artifact (folder, signature, config schema, class/method name) modified
- [ ] Constructor injection used throughout; no internal construction of concrete infrastructure in domain/application code
- [ ] If touching a `PipelineStage`, all five lifecycle methods (`initialize`, `execute`, `validate`, `cleanup`, `rollback`) are implemented per §4
- [ ] All new exceptions are `UAQEError` subclasses with `code`/`message`/`stage` populated
- [ ] All logging goes through `ILogger` with structured fields, including `run_id`
- [ ] New strategies/backends/plugins implement the documented interface and are registered via the correct factory or `PluginRegistry`, without modifying an existing `Engine`
- [ ] Matching unit test file created/updated; coverage threshold satisfied
- [ ] Integration/contract tests updated if a new interface implementation was added
- [ ] Docstrings present at module, class, and method level, matching the Documentation Rules in §16
- [ ] Code Review Checklist (§17) self-verified before requesting review
- [ ] Commit message follows Conventional Commits format per `07_Coding_Standards.md` §13

---

**End of `11_Implementation_Rules.md`.**
