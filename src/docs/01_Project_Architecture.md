# 01_Project_Architecture.md
## Universal AI Quantization Engine — System Architecture

**Document Status:** DRAFT — pending Architecture Lock (see `09_Architecture_Lock.md`)
**Document Owner:** Chief Software Architect
**Applies To:** Entire codebase, all contributors, all future modules

---

## 1. Executive Summary

The **Universal AI Quantization Engine (UAQE)** is a production-grade, hardware-aware Edge AI deployment platform. It ingests trained neural network models in common exchange formats (`.pth`, `.pt`, `.onnx`, `.pb`, `.h5`, `.keras`, `.tflite`) and produces deployment-ready artifacts for three fundamentally different hardware classes: **FPGA fabrics**, **embedded microcontrollers**, and **Raspberry Pi single-board computers**.

UAQE is not a thin wrapper around PyTorch/TensorFlow/ONNX Runtime conversion utilities. It is an **orchestration and decision-making system** that:

1. Understands the structural and numerical properties of an incoming model (`Analysis`).
2. Understands the physical and runtime constraints of the target hardware (`Hardware Profiles`).
3. Makes automated, explainable decisions about quantization strategy, compression strategy, and memory layout (`Advisor`, `Quantization`, `Compression`, `Optimization`).
4. Produces hardware-native deployment artifacts (`Exporter`).
5. Proves the resulting artifact is correct and performant (`Evaluation`, `Benchmark`).
6. Documents every decision it made and why (`Reports`).

The system is designed so that **the decision logic (what to do)** is fully decoupled from **the mechanics (how to do it)**, and both are decoupled from **hardware knowledge (what the target can support)**. This separation is the single most important architectural property of UAQE and is enforced throughout every layer below.

This document is the authoritative description of that architecture. It does not contain implementation code. It defines the shape of the system that all subsequent documents (`02` through `10`) must conform to, and that `09_Architecture_Lock.md` will freeze permanently.

---

## 2. Overall System Architecture

UAQE is organized as a **Clean / Layered Architecture** with a **Core Domain**, surrounded by **Application Services**, surrounded by **Infrastructure Adapters**, surrounded by **Interface/Entry Points**. Dependencies point strictly inward. Outer layers know about inner layers; inner layers never know about outer layers.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INTERFACE LAYER                              │
│   CLI Entry Point   │   REST API Entry Point   │   Batch Runner      │
└───────────────────────────────┬────────────────────────────────────--┘
                                 │  (calls)
┌───────────────────────────────▼──────────────────────────────────────┐
│                       APPLICATION LAYER                              │
│   PipelineOrchestrator  │  WorkflowController  │  SessionManager     │
└───────────────────────────────┬──────────────────────────────────────┘
                                 │  (uses)
┌───────────────────────────────▼──────────────────────────────────────┐
│                          DOMAIN LAYER (CORE)                          │
│  ModelLoader │ ModelAnalyzer │ HardwareManager │ QuantizationEngine   │
│  CompressionEngine │ OptimizationEngine │ Evaluator │ Benchmarker     │
│  OptimizationAdvisor │ ReportGenerator                                │
└───────────────────────────────┬──────────────────────────────────────┘
                                 │  (depends on abstractions)
┌───────────────────────────────▼──────────────────────────────────────┐
│                      INFRASTRUCTURE LAYER                            │
│  FrameworkAdapters(Torch/ONNX/TF/Keras/TFLite) │ ExporterBackends     │
│  ConfigRepository │ HardwareProfileRepository │ FileSystemRepository │
│  Logger │ MetricsCollector                                           │
└──────────────────────────────────────────────────────────────────────┘
```

**Rule:** Domain Layer classes depend only on interfaces (abstract base classes) defined within the Domain Layer itself. Infrastructure Layer classes implement those interfaces. This is the Dependency Inversion Principle applied at the architecture level and is non-negotiable (see `09_Architecture_Lock.md`).

---

## 3. Layered Architecture (Detailed)

### 3.1 Interface Layer
Responsible only for translating an external trigger (CLI invocation, HTTP request, batch job file) into a call against the Application Layer, and translating the Application Layer's result back into an external response (stdout, HTTP JSON, exit code). Contains **no business logic**.

### 3.2 Application Layer
Responsible for **orchestration**: sequencing domain operations in the correct order, managing session/run state, handling cross-cutting concerns (progress reporting, cancellation, checkpointing). Contains **no hardware knowledge and no ML logic** — it only knows the *order* in which Domain services must be invoked.

### 3.3 Domain Layer (Core)
Contains all ML/hardware decision-making logic: model analysis, quantization strategy selection, compression strategy selection, multi-objective optimization, evaluation, benchmarking, advisory scoring, and report content generation. Domain classes are framework-agnostic — they operate on UAQE's own **Internal Model Representation (IMR)**, never directly on a `torch.nn.Module` or `tf.keras.Model`.

### 3.4 Infrastructure Layer
Contains all "dirty" integration code: reading `.pth`/`.onnx`/`.h5` files, calling into PyTorch/TensorFlow/ONNX libraries to parse them into IMR, writing `.mem`/`.hex`/`.bin`/`.h`/`.tflite` files, reading/writing JSON/YAML config, logging, filesystem I/O. Infrastructure classes implement Domain-defined interfaces (ports) — this is the **Ports and Adapters (Hexagonal)** pattern nested inside the Clean Architecture.

---

## 4. Module Interaction Diagram

```
                     ┌────────────────────────┐
                     │  PipelineOrchestrator   │
                     └────────────┬────────────┘
     ┌───────────────┬────────────┼────────────┬───────────────┬────────────┐
     ▼               ▼            ▼            ▼               ▼            ▼
┌─────────┐   ┌─────────────┐ ┌──────────┐ ┌───────────┐ ┌───────────┐ ┌──────────┐
│ Model   │   │ Hardware    │ │ Quant    │ │ Compress  │ │ Optimize  │ │ Exporter │
│ Loader/ │──▶│ Manager     │▶│ Engine   │▶│ Engine    │▶│ Engine    │▶│          │
│ Analyzer│   └─────────────┘ └──────────┘ └───────────┘ └───────────┘ └────┬─────┘
└─────────┘                                                                 │
                                                                             ▼
                                                          ┌───────────────────────────┐
                                                          │ Evaluator │ Benchmarker   │
                                                          └─────────────┬─────────────┘
                                                                        ▼
                                                          ┌───────────────────────────┐
                                                          │  OptimizationAdvisor      │
                                                          └─────────────┬─────────────┘
                                                                        ▼
                                                          ┌───────────────────────────┐
                                                          │   ReportGenerator         │
                                                          └───────────────────────────┘
```

Every arrow represents a call made **through an interface**, mediated by the `PipelineOrchestrator`. No domain module calls another domain module directly by concrete class reference; all cross-module calls are injected (see §12, Dependency Injection Strategy).

---

## 5. Internal Workflow

This mirrors the workflow specified in the project brief, mapped explicitly onto owning modules:

| Step | Workflow Stage | Owning Module |
|---|---|---|
| 1 | Upload AI Model | `ModelIngestionService` (Application) |
| 2 | Framework Detection | `FrameworkDetector` (Infrastructure) |
| 3 | Model Validation | `ModelValidator` (Domain) |
| 4 | Model Analysis | `ModelAnalyzer` (Domain) |
| 5 | Unsupported Layer Detection | `LayerCompatibilityChecker` (Domain) |
| 6 | User Selects Target Hardware | `HardwareSelectionService` (Application) |
| 7 | Hardware Compatibility Check | `HardwareManager` (Domain) |
| 8 | Automatic Quantization Recommendation | `QuantizationAdvisor` (Domain) |
| 9 | Calibration | `CalibrationEngine` (Domain) |
| 10 | Layer Sensitivity Analysis | `SensitivityAnalyzer` (Domain) |
| 11 | Layer-wise Quantization | `QuantizationEngine` (Domain) |
| 12 | Automatic Compression Selection | `CompressionAdvisor` (Domain) |
| 13 | Compression | `CompressionEngine` (Domain) |
| 14 | Multi-objective Optimization | `OptimizationEngine` (Domain) |
| 15 | Memory Optimization | `MemoryOptimizer` (Domain) |
| 16 | Target-specific Export | `Exporter` (Domain) + `ExporterBackend` (Infrastructure) |
| 17 | Evaluation | `Evaluator` (Domain) |
| 18 | Benchmark | `Benchmarker` (Domain) |
| 19 | Deployment Readiness Score | `DeploymentReadinessScorer` (Domain) |
| 20 | Optimization Advisor | `OptimizationAdvisor` (Domain) |
| 21 | Generate Reports | `ReportGenerator` (Domain) |

Each stage produces a strongly-typed **Stage Result** object that is stored in the `PipelineContext` (a mutable, append-only run state object owned by the `PipelineOrchestrator`) and passed forward. No stage mutates the output of a previous stage in place — each stage produces a new immutable result object. This guarantees full run traceability and enables checkpoint/resume.

---

## 6. Component Responsibilities

| Component | Responsibility | Must NOT Do |
|---|---|---|
| `ModelLoader` | Load raw file, delegate to correct `FrameworkAdapter`, produce IMR | Perform analysis or validation logic |
| `ModelAnalyzer` | Compute layer graph, parameter counts, FLOPs, memory footprint, op types | Modify the model |
| `LayerCompatibilityChecker` | Flag layers unsupported by chosen hardware/runtime | Reject the whole pipeline silently |
| `HardwareManager` | Load hardware profile, check constraints, expose capability queries | Perform quantization math |
| `QuantizationEngine` | Apply INT8/INT4/FP16/mixed-precision transforms to IMR | Decide which layers to skip (that is `QuantizationAdvisor`) |
| `CalibrationEngine` | Run calibration dataset through model, collect activation statistics | Apply quantization itself |
| `SensitivityAnalyzer` | Measure per-layer accuracy impact of candidate precisions | Choose final precision (that is `QuantizationAdvisor`) |
| `CompressionEngine` | Apply pruning, weight clustering, Huffman/RLE encoding | Decide compression ratio targets |
| `OptimizationEngine` | Run multi-objective search (accuracy vs. latency vs. memory) | Perform export |
| `MemoryOptimizer` | Determine tensor layout, buffer reuse, arena sizing | Perform quantization |
| `Exporter` | Select correct `ExporterBackend` per hardware target | Contain hardware-specific binary logic (delegated to backend) |
| `Evaluator` | Compute accuracy/error metrics pre vs. post optimization | Run on-device benchmarking |
| `Benchmarker` | Measure latency/throughput/memory on target or simulator | Compute accuracy metrics |
| `OptimizationAdvisor` | Aggregate all prior results into human-readable recommendations | Re-run any prior stage |
| `ReportGenerator` | Render all five report types from `PipelineContext` | Compute any new metric |

---

## 7. Package Dependency Diagram

```
uaqe.interface  ──depends on──▶  uaqe.application
uaqe.application ──depends on──▶ uaqe.domain
uaqe.domain      ──depends on──▶ uaqe.domain.interfaces   (abstract only)
uaqe.infrastructure ──implements──▶ uaqe.domain.interfaces
uaqe.infrastructure ──depends on──▶ uaqe.common
uaqe.domain         ──depends on──▶ uaqe.common
uaqe.application    ──depends on──▶ uaqe.common
uaqe.interface       ──depends on──▶ uaqe.common

uaqe.common  ──depends on nothing inside uaqe──
```

**Prohibited edges (enforced by code review + lint rule, see `07_Coding_Standards.md`):**
- `uaqe.domain` → `uaqe.infrastructure` (would invert dependency direction)
- `uaqe.domain` → `uaqe.interface`
- `uaqe.infrastructure` → `uaqe.application`
- Any circular import between sibling packages within `uaqe.domain.*`

---

## 8. High-Level Sequence Diagram

```
User        Interface        Orchestrator     ModelLoader   Analyzer   HWManager   QuantEngine   Exporter   ReportGen
 │               │                 │               │            │           │            │           │           │
 │ run(model,hw) │                 │               │            │           │            │           │           │
 │──────────────▶│                 │               │            │           │            │           │           │
 │               │  execute(req)   │               │            │           │            │           │           │
 │               │────────────────▶│               │            │           │            │           │           │
 │               │                 │  load(path)   │            │           │            │           │           │
 │               │                 │──────────────▶│            │           │            │           │           │
 │               │                 │◀──IMR─────────│            │           │            │           │           │
 │               │                 │  analyze(IMR) │            │           │            │           │           │
 │               │                 │───────────────────────────▶│           │            │           │           │
 │               │                 │◀──AnalysisResult───────────│           │            │           │           │
 │               │                 │  checkCompat(HW, Analysis) │           │            │           │           │
 │               │                 │─────────────────────────────────────-─▶│            │           │           │
 │               │                 │◀──CompatibilityReport───────────────---│            │           │           │
 │               │                 │  quantize(IMR, plan)       │           │            │           │           │
 │               │                 │────────────────────────────────────────────────────▶│           │           │
 │               │                 │◀──QuantizedIMR──────────────────────────────────────│           │           │
 │               │                 │  export(QuantizedIMR, target)          │            │           │           │
 │               │                 │────────────────────────────────────────────────────────────────▶│           │
 │               │                 │◀──DeploymentArtifact───────────────────────────────────────────--│           │
 │               │                 │  generateReports(context)              │            │           │           │
 │               │                 │─────────────────────────────────────────────────────────────────────────────▶│
 │               │                 │◀──ReportBundle─────────────────────────────────────────────────────────────--│
 │               │◀──RunResult─────│               │            │           │            │           │           │
 │◀──output──────│                 │               │            │           │            │           │           │
```

(Compression, optimization, evaluation, and benchmark stages omitted from this diagram for readability — full detail in `04_Data_Flow.md`.)

---

## 9. Internal Processing Pipeline

The pipeline is implemented as a **Chain of Stage Handlers**, each conforming to a single `PipelineStage` interface (`execute(context) -> StageResult`). The `PipelineOrchestrator` holds an ordered list of stages built by a `PipelineBuilder` (Builder Pattern) at run start, based on `config.json` and the selected hardware class. This allows:

- Skipping stages (e.g., no `CalibrationEngine` stage if static quantization is chosen).
- Inserting hardware-class-specific stages (e.g., FPGA runs get an extra `BitstreamPackaging` stage the others do not).
- Re-running a stage in isolation for debugging without re-running the whole pipeline (each stage reads only from `PipelineContext`, never from a prior stage's live object).

```
PipelineBuilder.build(hardwareClass, config)
        │
        ▼
[Stage1] → [Stage2] → [Stage3] → ... → [StageN]
   │           │           │               │
   ▼           ▼           ▼               ▼
PipelineContext.append(result) at every step
```

---

## 10. Future Expansion Strategy

UAQE anticipates the following expansion axes and reserves architectural seams for each:

1. **New input frameworks** (e.g., JAX, PaddlePaddle) → implement a new `FrameworkAdapter`; zero changes to Domain Layer.
2. **New hardware targets** (e.g., NPUs, RISC-V AI accelerators) → add a new `HardwareProfile` entry + a new `ExporterBackend`; zero changes to `QuantizationEngine`/`CompressionEngine`.
3. **New quantization algorithms** (e.g., GPTQ-style, AWQ-style) → implement a new `QuantizationStrategy` plugin (see §11).
4. **New compression algorithms** → implement a new `CompressionStrategy` plugin.
5. **New report types** → implement a new `ReportRenderer` plugin registered with `ReportGenerator`.
6. **Distributed/cloud execution** → `PipelineOrchestrator` is designed stage-serializable so stages can later be dispatched to remote workers without redesign (out of scope for v1, seam only).

This is achieved by strict adherence to the **Open/Closed Principle**: every one of the above is an *addition* of a new class implementing an existing interface, never a *modification* of existing Domain code.

---

## 11. Plugin Architecture

UAQE defines four plugin extension points, all discovered via a `PluginRegistry` (Registry + Factory Pattern) at startup, driven by `config.json` → `plugins` section:

| Extension Point | Interface | Registered By |
|---|---|---|
| Framework Adapter | `IFrameworkAdapter` | `FrameworkAdapterRegistry` |
| Quantization Strategy | `IQuantizationStrategy` | `QuantizationStrategyRegistry` |
| Compression Strategy | `ICompressionStrategy` | `CompressionStrategyRegistry` |
| Exporter Backend | `IExporterBackend` | `ExporterBackendRegistry` |
| Report Renderer | `IReportRenderer` | `ReportRendererRegistry` |

Plugins are resolved via **Strategy Pattern** at the point of use (e.g., `QuantizationEngine` receives an `IQuantizationStrategy` instance chosen by `QuantizationAdvisor`, not a hardcoded class). Built-in strategies ship in `uaqe.infrastructure.plugins.builtin.*`; third-party/custom strategies may be dropped into a configured plugin directory and are discovered by entry-point scanning at process start. Exact contracts are frozen in `03_API_Specification.md`.

---

## 12. Error Handling Strategy

- **Exception hierarchy** rooted at `UAQEError`, with domain-specific subclasses: `ModelLoadError`, `ModelValidationError`, `UnsupportedLayerError`, `HardwareIncompatibilityError`, `QuantizationError`, `CompressionError`, `OptimizationError`, `ExportError`, `EvaluationError`, `BenchmarkError`, `ConfigurationError`, `PluginLoadError`.
- **No bare `except:`** anywhere in the codebase (enforced lint rule).
- **Fail-fast at boundaries:** Infrastructure adapters validate and convert third-party exceptions (e.g., a PyTorch `RuntimeError`) into the appropriate `UAQEError` subclass before it crosses into the Domain Layer. Domain code never catches a raw third-party exception type.
- **Stage-level containment:** `PipelineOrchestrator` catches `UAQEError` at each stage boundary, records it into `PipelineContext.errors`, and — per `config.json` → `pipeline.on_error` (`abort` | `skip_stage` | `best_effort`) — either halts, skips to the next independent stage, or continues with a degraded result flagged in the final `SummaryReport`.
- **User-facing errors** are always structured objects (`code`, `message`, `remediation_hint`, `stage`), never raw stack traces, at the Interface Layer. Full stack traces are always available in logs.

---

## 13. Logging Strategy

- Single injected `ILogger` interface used everywhere; concrete implementation is a structured JSON-lines logger (`StructuredLogger`) writing to both console and `logs/<run_id>/pipeline.log`.
- Five levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
- Every log line carries: `timestamp`, `run_id`, `stage`, `module`, `level`, `message`, and optional `metrics` dict.
- Domain modules log through their injected `ILogger`; they never `print()`.
- Log verbosity is configured per-module via `settings.yaml` → `logging.levels`, allowing e.g. `QuantizationEngine: DEBUG` while everything else stays `INFO`.
- Exact schema frozen in `06_Config_Spec.md`.

---

## 14. Configuration Strategy

- **Configuration-Driven Design**: no behavioral constant is hardcoded; every threshold, default, and toggle lives in one of the seven config files enumerated in `06_Config_Spec.md`.
- Configuration is loaded once per run by `ConfigRepository` (Infrastructure), validated against a JSON Schema, and exposed to the Domain Layer as immutable, strongly-typed **Config Value Objects** (never raw dicts) — e.g., `QuantizationConfig`, `HardwareConfig`.
- Layered override precedence: **built-in defaults → `config.json` → `settings.yaml` → environment variables → CLI flags**, highest wins.
- Config Value Objects are constructed once by `ConfigFactory` and injected into every Domain object that needs them — Domain objects never read config files directly.

---

## 15. Dependency Injection Strategy

- UAQE uses **constructor injection** exclusively; no service locators, no global singletons (except the `PluginRegistry`, which is intentionally global by design, and the `ILogger`, which is provided via a lightweight injected context object, not a global).
- A single `CompositionRoot` (in `uaqe.interface`) is the only place in the codebase where concrete Infrastructure classes are instantiated and wired into Domain interfaces. No Domain or Application class ever performs `import` of a concrete Infrastructure class.
- All injectable dependencies are declared as constructor parameters typed by interface (ABC), never by concrete class, enforced by `07_Coding_Standards.md` type-hint rules.
- This enables full unit-testability of every Domain class via mock/stub implementations of its interfaces, and is a precondition for the Testing Strategy in `07_Coding_Standards.md`.

---

## 16. Threading Strategy

- The default execution mode is **single-threaded, single-process**, sequential-stage pipeline — this is the guaranteed-correct baseline for all hardware targets, including resource-constrained CI runners.
- **Parallelism is opt-in and isolated to specific, independently-verified hot paths**, controlled by `config.json` → `execution.parallelism`:
  - `CalibrationEngine` may run calibration batches across a `ThreadPoolExecutor` (I/O + numeric library releases GIL during heavy tensor ops).
  - `Benchmarker` may run repeated timing trials across a `ProcessPoolExecutor` when benchmarking multiple hardware targets in one invocation, since these are fully independent runs.
  - `OptimizationEngine`'s multi-objective search may evaluate candidate configurations in parallel (embarrassingly parallel search space).
- The `PipelineOrchestrator` itself is never parallelized across stages — stage N+1 always strictly depends on stage N's committed `PipelineContext` state, by design (§9), so parallel stage execution is architecturally disallowed, not just unimplemented.
- All parallel workers are pure functions over immutable inputs; no shared mutable state is accessed from worker threads/processes. Results are merged back into `PipelineContext` only by the orchestrating thread.

---

## 17. Memory Management Strategy

- Large tensors (model weights/activations) are never duplicated across stages without explicit justification logged via `ILogger.debug`; the IMR uses reference-counted tensor buffers so `QuantizationEngine` and `CompressionEngine` can transform in a copy-on-write manner.
- `MemoryOptimizer` is a first-class Domain module responsible for target-specific memory planning: static buffer arena sizing for microcontrollers (no heap allocation at inference time), tensor reuse graphs for Raspberry Pi/TFLite export, and on-chip BRAM/URAM allocation planning for FPGA export.
- The pipeline enforces a **peak memory budget check** before each heavy stage (quantization, compression, optimization) using the target `HardwareProfile.ram_bytes` / `HardwareProfile.tensor_memory_bytes` ceiling; exceeding it raises `HardwareIncompatibilityError` rather than allowing an OOM crash mid-stage.
- Intermediate stage artifacts (IMR snapshots) are streamed to disk (`workdir/<run_id>/artifacts/`) rather than held entirely in process memory when a model exceeds a configurable size threshold (`config.json` → `execution.large_model_threshold_mb`).

---

## 18. Scalability Strategy

- **Vertical scalability** (bigger models, bigger machines): governed by the streaming-artifact and large-model-threshold behavior in §17.
- **Horizontal scalability** (many models / many hardware targets per batch job): the `BatchRunner` (Interface Layer) fans out independent `PipelineOrchestrator` runs, each with its own `run_id` and `PipelineContext`; runs share only the read-only `HardwareProfileRepository` and `ConfigRepository` caches.
- **Stateless Domain services**: every Domain class is stateless between invocations (all state lives in `PipelineContext` or Config Value Objects), which is what makes horizontal fan-out and the future distributed-execution seam (§10.6) possible without redesign.
- **Report/Benchmark scaling**: `Benchmarker` and `ReportGenerator` are designed to consume N hardware-target results per model in a single pass, avoiding O(N²) recomputation when comparing multiple targets for the same model.

---

## 19. UML Class Diagram (Text Form)

```
«interface» IFrameworkAdapter
  + load(path: str) : IMR
  + supports(extension: str) : bool

«interface» IQuantizationStrategy
  + apply(imr: IMR, plan: QuantizationPlan) : IMR
  + name() : str

«interface» ICompressionStrategy
  + apply(imr: IMR, plan: CompressionPlan) : IMR
  + name() : str

«interface» IExporterBackend
  + export(imr: IMR, target: HardwareProfile) : DeploymentArtifact
  + supported_targets() : List[str]

«interface» IReportRenderer
  + render(context: PipelineContext) : ReportDocument
  + report_type() : str

«interface» ILogger
  + debug/info/warning/error/critical(msg: str, **fields) : None

class PipelineOrchestrator
  - stages: List[PipelineStage]
  - context: PipelineContext
  - logger: ILogger
  + run() : RunResult
  + resume(run_id: str) : RunResult

class PipelineContext
  - run_id: str
  - stage_results: Dict[str, StageResult]
  - errors: List[UAQEError]
  + append(stage_name: str, result: StageResult) : None
  + get(stage_name: str) : StageResult

abstract class PipelineStage
  + execute(context: PipelineContext) : StageResult

class ModelLoader implements PipelineStage
  - adapters: List[IFrameworkAdapter]
  - logger: ILogger
  + execute(context) : StageResult

class ModelAnalyzer implements PipelineStage
  + execute(context) : StageResult

class HardwareManager implements PipelineStage
  - hardwareRepo: IHardwareProfileRepository
  + execute(context) : StageResult
  + checkCompatibility(imr: IMR, profile: HardwareProfile) : CompatibilityReport

class QuantizationEngine implements PipelineStage
  - strategy: IQuantizationStrategy
  + execute(context) : StageResult

class CompressionEngine implements PipelineStage
  - strategy: ICompressionStrategy
  + execute(context) : StageResult

class OptimizationEngine implements PipelineStage
  + execute(context) : StageResult

class Exporter implements PipelineStage
  - backends: List[IExporterBackend]
  + execute(context) : StageResult

class Evaluator implements PipelineStage
  + execute(context) : StageResult

class Benchmarker implements PipelineStage
  + execute(context) : StageResult

class OptimizationAdvisor implements PipelineStage
  + execute(context) : StageResult

class ReportGenerator implements PipelineStage
  - renderers: List[IReportRenderer]
  + execute(context) : StageResult

PipelineOrchestrator "1" o── "many" PipelineStage
PipelineOrchestrator "1" ──── "1" PipelineContext
ModelLoader ..> IFrameworkAdapter : uses
QuantizationEngine ..> IQuantizationStrategy : uses
CompressionEngine ..> ICompressionStrategy : uses
Exporter ..> IExporterBackend : uses
ReportGenerator ..> IReportRenderer : uses
(all PipelineStage subclasses) ..> ILogger : uses
```

Full method signatures, constructor parameters, exceptions, and return types for every class above are defined per-file in `03_API_Specification.md`. Names, structure, and relationships shown here are considered final pending sign-off and will be permanently frozen in `09_Architecture_Lock.md`.

---

## 20. Document Control

| Field | Value |
|---|---|
| Depends on | None (this is the root document) |
| Consumed by | `02` through `10`, all future implementation work |
| Change policy | Free to revise until `09_Architecture_Lock.md` is approved; frozen thereafter |

**End of `01_Project_Architecture.md`.**
Awaiting approval to proceed to `02_Folder_Structure.md`.
