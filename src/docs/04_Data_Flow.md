# 04_Data_Flow.md
## Universal AI Quantization Engine — Data Flow Specification

**Depends on:** `01_Project_Architecture.md`, `02_Folder_Structure.md`, `03_API_Specification.md`
**Status:** DRAFT — pending Architecture Lock
**Scope:** Describes how data moves between stages, what each stage reads from and writes to `PipelineContext`, and the exact `StageResult.payload` type produced at each step. No implementation code. All class/method names below are identical to `03_API_Specification.md`; no new names are introduced.

**Notation conventions used below:**
- `ctx.get("<stage_name>")` reads a prior `StageResult` from `PipelineContext`
- `ctx.append("<stage_name>", result)` writes a `StageResult` to `PipelineContext`
- Diagrams use plain-text box/arrow notation for portability across renderers

---

## 1. Global Data Flow Contract

Every stage in the pipeline is a `PipelineStage` subclass. Every stage:

1. Reads zero or more prior `StageResult.payload` objects via `PipelineContext.get(stage_name)`.
2. Performs domain logic (no I/O beyond what its constructor-injected collaborators provide).
3. Returns exactly one `StageResult`, whose `payload` becomes readable by all downstream stages via `context.append(stage_name, result)`.

No stage may read another stage's private fields directly. No stage may reach into `workdir/`, `outputs/`, or `reports/` except through `FilesystemRepository`, which is only ever injected into `Exporter`, `ReportGenerator`, and `PipelineOrchestrator` (for checkpointing).

```
PipelineOrchestrator.run()
   │
   ├── for stage in stages:
   │       result = stage.execute(context)
   │       context.append(stage.name(), result)
   │       if not result.success:
   │           apply ExecutionConfig.on_error policy
   │
   └── return RunResult(stage_results=context.stage_results, ...)
```

---

## 2. Input Flow

**Stages involved:** `ModelIngestionService` → `ModelLoader`

```
RunRequest (model_path, hardware_profile_id, dataset paths, config_overrides)
        │
        ▼
ModelIngestionService.ingest(request, context)
        │  invokes
        ▼
ModelLoader.execute(context)
        │
        ├── detect_framework(path) -> str          [".pth"->torch, ".onnx"->onnx, etc.]
        │
        ├── select matching IFrameworkAdapter from self.adapters
        │       (first adapter where adapter.supports(extension) is True)
        │
        ├── adapter.load(path) -> IMR
        │       raises ModelLoadError on failure
        │
        └── returns StageResult(stage_name="model_loader", payload=IMR)
```

**Context writes:** `ctx.append("model_loader", StageResult(payload=IMR))`
**Downstream readers:** `ModelValidator`, `ModelAnalyzer`

**Failure path:** `ModelLoadError` propagates to `PipelineOrchestrator`, which applies `ExecutionConfig.on_error` (`ABORT` halts the run; `SKIP_STAGE` is invalid here since no valid IMR exists — `ModelLoader` failures always force `ABORT` regardless of configured policy).

---

## 3. Validation Flow

**Stages involved:** `ModelValidator`

```
IMR (from ctx.get("model_loader").payload)
        │
        ▼
ModelValidator.execute(context)
        │
        ├── validate(imr) -> None
        │       checks:
        │         - IMR.layers is non-empty
        │         - every IMRLayer.inputs/outputs resolve to existing tensor names
        │         - IMRMetadata.op_count matches len(imr.layers)
        │       raises ModelValidationError on any failure
        │
        └── returns StageResult(stage_name="model_validator", payload=None, warnings=[...])
```

**Context writes:** `ctx.append("model_validator", StageResult(payload=None))`
**Downstream readers:** none consume `payload` directly (validation is a gate, not a data source); downstream stages rely on `ctx.has("model_validator")` and `result.success` as a precondition check before reading `model_loader`'s IMR.

---

## 4. Analysis Flow

**Stages involved:** `ModelAnalyzer` → `LayerCompatibilityChecker` → `HardwareManager`

```
IMR (from ctx.get("model_loader").payload)
        │
        ▼
ModelAnalyzer.execute(context)
        │
        ├── analyze(imr) -> AnalysisResult
        │       { layer_graph_summary, parameter_count,
        │         estimated_flops, estimated_memory_bytes,
        │         op_type_histogram }
        │
        └── StageResult(stage_name="model_analyzer", payload=AnalysisResult)

        │
        ▼
HardwareManager  (reads HardwareConfig.target_profile_id from RunRequest/config_overrides)
        │
        ├── resolves HardwareProfile via IHardwareProfileRepository.get(profile_id)
        │
        └── makes HardwareProfile available on context as part of
            "hardware_manager" StageResult.payload

        │
        ▼
LayerCompatibilityChecker.execute(context)
        │
        ├── reads AnalysisResult (from "model_analyzer")
        ├── reads HardwareProfile (from "hardware_manager")
        │
        ├── produces CompatibilityReport
        │       { compatible: bool,
        │         unsupported_layers: List[str],
        │         constraint_violations: List[str] }
        │
        └── StageResult(stage_name="layer_compatibility_checker",
                         payload=CompatibilityReport)
```

**Context writes:** `"model_analyzer"` → `AnalysisResult`; `"hardware_manager"` → `HardwareProfile`; `"layer_compatibility_checker"` → `CompatibilityReport`

**Downstream readers:** `QuantizationAdvisor`, `CompressionAdvisor`, `OptimizationEngine`, `Exporter` all read `HardwareProfile`. `QuantizationAdvisor` additionally reads `CompatibilityReport.unsupported_layers` to exclude those layers from quantization candidacy.

**Failure path:** if `CompatibilityReport.compatible is False` and `ExecutionConfig.on_error == ABORT`, the run halts before quantization begins; `HardwareIncompatibilityError` is raised by `LayerCompatibilityChecker.execute()` (mapped internally, not shown in `03_API_Specification.md` as it is a domain-level guard raised from within `execute()`, not a distinct method signature).

---

## 5. Quantization Flow

**Stages involved:** `QuantizationAdvisor` → `CalibrationEngine` → `SensitivityAnalyzer` → `QuantizationEngine`

```
AnalysisResult, HardwareProfile, CompatibilityReport
        │
        ▼
QuantizationAdvisor.execute(context)
        │
        ├── produces a recommended QuantizationConfig
        │       (default_precision, per_layer_overrides={}, ...)
        │
        └── StageResult("quantization_advisor", payload=QuantizationConfig)

        │
        ▼
CalibrationEngine.execute(context)
        │
        ├── reads calibration_dataset_path (from RunRequest, via context)
        ├── reads IMR (from "model_loader")
        ├── produces calibration statistics (per-layer activation ranges)
        │
        └── StageResult("calibration_engine", payload=CalibrationStats)

        │
        ▼
SensitivityAnalyzer.execute(context)
        │
        ├── reads IMR, CalibrationStats, QuantizationConfig
        ├── perturbs each candidate layer's precision and measures
        │   accuracy delta against QuantizationConfig.sensitivity_threshold
        │
        └── StageResult("sensitivity_analyzer", payload=SensitivityReport)
                { per_layer_sensitivity: Dict[str, float] }

        │
        ▼
QuantizationEngine.execute(context)
        │
        ├── reads IMR, QuantizationConfig, SensitivityReport
        ├── builds final per_layer_overrides using sensitivity results
        │   (layers above threshold retained at higher precision)
        │
        ├── for each IQuantizationStrategy candidate:
        │       strategy.apply(imr, plan) -> IMR
        │       raises QuantizationError
        │
        └── StageResult("quantization_engine", payload=IMR)   # quantized IMR
```

**Context writes:** `"quantization_advisor"` → `QuantizationConfig`; `"calibration_engine"` → `CalibrationStats`; `"sensitivity_analyzer"` → `SensitivityReport`; `"quantization_engine"` → quantized `IMR`

**Downstream readers:** `CompressionAdvisor`, `CompressionEngine`, `OptimizationEngine`, `Exporter`, `Evaluator` all read the quantized `IMR` from `"quantization_engine"` rather than the original `"model_loader"` IMR from this point forward in the pipeline.

**Important invariant:** once `"quantization_engine"` exists in `PipelineContext`, it — not `"model_loader"` — is the canonical current-IMR source. Each subsequent stage that mutates the IMR (`CompressionEngine`, `OptimizationEngine`, `MemoryOptimizer`) writes its own new `StageResult` payload; it never overwrites a previous stage's result (`PipelineContext.append` is additive, never mutating, per Contract Rule 4 in `03_API_Specification.md`).

---

## 6. Compression Flow

**Stages involved:** `CompressionAdvisor` → `CompressionEngine`

```
quantized IMR (from "quantization_engine"), HardwareProfile, AnalysisResult
        │
        ▼
CompressionAdvisor.execute(context)
        │
        ├── produces recommended CompressionConfig
        │       (enabled_types, target_ratio, pruning_sparsity)
        │
        └── StageResult("compression_advisor", payload=CompressionConfig)

        │
        ▼
CompressionEngine.execute(context)
        │
        ├── reads quantized IMR, CompressionConfig
        ├── for each ICompressionStrategy in CompressionConfig.enabled_types:
        │       strategy.apply(imr, plan) -> IMR
        │       raises CompressionError
        │
        └── StageResult("compression_engine", payload=IMR)   # compressed IMR
```

**Context writes:** `"compression_advisor"` → `CompressionConfig`; `"compression_engine"` → compressed `IMR`

**Downstream readers:** `OptimizationEngine`, `MemoryOptimizer`, `Exporter`, `Evaluator` read the compressed IMR from `"compression_engine"`.

---

## 7. Optimization Flow

**Stages involved:** `OptimizationEngine` → `MemoryOptimizer`

```
compressed IMR (from "compression_engine"), HardwareProfile, OptimizationConfig
        │
        ▼
OptimizationEngine.execute(context)
        │
        ├── multi-objective search over OptimizationConfig.objectives
        │   (e.g. accuracy, latency, memory) weighted by objective_weights,
        │   bounded by max_search_iterations
        │
        ├── candidate IMR variants are evaluated internally against
        │   HardwareProfile constraints (RAM, flash, max model size)
        │
        └── StageResult("optimization_engine", payload=IMR)   # optimized IMR

        │
        ▼
MemoryOptimizer.execute(context)
        │
        ├── reads optimized IMR, HardwareProfile
        ├── applies memory-layout-specific transforms
        │   (e.g. tensor arena reuse planning for embedded targets)
        │
        └── StageResult("memory_optimizer", payload=IMR)   # final pre-export IMR
```

**Context writes:** `"optimization_engine"` → optimized `IMR`; `"memory_optimizer"` → final `IMR`

**Downstream readers:** `Exporter`, `Evaluator`, `Benchmarker` all read the final IMR from `"memory_optimizer"` — this is the last IMR mutation point in the pipeline. No stage after `MemoryOptimizer` may alter tensor data or layer graph structure.

---

## 8. Export Flow

**Stages involved:** `Exporter`

```
final IMR (from "memory_optimizer"), HardwareProfile
        │
        ▼
Exporter.execute(context)
        │
        ├── selects IExporterBackend matching HardwareProfile.hardware_class
        │   and HardwareProfile.target_profile_id via IExporterBackend.supported_targets()
        │
        ├── backend.export(imr, target=HardwareProfile) -> DeploymentArtifact
        │       raises ExportError
        │
        ├── writes artifact bytes via FilesystemRepository.write_bytes(...)
        │   into outputs/<run_id>/{fpga|embedded|raspberrypi}/
        │
        └── StageResult("exporter", payload=DeploymentArtifact)
                { file_paths: List[str], export_format: ExportFormat }
```

**Context writes:** `"exporter"` → `DeploymentArtifact`
**Downstream readers:** `Evaluator`, `Benchmarker`, `ReportGenerator`

**Backend resolution table** (which `IExporterBackend` implementation is selected):

| `HardwareProfile.hardware_class` | `target_profile_id` examples | Backend class |
|---|---|---|
| `FPGA` | `artix7`, `zynq7000`, `kintex`, `cyclonev` | `Artix7Backend`, `Zynq7000Backend`, `KintexBackend`, `CycloneVBackend` |
| `EMBEDDED` | `esp32`, `esp32s3`, `stm32f4`, `stm32h7`, `rp2040`, `portenta_h7` | `Esp32Backend`, `Esp32S3Backend`, `Stm32F4Backend`, `Stm32H7Backend`, `Rp2040Backend`, `PortentaH7Backend` |
| `RASPBERRY_PI` | `raspberrypi4`, `raspberrypi5` | `RaspberryPi4Backend`, `RaspberryPi5Backend` |

---

## 9. Evaluation Flow

**Stages involved:** `Evaluator`

```
final IMR (from "memory_optimizer"), DeploymentArtifact (from "exporter"),
evaluation_dataset_path (from RunRequest)
        │
        ▼
Evaluator.execute(context)
        │
        ├── runs inference over the evaluation dataset using the exported
        │   artifact (or the pre-export IMR, for frameworks lacking a
        │   target-native runtime in the dev environment)
        │
        ├── compares against original (pre-quantization) IMR outputs
        │   as an accuracy baseline
        │
        └── StageResult("evaluator", payload=EvaluationResult)
                { accuracy_delta: float, per_class_metrics: Dict[str, float] }
```

**Context writes:** `"evaluator"` → `EvaluationResult`
**Downstream readers:** `DeploymentReadinessScorer`, `ReportGenerator`

---

## 10. Benchmark Flow

**Stages involved:** `Benchmarker`

```
DeploymentArtifact (from "exporter"), HardwareProfile
        │
        ▼
Benchmarker.execute(context)
        │
        ├── run_benchmark(artifact, profile, trials) -> BenchmarkResult
        │       measures latency, throughput, peak memory on the
        │       target profile (real hardware if attached, else
        │       profile-derived simulation)
        │
        └── StageResult("benchmarker", payload=BenchmarkResult)
                { latency_ms, throughput_ops_sec, peak_memory_bytes }
```

**Context writes:** `"benchmarker"` → `BenchmarkResult`
**Downstream readers:** `DeploymentReadinessScorer`, `OptimizationAdvisor`, `ReportGenerator`

---

## 11. Advisory Flow

**Stages involved:** `DeploymentReadinessScorer` → `OptimizationAdvisor`

```
EvaluationResult (from "evaluator"), BenchmarkResult (from "benchmarker"),
HardwareProfile (from "hardware_manager")
        │
        ▼
DeploymentReadinessScorer.execute(context)
        │
        ├── score(evaluation, benchmark, profile) -> ReadinessScore
        │       { score_0_to_100, category, contributing_factors }
        │
        └── StageResult("deployment_readiness_scorer", payload=ReadinessScore)

        │
        ▼
OptimizationAdvisor.execute(context)
        │
        ├── reads full PipelineContext history
        │   (AnalysisResult, SensitivityReport, EvaluationResult,
        │    BenchmarkResult, ReadinessScore)
        │
        ├── generate_recommendations(context) -> List[Recommendation]
        │       each { title, description, priority }
        │
        └── StageResult("optimization_advisor", payload=List[Recommendation])
```

**Context writes:** `"deployment_readiness_scorer"` → `ReadinessScore`; `"optimization_advisor"` → `List[Recommendation]`
**Downstream readers:** `ReportGenerator`

---

## 12. Report Flow

**Stages involved:** `ReportGenerator`

```
Full PipelineContext (every prior StageResult)
        │
        ▼
ReportGenerator.execute(context)
        │
        ├── for each IReportRenderer in self.renderers:
        │       renderer.render(context) -> ReportDocument
        │           { report_type, content, file_path }
        │
        ├── writes each ReportDocument via FilesystemRepository.write_text(...)
        │   into reports/<run_id>/
        │
        └── StageResult("report_generator", payload=List[ReportDocument])
```

**Report-to-source mapping:**

| Report | `report_type` | Primary `PipelineContext` sources |
|---|---|---|
| Accuracy Report | `"accuracy"` | `"evaluator"`, `"sensitivity_analyzer"` |
| Benchmark Report | `"benchmark"` | `"benchmarker"`, `"hardware_manager"` |
| Compression Report | `"compression"` | `"compression_advisor"`, `"compression_engine"` |
| Deployment Report | `"deployment"` | `"exporter"`, `"deployment_readiness_scorer"` |
| Summary Report | `"summary"` | all prior `StageResult`s |

**Context writes:** `"report_generator"` → `List[ReportDocument]`
**Downstream readers:** none within the pipeline; `RunResult.output_paths` is populated from `"exporter"` and `"report_generator"` file paths by `PipelineOrchestrator.run()` after the final stage completes.

---

## 13. End-to-End Data Flow Diagram

```
RunRequest
   │
   ▼
[model_loader]──IMR──▶[model_validator]
   │
   ▼ IMR
[model_analyzer]──AnalysisResult─┐
   │                             │
   ▼                             ▼
[hardware_manager]──HardwareProfile──▶[layer_compatibility_checker]──CompatibilityReport
   │                                            │
   └────────────────────┬───────────────────────┘
                         ▼
                 [quantization_advisor]──QuantizationConfig
                         │
                         ▼
                 [calibration_engine]──CalibrationStats
                         │
                         ▼
                 [sensitivity_analyzer]──SensitivityReport
                         │
                         ▼
                 [quantization_engine]──IMR(quantized)
                         │
                         ▼
                 [compression_advisor]──CompressionConfig
                         │
                         ▼
                 [compression_engine]──IMR(compressed)
                         │
                         ▼
                 [optimization_engine]──IMR(optimized)
                         │
                         ▼
                 [memory_optimizer]──IMR(final)
                         │
                         ▼
                    [exporter]──DeploymentArtifact
                    ╱          ╲
                   ▼            ▼
           [evaluator]    [benchmarker]
        EvaluationResult   BenchmarkResult
                   ╲            ╱
                    ▼          ▼
            [deployment_readiness_scorer]──ReadinessScore
                         │
                         ▼
                 [optimization_advisor]──List[Recommendation]
                         │
                         ▼
                 [report_generator]──List[ReportDocument]
                         │
                         ▼
                     RunResult
```

---

## 14. Data Flow Rules (apply across all sections above)

1. Every IMR mutation produces a **new** `StageResult` under a new stage-name key; `PipelineContext` never overwrites an existing key (append-only, per `03_API_Specification.md` Contract Rule 4).
2. Any stage needing the "current" IMR must read the **latest** stage in the chain `model_loader → quantization_engine → compression_engine → optimization_engine → memory_optimizer`, never assume a fixed key name is always "the IMR" — resolution order is fixed by pipeline position, not by convention.
3. `HardwareProfile` is resolved exactly once, by `HardwareManager`, and is read (never re-fetched) by every downstream stage that needs it.
4. No stage other than `Exporter` and `ReportGenerator` (via `FilesystemRepository`) may write to `outputs/` or `reports/`.
5. `PipelineOrchestrator.resume(run_id)` reconstructs `PipelineContext` from `workdir/<run_id>/checkpoint.json` and re-enters the flow at the first stage whose `StageResult` is absent from the checkpoint — all flows above must therefore be idempotent and side-effect-free when re-run against an already-populated context entry (a stage never re-executes if `context.has(stage.name())` is already `True`).

---

**End of `04_Data_Flow.md`.**
