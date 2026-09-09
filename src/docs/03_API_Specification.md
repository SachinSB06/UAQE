# 03_API_Specification.md
## Universal AI Quantization Engine — API / Contract Specification

**Depends on:** `01_Project_Architecture.md`, `02_Folder_Structure.md`
**Status:** DRAFT — pending Architecture Lock
**Scope:** Defines class names, method signatures, constructor parameters, inputs, outputs, return types, exceptions, visibility, and inheritance for every file in `02_Folder_Structure.md`. **No implementation code is included.** All signatures are contracts only.

**Notation conventions used below:**
- `+` public, `-` private, `#` protected
- `{abstract}` marks an abstract method/class
- `{static}` marks a static/class method
- Types reference `uaqe.common.types`, `uaqe.common.value_objects`, and `uaqe.common.result_types` unless otherwise noted.

---

## 1. `uaqe/common/`

### 1.1 `imr.py`
```
class IMR
  + layers: List[IMRLayer]
  + metadata: IMRMetadata
  + get_layer(name: str) -> IMRLayer
  + topological_order() -> List[IMRLayer]

class IMRLayer
  + name: str
  + op_type: str
  + inputs: List[str]
  + outputs: List[str]
  + parameters: Dict[str, IMRTensor]
  + attributes: Dict[str, Any]
  + precision: Precision

class IMRTensor
  + shape: Tuple[int, ...]
  + dtype: str
  + data: bytes            # opaque buffer handle, reference-counted (see 01 §17)
  + is_shared() -> bool

class IMRMetadata
  + source_framework: str
  + source_format: str
  + original_input_shapes: Dict[str, Tuple[int, ...]]
  + op_count: int
  + total_parameters: int
```

### 1.2 `types.py`
```
enum Precision: FP32, FP16, INT8, INT4, MIXED
enum HardwareClass: FPGA, EMBEDDED, RASPBERRY_PI
enum CompressionType: PRUNING, WEIGHT_CLUSTERING, HUFFMAN, RLE, NONE
enum ExportFormat: MEM, HEX, BIN, TFLITE, ONNX, H_HEADER
enum PipelineErrorPolicy: ABORT, SKIP_STAGE, BEST_EFFORT
```

### 1.3 `value_objects.py`
```
@immutable class QuantizationConfig
  + default_precision: Precision
  + per_layer_overrides: Dict[str, Precision]
  + calibration_batch_size: int
  + sensitivity_threshold: float

@immutable class CompressionConfig
  + enabled_types: List[CompressionType]
  + target_ratio: float
  + pruning_sparsity: float

@immutable class HardwareConfig
  + target_profile_id: str
  + hardware_class: HardwareClass

@immutable class OptimizationConfig
  + objectives: List[str]          # e.g. ["accuracy", "latency", "memory"]
  + objective_weights: Dict[str, float]
  + max_search_iterations: int

@immutable class ExecutionConfig
  + parallelism: Dict[str, bool]
  + large_model_threshold_mb: int
  + on_error: PipelineErrorPolicy
```

### 1.4 `exceptions.py`
```
class UAQEError(Exception)
  + code: str
  + message: str
  + remediation_hint: Optional[str]
  + stage: Optional[str]

class ModelLoadError(UAQEError)
class ModelValidationError(UAQEError)
class UnsupportedLayerError(UAQEError)
class HardwareIncompatibilityError(UAQEError)
class QuantizationError(UAQEError)
class CompressionError(UAQEError)
class OptimizationError(UAQEError)
class ExportError(UAQEError)
class EvaluationError(UAQEError)
class BenchmarkError(UAQEError)
class ConfigurationError(UAQEError)
class PluginLoadError(UAQEError)
```

### 1.5 `result_types.py`
```
@dataclass class StageResult
  + stage_name: str
  + success: bool
  + payload: Any
  + warnings: List[str]
  + duration_ms: float

@dataclass class RunResult
  + run_id: str
  + success: bool
  + stage_results: Dict[str, StageResult]
  + errors: List[UAQEError]
  + output_paths: List[str]

@dataclass class CompatibilityReport
  + compatible: bool
  + unsupported_layers: List[str]
  + constraint_violations: List[str]
```

### 1.6 `interfaces/i_logger.py`
```
abstract class ILogger
  + {abstract} debug(msg: str, **fields) -> None
  + {abstract} info(msg: str, **fields) -> None
  + {abstract} warning(msg: str, **fields) -> None
  + {abstract} error(msg: str, **fields) -> None
  + {abstract} critical(msg: str, **fields) -> None
```

### 1.7 `interfaces/i_framework_adapter.py`
```
abstract class IFrameworkAdapter
  + {abstract} load(path: str) -> IMR
    raises ModelLoadError
  + {abstract} supports(extension: str) -> bool
```

### 1.8 `interfaces/i_quantization_strategy.py`
```
abstract class IQuantizationStrategy
  + {abstract} apply(imr: IMR, plan: QuantizationConfig) -> IMR
    raises QuantizationError
  + {abstract} name() -> str
```

### 1.9 `interfaces/i_compression_strategy.py`
```
abstract class ICompressionStrategy
  + {abstract} apply(imr: IMR, plan: CompressionConfig) -> IMR
    raises CompressionError
  + {abstract} name() -> str
```

### 1.10 `interfaces/i_exporter_backend.py`
```
abstract class IExporterBackend
  + {abstract} export(imr: IMR, target: HardwareProfile) -> DeploymentArtifact
    raises ExportError
  + {abstract} supported_targets() -> List[str]
```

### 1.11 `interfaces/i_report_renderer.py`
```
abstract class IReportRenderer
  + {abstract} render(context: "PipelineContext") -> ReportDocument
  + {abstract} report_type() -> str
```

### 1.12 `interfaces/i_hardware_profile_repository.py`
```
abstract class IHardwareProfileRepository
  + {abstract} get(profile_id: str) -> HardwareProfile
    raises ConfigurationError
  + {abstract} list_all() -> List[HardwareProfile]
```

### 1.13 `interfaces/i_config_repository.py`
```
abstract class IConfigRepository
  + {abstract} load_quantization_config() -> QuantizationConfig
  + {abstract} load_compression_config() -> CompressionConfig
  + {abstract} load_execution_config() -> ExecutionConfig
  + {abstract} load_optimization_config() -> OptimizationConfig
```

---

## 2. `uaqe/domain/pipeline/`

### 2.1 `pipeline_context.py`
```
class PipelineContext
  - run_id: str
  - stage_results: Dict[str, StageResult]
  - errors: List[UAQEError]
  + __init__(run_id: str)
  + append(stage_name: str, result: StageResult) -> None
  + get(stage_name: str) -> StageResult
    raises KeyError
  + has(stage_name: str) -> bool
```

### 2.2 `pipeline_stage.py`
```
abstract class PipelineStage
  + {abstract} execute(context: PipelineContext) -> StageResult
  + name() -> str
```

### 2.3 `pipeline_builder.py`
```
class PipelineBuilder
  - stage_registry: Dict[str, PipelineStage]
  + __init__(stage_registry: Dict[str, PipelineStage])
  + build(hardware_class: HardwareClass, config: ExecutionConfig) -> List[PipelineStage]
```

### 2.4 `pipeline_orchestrator.py`
```
class PipelineOrchestrator
  - stages: List[PipelineStage]
  - context: PipelineContext
  - logger: ILogger
  - error_policy: PipelineErrorPolicy
  + __init__(stages: List[PipelineStage], context: PipelineContext, logger: ILogger, error_policy: PipelineErrorPolicy)
  + run() -> RunResult
  + resume(run_id: str) -> RunResult
    raises ConfigurationError
```

---

## 3. `uaqe/domain/model/`

### 3.1 `model_loader.py`
```
class ModelLoader(PipelineStage)
  - adapters: List[IFrameworkAdapter]
  - logger: ILogger
  + __init__(adapters: List[IFrameworkAdapter], logger: ILogger)
  + execute(context: PipelineContext) -> StageResult
  + detect_framework(path: str) -> str
    raises ModelLoadError
```

### 3.2 `model_validator.py`
```
class ModelValidator(PipelineStage)
  - logger: ILogger
  + __init__(logger: ILogger)
  + execute(context: PipelineContext) -> StageResult
  + validate(imr: IMR) -> None
    raises ModelValidationError
```

### 3.3 `model_analyzer.py`
```
@dataclass class AnalysisResult
  + layer_graph_summary: Dict[str, Any]
  + parameter_count: int
  + estimated_flops: int
  + estimated_memory_bytes: int
  + op_type_histogram: Dict[str, int]

class ModelAnalyzer(PipelineStage)
  - logger: ILogger
  + __init__(logger: ILogger)
  + execute(context: PipelineContext) -> StageResult
  + analyze(imr: IMR) -> AnalysisResult
```

---

## 4. `uaqe/domain/compatibility/`

### 4.1 `layer_compatibility_checker.py`
```
class LayerCompatibilityChecker(PipelineStage)
  - hardware_repo: IHardwareProfileRepository
  - logger: ILogger
  + __init__(hardware_repo: IHardwareProfileRepository, logger: ILogger)
  + execute(context: PipelineContext) -> StageResult
  + find_unsupported_layers(imr: IMR, profile: HardwareProfile) -> List[str]
```

---

## 5. `uaqe/domain/hardware/`

### 5.1 `hardware_manager.py`
```
@dataclass class HardwareProfile
  + profile_id: str
  + hardware_class: HardwareClass
  + ram_bytes: int
  + flash_bytes: int
  + tensor_memory_bytes: int
  + runtime: str
  + supported_precisions: List[Precision]
  + max_model_size_bytes: int
  + preferred_export_format: ExportFormat

class HardwareManager(PipelineStage)
  - repo: IHardwareProfileRepository
  - logger: ILogger
  + __init__(repo: IHardwareProfileRepository, logger: ILogger)
  + execute(context: PipelineContext) -> StageResult
  + check_compatibility(imr: IMR, profile: HardwareProfile) -> CompatibilityReport
    raises HardwareIncompatibilityError
```

---

## 6. `uaqe/domain/quantization/`

### 6.1 `quantization_advisor.py`
```
@dataclass class QuantizationPlan
  + per_layer_precision: Dict[str, Precision]
  + rationale: Dict[str, str]

class QuantizationAdvisor(PipelineStage)
  - logger: ILogger
  + __init__(logger: ILogger)
  + execute(context: PipelineContext) -> StageResult
  + recommend(analysis: AnalysisResult, profile: HardwareProfile, config: QuantizationConfig) -> QuantizationPlan
```

### 6.2 `calibration_engine.py`
```
@dataclass class CalibrationStats
  + per_layer_activation_range: Dict[str, Tuple[float, float]]
  + per_layer_histogram: Dict[str, List[float]]

class CalibrationEngine(PipelineStage)
  - logger: ILogger
  + __init__(logger: ILogger)
  + execute(context: PipelineContext) -> StageResult
  + calibrate(imr: IMR, dataset_path: str, batch_size: int) -> CalibrationStats
    raises QuantizationError
```

### 6.3 `sensitivity_analyzer.py`
```
@dataclass class SensitivityReport
  + per_layer_accuracy_delta: Dict[str, float]
  + flagged_layers: List[str]

class SensitivityAnalyzer(PipelineStage)
  - logger: ILogger
  + __init__(logger: ILogger)
  + execute(context: PipelineContext) -> StageResult
  + analyze(imr: IMR, plan: QuantizationPlan, stats: CalibrationStats) -> SensitivityReport
```

### 6.4 `quantization_engine.py`
```
class QuantizationEngine(PipelineStage)
  - strategy: IQuantizationStrategy
  - logger: ILogger
  + __init__(strategy: IQuantizationStrategy, logger: ILogger)
  + execute(context: PipelineContext) -> StageResult
    raises QuantizationError
```

---

## 7. `uaqe/domain/compression/`

### 7.1 `compression_advisor.py`
```
@dataclass class CompressionPlan
  + selected_types: List[CompressionType]
  + target_ratio: float
  + rationale: str

class CompressionAdvisor(PipelineStage)
  - logger: ILogger
  + __init__(logger: ILogger)
  + execute(context: PipelineContext) -> StageResult
  + recommend(analysis: AnalysisResult, profile: HardwareProfile, config: CompressionConfig) -> CompressionPlan
```

### 7.2 `compression_engine.py`
```
class CompressionEngine(PipelineStage)
  - strategy: ICompressionStrategy
  - logger: ILogger
  + __init__(strategy: ICompressionStrategy, logger: ILogger)
  + execute(context: PipelineContext) -> StageResult
    raises CompressionError
```

---

## 8. `uaqe/domain/optimization/`

### 8.1 `optimization_engine.py`
```
@dataclass class OptimizationResult
  + selected_configuration: Dict[str, Any]
  + pareto_front: List[Dict[str, Any]]
  + objective_scores: Dict[str, float]

class OptimizationEngine(PipelineStage)
  - logger: ILogger
  + __init__(logger: ILogger)
  + execute(context: PipelineContext) -> StageResult
    raises OptimizationError
  + search(imr: IMR, config: OptimizationConfig) -> OptimizationResult
```

### 8.2 `memory_optimizer.py`
```
@dataclass class MemoryPlan
  + buffer_arena_bytes: int
  + tensor_reuse_map: Dict[str, str]
  + peak_memory_bytes: int

class MemoryOptimizer(PipelineStage)
  - logger: ILogger
  + __init__(logger: ILogger)
  + execute(context: PipelineContext) -> StageResult
  + plan(imr: IMR, profile: HardwareProfile) -> MemoryPlan
    raises HardwareIncompatibilityError
```

---

## 9. `uaqe/domain/export/`

### 9.1 `exporter.py`
```
@dataclass class DeploymentArtifact
  + file_paths: List[str]
  + export_format: ExportFormat
  + target_profile_id: str
  + size_bytes: int

class Exporter(PipelineStage)
  - backends: List[IExporterBackend]
  - logger: ILogger
  + __init__(backends: List[IExporterBackend], logger: ILogger)
  + execute(context: PipelineContext) -> StageResult
    raises ExportError
  + select_backend(profile: HardwareProfile) -> IExporterBackend
    raises ExportError
```

---

## 10. `uaqe/domain/evaluation/`

### 10.1 `evaluator.py`
```
@dataclass class EvaluationResult
  + baseline_accuracy: float
  + optimized_accuracy: float
  + accuracy_delta: float
  + per_layer_error: Dict[str, float]

class Evaluator(PipelineStage)
  - logger: ILogger
  + __init__(logger: ILogger)
  + execute(context: PipelineContext) -> StageResult
    raises EvaluationError
  + evaluate(baseline_imr: IMR, optimized_imr: IMR, dataset_path: str) -> EvaluationResult
```

---

## 11. `uaqe/domain/benchmark/`

### 11.1 `benchmarker.py`
```
@dataclass class BenchmarkResult
  + latency_ms_p50: float
  + latency_ms_p99: float
  + throughput_inferences_per_sec: float
  + peak_memory_bytes: int

class Benchmarker(PipelineStage)
  - logger: ILogger
  + __init__(logger: ILogger)
  + execute(context: PipelineContext) -> StageResult
    raises BenchmarkError
  + run_benchmark(artifact: DeploymentArtifact, profile: HardwareProfile, trials: int) -> BenchmarkResult
```

---

## 12. `uaqe/domain/advisory/`

### 12.1 `deployment_readiness_scorer.py`
```
@dataclass class ReadinessScore
  + score_0_to_100: float
  + category: str            # e.g. "Ready", "Marginal", "Not Recommended"
  + contributing_factors: Dict[str, float]

class DeploymentReadinessScorer(PipelineStage)
  - logger: ILogger
  + __init__(logger: ILogger)
  + execute(context: PipelineContext) -> StageResult
  + score(evaluation: EvaluationResult, benchmark: BenchmarkResult, profile: HardwareProfile) -> ReadinessScore
```

### 12.2 `optimization_advisor.py`
```
@dataclass class Recommendation
  + title: str
  + description: str
  + priority: str             # "High" | "Medium" | "Low"

class OptimizationAdvisor(PipelineStage)
  - logger: ILogger
  + __init__(logger: ILogger)
  + execute(context: PipelineContext) -> StageResult
  + generate_recommendations(context: PipelineContext) -> List[Recommendation]
```

---

## 13. `uaqe/domain/reporting/`

### 13.1 `report_generator.py`
```
@dataclass class ReportDocument
  + report_type: str
  + content: str
  + file_path: str

class ReportGenerator(PipelineStage)
  - renderers: List[IReportRenderer]
  - logger: ILogger
  + __init__(renderers: List[IReportRenderer], logger: ILogger)
  + execute(context: PipelineContext) -> StageResult
```

---

## 14. `uaqe/application/`

### 14.1 `run_request.py`
```
@dataclass class RunRequest
  + model_path: str
  + hardware_profile_id: str
  + calibration_dataset_path: Optional[str]
  + evaluation_dataset_path: Optional[str]
  + config_overrides: Dict[str, Any]
```

### 14.2 `model_ingestion_service.py`
```
class ModelIngestionService
  - model_loader: ModelLoader
  - logger: ILogger
  + __init__(model_loader: ModelLoader, logger: ILogger)
  + ingest(request: RunRequest, context: PipelineContext) -> StageResult
```

### 14.3 `hardware_selection_service.py`
```
class HardwareSelectionService
  - hardware_repo: IHardwareProfileRepository
  + __init__(hardware_repo: IHardwareProfileRepository)
  + select(profile_id: str) -> HardwareProfile
    raises ConfigurationError
```

### 14.4 `session_manager.py`
```
class SessionManager
  - active_runs: Dict[str, PipelineContext]
  + __init__()
  + create_session(run_id: str) -> PipelineContext
  + get_session(run_id: str) -> PipelineContext
    raises KeyError
  + close_session(run_id: str) -> None
```

### 14.5 `workflow_controller.py`
```
class WorkflowController
  - orchestrator_factory: Callable[[RunRequest], PipelineOrchestrator]
  - session_manager: SessionManager
  + __init__(orchestrator_factory: Callable[[RunRequest], PipelineOrchestrator], session_manager: SessionManager)
  + execute(request: RunRequest) -> RunResult
```

---

## 15. `uaqe/infrastructure/framework_adapters/`

Each adapter implements `IFrameworkAdapter` with an identical contract:
```
class TorchAdapter(IFrameworkAdapter)
  + load(path: str) -> IMR
    raises ModelLoadError
  + supports(extension: str) -> bool     # ".pth", ".pt"

class OnnxAdapter(IFrameworkAdapter)      # ".onnx"
class TensorFlowAdapter(IFrameworkAdapter) # ".pb"
class KerasAdapter(IFrameworkAdapter)      # ".h5", ".keras"
class TFLiteAdapter(IFrameworkAdapter)     # ".tflite"
```

---

## 16. `uaqe/infrastructure/exporter_backends/`

Each backend implements `IExporterBackend` with an identical contract; only `supported_targets()` and internal encoding differ:
```
class Artix7Backend(IExporterBackend)
  + export(imr: IMR, target: HardwareProfile) -> DeploymentArtifact
    raises ExportError
  + supported_targets() -> List[str]      # ["artix7"]

class Zynq7000Backend(IExporterBackend)
class KintexBackend(IExporterBackend)
class CycloneVBackend(IExporterBackend)
class Esp32Backend(IExporterBackend)
class Esp32S3Backend(IExporterBackend)
class Stm32F4Backend(IExporterBackend)
class Stm32H7Backend(IExporterBackend)
class Rp2040Backend(IExporterBackend)
class PortentaH7Backend(IExporterBackend)
class RaspberryPi4Backend(IExporterBackend)
class RaspberryPi5Backend(IExporterBackend)
```

---

## 17. `uaqe/infrastructure/repositories/`

### 17.1 `config_repository.py`
```
class ConfigRepository(IConfigRepository)
  - base_path: str
  + __init__(base_path: str)
  + load_quantization_config() -> QuantizationConfig
    raises ConfigurationError
  + load_compression_config() -> CompressionConfig
  + load_execution_config() -> ExecutionConfig
  + load_optimization_config() -> OptimizationConfig
```

### 17.2 `hardware_profile_repository.py`
```
class HardwareProfileRepository(IHardwareProfileRepository)
  - profiles_path: str
  - cache: Dict[str, HardwareProfile]
  + __init__(profiles_path: str)
  + get(profile_id: str) -> HardwareProfile
    raises ConfigurationError
  + list_all() -> List[HardwareProfile]
```

### 17.3 `filesystem_repository.py`
```
class FilesystemRepository
  + __init__(base_path: str)
  + write_bytes(relative_path: str, data: bytes) -> str
  + write_text(relative_path: str, text: str) -> str
  + read_bytes(relative_path: str) -> bytes
    raises FileNotFoundError
```

---

## 18. `uaqe/infrastructure/logging/`

### 18.1 `structured_logger.py`
```
class StructuredLogger(ILogger)
  - run_id: str
  - min_level: str
  - sink_paths: List[str]
  + __init__(run_id: str, min_level: str, sink_paths: List[str])
  + debug(msg: str, **fields) -> None
  + info(msg: str, **fields) -> None
  + warning(msg: str, **fields) -> None
  + error(msg: str, **fields) -> None
  + critical(msg: str, **fields) -> None
```

---

## 19. `uaqe/infrastructure/metrics/`

### 19.1 `metrics_collector.py`
```
class MetricsCollector
  - metrics: Dict[str, List[float]]
  + __init__()
  + record(name: str, value: float) -> None
  + summary() -> Dict[str, Dict[str, float]]   # {name: {"min":.., "max":.., "avg":..}}
```

---

## 20. `uaqe/infrastructure/plugins/`

### 20.1 `plugin_registry.py`
```
class PluginRegistry
  - quantization_strategies: Dict[str, IQuantizationStrategy]
  - compression_strategies: Dict[str, ICompressionStrategy]
  - exporter_backends: Dict[str, IExporterBackend]
  - report_renderers: Dict[str, IReportRenderer]
  + __init__()
  + register_quantization_strategy(strategy: IQuantizationStrategy) -> None
  + register_compression_strategy(strategy: ICompressionStrategy) -> None
  + register_exporter_backend(backend: IExporterBackend) -> None
  + register_report_renderer(renderer: IReportRenderer) -> None
  + discover(plugin_dir: str) -> None
    raises PluginLoadError
  + get_quantization_strategy(name: str) -> IQuantizationStrategy
    raises PluginLoadError
```

---

## 21. `uaqe/interface/`

### 21.1 `composition_root.py`
```
class CompositionRoot
  + {static} build_workflow_controller(config_path: str) -> WorkflowController
```

### 21.2 `cli/cli_entry_point.py`
```
class CliEntryPoint
  - controller: WorkflowController
  + __init__(controller: WorkflowController)
  + run(argv: List[str]) -> int    # process exit code
```

### 21.3 `api/rest_entry_point.py`
```
class RestEntryPoint
  - controller: WorkflowController
  + __init__(controller: WorkflowController)
  + handle_run_request(payload: dict) -> dict
    raises ConfigurationError
```

### 21.4 `batch/batch_runner.py`
```
class BatchRunner
  - controller: WorkflowController
  + __init__(controller: WorkflowController)
  + run_batch(requests: List[RunRequest]) -> List[RunResult]
```

---

## 22. Contract Rules (apply to every class above)

1. Every constructor parameter typed as an interface (`I*`) MUST be satisfied via injection from `CompositionRoot` — never instantiated internally.
2. Every method that can fail MUST raise a subclass of `UAQEError`; no method may raise a bare `Exception` or an unlisted exception type.
3. Every `PipelineStage.execute()` MUST read its inputs only via `PipelineContext.get(...)` and MUST return a `StageResult` — never `None`, never a raw domain object.
4. Dataclasses listed as stage payloads (`AnalysisResult`, `QuantizationPlan`, etc.) are immutable value objects; no method on a `PipelineStage` mutates a payload from a prior stage.
5. No signature above may be altered post-Architecture-Lock (`09_Architecture_Lock.md`) without a formal RFC.

---

**End of `03_API_Specification.md`.**
