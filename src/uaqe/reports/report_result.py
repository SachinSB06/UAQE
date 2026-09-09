"""Aggregation of a completed (or in-progress) run's ``PipelineContext``
into one flat, renderer-agnostic :class:`ReportResult`.

Every format-specific renderer in this package (:mod:`markdown_report`,
:mod:`html_report`, :mod:`json_report`, :mod:`csv_report`,
:mod:`pdf_report`, :mod:`summary_report`, :mod:`comparison_report`)
consumes the same :class:`ReportResult` rather than re-reading
``PipelineContext`` itself, so the "what did this run produce" question
is answered in exactly one place, per
``11_Implementation_Rules.md`` §19.6 ("Keep ``PipelineContext``
narrow... scratch state stays local to the stage that created it") —
this module is that one boundary translating context lookups into a
plain, immutable value object every renderer can format independently.

This mirrors the role
:class:`~uaqe.evaluation.evaluation_result.EvaluationResult` and
:class:`~uaqe.benchmark.benchmark_result.BenchmarkResult` already play
for their own packages, extended here to span every stage in the
pipeline rather than one package's own sub-evaluators.

Every extraction in :func:`build_report_result` is defensive
(``context.has(...)`` checked before ``context.get(...)``): a report
must always be producible from a partial run (e.g. under
``PipelineErrorPolicy.BEST_EFFORT`` or ``SKIP_STAGE``, per
``09_Architecture_Lock.md`` §13) — a missing upstream stage degrades
the corresponding ``ReportResult`` field to ``None`` and adds a
warning, it never raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from uaqe.analyzer.model_analyzer import AnalysisResult
from uaqe.benchmark.benchmark_result import BenchmarkResult
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.compression.compression_planner import CompressionPlan
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.evaluation.evaluation_result import EvaluationResult
from uaqe.hardware.memory_planner import MemoryPlan
from uaqe.optimizer.optimization_result import OptimizationResult
from uaqe.quantization.quantization_planner import QuantizationPlan

if TYPE_CHECKING:
    # Deferred per the same TYPE_CHECKING-guarded cross-package pattern
    # uaqe.evaluation.evaluator already uses for this exact type: this
    # module only needs DeploymentArtifact as a type hint for an
    # optional field, so importing uaqe.exporter at module scope would
    # otherwise pull in that package's full __init__ for no runtime
    # benefit here.
    from uaqe.exporter.exporter import DeploymentArtifact

#: Registered ``PipelineStage.name()`` context keys this module reads
#: from, exactly as registered by each owning package (see each
#: package's own ``_..._STAGE_NAME`` constants, which this module
#: mirrors rather than redefines a second, possibly drifting, copy of).
MODEL_LOADER_STAGE_NAME = "model_loader"
MODEL_VALIDATOR_STAGE_NAME = "model_validator"
MODEL_ANALYZER_STAGE_NAME = "model_analyzer"
HARDWARE_MANAGER_STAGE_NAME = "hardware_manager"
QUANTIZATION_PLANNER_STAGE_NAME = "quantization_planner"
COMPRESSION_PLANNER_STAGE_NAME = "compression_planner"
OPTIMIZER_STAGE_NAME = "optimizer"
MEMORY_OPTIMIZER_STAGE_NAME = "memory_optimizer"
EXPORTER_STAGE_NAME = "exporter"
EVALUATOR_STAGE_NAME = "evaluator"
BENCHMARKER_STAGE_NAME = "benchmarker"

#: Every stage this module summarizes into ``ReportResult.stage_summaries``,
#: in canonical pipeline order (``09_Architecture_Lock.md`` §12).
_KNOWN_STAGE_NAMES: tuple = (
    MODEL_LOADER_STAGE_NAME,
    MODEL_VALIDATOR_STAGE_NAME,
    MODEL_ANALYZER_STAGE_NAME,
    HARDWARE_MANAGER_STAGE_NAME,
    QUANTIZATION_PLANNER_STAGE_NAME,
    COMPRESSION_PLANNER_STAGE_NAME,
    OPTIMIZER_STAGE_NAME,
    MEMORY_OPTIMIZER_STAGE_NAME,
    EXPORTER_STAGE_NAME,
    EVALUATOR_STAGE_NAME,
    BENCHMARKER_STAGE_NAME,
)

#: Fallback run identifier used when ``PipelineContext`` exposes no
#: usable run id (see :func:`resolve_run_id`).
_UNKNOWN_RUN_ID = "unknown-run"


@dataclass(frozen=True)
class ModelSummary:
    """A flattened view of ``\"model_analyzer\"``'s locked
    ``AnalysisResult`` payload.

    Attributes:
        parameter_count: Total parameter element count.
        estimated_flops: Total estimated FLOPs.
        estimated_memory_bytes: Total estimated memory footprint, in
            bytes.
        op_type_histogram: Per-``op_type`` layer counts.
    """

    parameter_count: int = 0
    estimated_flops: int = 0
    estimated_memory_bytes: int = 0
    op_type_histogram: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return this summary as a plain, JSON-serializable ``dict``."""
        return {
            "parameter_count": self.parameter_count,
            "estimated_flops": self.estimated_flops,
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "op_type_histogram": dict(self.op_type_histogram),
        }


@dataclass(frozen=True)
class HardwareSummary:
    """A flattened view of ``\"hardware_manager\"``'s ``HardwareProfile``
    payload.

    Attributes:
        profile_id: The resolved target's unique identifier.
        display_name: The resolved target's human-readable name.
        hardware_class: The resolved target's broad hardware category,
            as its ``.value`` string.
        runtime: The resolved target's runtime identifier.
        ram_bytes: The resolved target's total RAM, in bytes, if
            applicable.
        tensor_memory_bytes: The resolved target's usable working
            memory for activations/arena.
        max_model_size_bytes: The resolved target's maximum accepted
            total serialized model size.
    """

    profile_id: str = ""
    display_name: str = ""
    hardware_class: str = ""
    runtime: str = ""
    ram_bytes: Optional[int] = None
    tensor_memory_bytes: int = 0
    max_model_size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return this summary as a plain, JSON-serializable ``dict``."""
        return {
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "hardware_class": self.hardware_class,
            "runtime": self.runtime,
            "ram_bytes": self.ram_bytes,
            "tensor_memory_bytes": self.tensor_memory_bytes,
            "max_model_size_bytes": self.max_model_size_bytes,
        }


@dataclass(frozen=True)
class StageExecutionSummary:
    """One stage's execution outcome, for the run-progress table every
    renderer includes.

    Attributes:
        stage_name: The stage's registered context key.
        success: Whether the stage's ``StageResult.success`` was
            ``True``.
        duration_ms: The stage's recorded wall-clock duration.
        warning_count: The number of entries in the stage's
            ``StageResult.warnings``.
    """

    stage_name: str
    success: bool
    duration_ms: float = 0.0
    warning_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return this summary as a plain, JSON-serializable ``dict``."""
        return {
            "stage_name": self.stage_name,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "warning_count": self.warning_count,
        }


@dataclass(frozen=True)
class QuantizationSummary:
    """A flattened, serialization-friendly view of a ``QuantizationPlan``.

    Attributes:
        per_layer_precision: Each layer's resolved precision, as
            ``Precision.value`` strings, keyed by layer name.
        precision_distribution: The count of layers assigned each
            precision value.
        selected_strategy_name: The ``IQuantizationStrategy.name()``
            actually invoked.
        rationale: The per-layer rationale, keyed by layer name.
    """

    per_layer_precision: Dict[str, str] = field(default_factory=dict)
    precision_distribution: Dict[str, int] = field(default_factory=dict)
    selected_strategy_name: str = ""
    rationale: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return this summary as a plain, JSON-serializable ``dict``."""
        return {
            "per_layer_precision": dict(self.per_layer_precision),
            "precision_distribution": dict(self.precision_distribution),
            "selected_strategy_name": self.selected_strategy_name,
            "rationale": dict(self.rationale),
        }


@dataclass(frozen=True)
class CompressionSummary:
    """A flattened, serialization-friendly view of a ``CompressionPlan``.

    Attributes:
        selected_types: The ``CompressionType.value`` strings applied,
            in application order.
        strategy_names: The ``ICompressionStrategy.name()`` resolved
            for every selected type, keyed by that type's ``.value``.
        target_ratio: The compression ratio the plan targeted.
        rationale: The human-readable rationale for the plan.
    """

    selected_types: List[str] = field(default_factory=list)
    strategy_names: Dict[str, str] = field(default_factory=dict)
    target_ratio: float = 1.0
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return this summary as a plain, JSON-serializable ``dict``."""
        return {
            "selected_types": list(self.selected_types),
            "strategy_names": dict(self.strategy_names),
            "target_ratio": self.target_ratio,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class OptimizationSummary:
    """A flattened, serialization-friendly view of an ``OptimizationResult``.

    Attributes:
        selected_configuration: The winning candidate's serialized
            summary.
        objective_scores: The winning candidate's per-objective scores.
        candidate_count: The number of candidates evaluated during the
            search.
        pareto_front_size: The number of non-dominated candidates.
        rationale: The human-readable explanation of why the winning
            candidate was selected.
    """

    selected_configuration: Dict[str, Any] = field(default_factory=dict)
    objective_scores: Dict[str, float] = field(default_factory=dict)
    candidate_count: int = 0
    pareto_front_size: int = 0
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return this summary as a plain, JSON-serializable ``dict``."""
        return {
            "selected_configuration": dict(self.selected_configuration),
            "objective_scores": dict(self.objective_scores),
            "candidate_count": self.candidate_count,
            "pareto_front_size": self.pareto_front_size,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class MemoryPlanSummary:
    """A flattened, serialization-friendly view of a ``MemoryPlan``.

    Attributes:
        buffer_arena_bytes: The size of the planned activation arena.
        peak_memory_bytes: The highest simultaneous activation-arena
            usage observed across the plan.
        static_weight_bytes: The total size of every parameter tensor.
    """

    buffer_arena_bytes: int = 0
    peak_memory_bytes: int = 0
    static_weight_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return this summary as a plain, JSON-serializable ``dict``."""
        return {
            "buffer_arena_bytes": self.buffer_arena_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,
            "static_weight_bytes": self.static_weight_bytes,
        }


@dataclass(frozen=True)
class ExportSummary:
    """A flattened, serialization-friendly view of a ``DeploymentArtifact``.

    Attributes:
        file_paths: Every artifact file path the exporter backend
            wrote.
        export_format: The ``ExportFormat.value`` produced.
        target_profile_id: The ``HardwareProfile.profile_id`` this
            artifact was produced for.
        size_bytes: The total size, in bytes, of ``file_paths``.
        manifest_path: The path of the written manifest, if any.
        package_path: The path of the assembled deployment package, if
            packaging was enabled.
    """

    file_paths: List[str] = field(default_factory=list)
    export_format: str = ""
    target_profile_id: str = ""
    size_bytes: int = 0
    manifest_path: str = ""
    package_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return this summary as a plain, JSON-serializable ``dict``."""
        return {
            "file_paths": list(self.file_paths),
            "export_format": self.export_format,
            "target_profile_id": self.target_profile_id,
            "size_bytes": self.size_bytes,
            "manifest_path": self.manifest_path,
            "package_path": self.package_path,
        }


@dataclass(frozen=True)
class ReportResult:
    """The complete, flattened summary of one pipeline run, assembled
    from every stage's ``StageResult`` recorded in ``PipelineContext``.

    Attributes:
        run_id: The run this result summarizes.
        generated_at: The UTC ISO-8601 timestamp this result was built
            at.
        overall_success: ``False`` if any recorded stage's
            ``StageResult.success`` is ``False``, or if evaluation ran
            and ``EvaluationResult.passed`` is ``False``; ``True``
            otherwise.
        model: The analyzed model's summary, or ``None`` if
            ``\"model_analyzer\"`` has not run.
        hardware: The resolved deployment target's summary, or ``None``
            if ``\"hardware_manager\"`` has not run.
        quantization: The applied quantization plan's summary, or
            ``None`` if ``\"quantization_planner\"`` has not run.
        compression: The applied compression plan's summary, or
            ``None`` if ``\"compression_planner\"`` has not run.
        optimization: The structural optimization search's summary, or
            ``None`` if ``\"optimizer\"`` has not run.
        memory_plan: The activation-arena memory plan's summary, or
            ``None`` if ``\"memory_optimizer\"`` has not run.
        export: The produced deployment artifact's summary, or ``None``
            if ``\"exporter\"`` has not run.
        evaluation: The full ``EvaluationResult``, or ``None`` if
            ``\"evaluator\"`` has not run.
        benchmark: The full ``BenchmarkResult``, or ``None`` if
            ``\"benchmarker\"`` has not run.
        stage_summaries: One :class:`StageExecutionSummary` per stage
            found in ``PipelineContext``, in canonical pipeline order.
        warnings: Every stage's ``StageResult.warnings`` entries,
            prefixed with the owning stage name, plus any warning this
            module itself raised while assembling the result (e.g. a
            missing upstream stage).
    """

    run_id: str = _UNKNOWN_RUN_ID
    generated_at: str = ""
    overall_success: bool = True
    model: Optional[ModelSummary] = None
    hardware: Optional[HardwareSummary] = None
    quantization: Optional[QuantizationSummary] = None
    compression: Optional[CompressionSummary] = None
    optimization: Optional[OptimizationSummary] = None
    memory_plan: Optional[MemoryPlanSummary] = None
    export: Optional[ExportSummary] = None
    evaluation: Optional[EvaluationResult] = None
    benchmark: Optional[BenchmarkResult] = None
    stage_summaries: List[StageExecutionSummary] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return this result as a plain, JSON-serializable ``dict``."""
        return {
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "overall_success": self.overall_success,
            "model": self.model.to_dict() if self.model is not None else None,
            "hardware": self.hardware.to_dict() if self.hardware is not None else None,
            "quantization": (
                self.quantization.to_dict() if self.quantization is not None else None
            ),
            "compression": (
                self.compression.to_dict() if self.compression is not None else None
            ),
            "optimization": (
                self.optimization.to_dict() if self.optimization is not None else None
            ),
            "memory_plan": (
                self.memory_plan.to_dict() if self.memory_plan is not None else None
            ),
            "export": self.export.to_dict() if self.export is not None else None,
            "evaluation": (
                self.evaluation.to_dict() if self.evaluation is not None else None
            ),
            "benchmark": (
                self.benchmark.to_dict() if self.benchmark is not None else None
            ),
            "stage_summaries": [s.to_dict() for s in self.stage_summaries],
            "warnings": list(self.warnings),
        }


def resolve_run_id(context: PipelineContext) -> str:
    """Best-effort resolution of ``context``'s run identifier.

    ``PipelineContext`` (``uaqe.domain.pipeline_context``) carries its
    run id as a constructor-set attribute but exposes no locked public
    accessor for it (``03_API_Specification.md`` §2.1 lists only
    ``append``/``get``/``has``). ``PipelineOrchestrator`` itself reads
    this same attribute directly (see its ``# noqa: SLF001`` accesses),
    so this module follows that established, already-reviewed
    precedent rather than requiring an unrelated API change to
    ``PipelineContext`` just to plumb a run id through to reporting.

    Args:
        context: The pipeline context to resolve a run id from.

    Returns:
        The resolved run id, or :data:`_UNKNOWN_RUN_ID` if ``context``
        exposes no usable identifier.
    """
    run_id = getattr(context, "_run_id", None)  # noqa: SLF001 - see docstring
    if isinstance(run_id, str) and run_id:
        return run_id
    return _UNKNOWN_RUN_ID


def _unwrap(payload: Any) -> Any:
    """Return the leading element of a ``(Plan, IMR)``-shaped payload.

    Several stages (``quantization_planner``, ``compression_planner``,
    ``optimizer``, ``memory_optimizer``, ``exporter``) record a
    ``(domain_object, IMR)`` tuple as their ``StageResult.payload``,
    per each package's own locked-resolution precedent (see e.g.
    :mod:`uaqe.quantization.quantization_planner`'s module docstring).
    This module only ever needs the domain object half.

    Args:
        payload: The raw ``StageResult.payload`` value.

    Returns:
        ``payload[0]`` if ``payload`` is a tuple, else ``payload``
        itself unchanged.
    """
    return payload[0] if isinstance(payload, tuple) else payload


def build_report_result(
    context: PipelineContext,
    run_id: Optional[str] = None,
    *,
    logger: Optional[ILogger] = None,
) -> ReportResult:
    """Assemble a :class:`ReportResult` from ``context``.

    Every field is extracted defensively: a stage that has not (yet)
    run simply leaves the corresponding field ``None`` rather than
    raising, so this function is safe to call at any point in a run,
    not only after every stage has completed.

    Args:
        context: The pipeline context to summarize.
        run_id: The run identifier to record. Resolved from ``context``
            via :func:`resolve_run_id` when omitted.
        logger: Optional structured logging sink; if omitted, this
            function operates silently.

    Returns:
        The assembled :class:`ReportResult`.
    """
    resolved_run_id = run_id or resolve_run_id(context)
    warnings: List[str] = []
    stage_summaries: List[StageExecutionSummary] = []

    for stage_name in _KNOWN_STAGE_NAMES:
        if not context.has(stage_name):
            continue
        stage_result = context.get(stage_name)
        stage_summaries.append(
            StageExecutionSummary(
                stage_name=stage_name,
                success=stage_result.success,
                duration_ms=stage_result.duration_ms,
                warning_count=len(stage_result.warnings),
            )
        )
        warnings.extend(f"[{stage_name}] {w}" for w in stage_result.warnings)

    model: Optional[ModelSummary] = None
    if context.has(MODEL_ANALYZER_STAGE_NAME):
        analysis: AnalysisResult = context.get(MODEL_ANALYZER_STAGE_NAME).payload
        model = ModelSummary(
            parameter_count=analysis.parameter_count,
            estimated_flops=analysis.estimated_flops,
            estimated_memory_bytes=analysis.estimated_memory_bytes,
            op_type_histogram=dict(analysis.op_type_histogram),
        )
    else:
        warnings.append(
            f"[{MODEL_ANALYZER_STAGE_NAME}] stage has not run; model summary omitted."
        )

    hardware: Optional[HardwareSummary] = None
    if context.has(HARDWARE_MANAGER_STAGE_NAME):
        profile: HardwareProfile = context.get(HARDWARE_MANAGER_STAGE_NAME).payload
        hardware = HardwareSummary(
            profile_id=profile.profile_id,
            display_name=profile.display_name,
            hardware_class=profile.hardware_class.value,
            runtime=profile.runtime,
            ram_bytes=profile.ram_bytes,
            tensor_memory_bytes=profile.tensor_memory_bytes,
            max_model_size_bytes=profile.max_model_size_bytes,
        )
    else:
        warnings.append(
            f"[{HARDWARE_MANAGER_STAGE_NAME}] stage has not run; hardware summary omitted."
        )

    quantization: Optional[QuantizationSummary] = None
    if context.has(QUANTIZATION_PLANNER_STAGE_NAME):
        plan: QuantizationPlan = _unwrap(
            context.get(QUANTIZATION_PLANNER_STAGE_NAME).payload
        )
        distribution: Dict[str, int] = {}
        per_layer: Dict[str, str] = {}
        for layer_name, precision in plan.per_layer_precision.items():
            per_layer[layer_name] = precision.value
            distribution[precision.value] = distribution.get(precision.value, 0) + 1
        quantization = QuantizationSummary(
            per_layer_precision=per_layer,
            precision_distribution=distribution,
            selected_strategy_name=plan.selected_strategy_name,
            rationale=dict(plan.rationale),
        )

    compression: Optional[CompressionSummary] = None
    if context.has(COMPRESSION_PLANNER_STAGE_NAME):
        plan = _unwrap(context.get(COMPRESSION_PLANNER_STAGE_NAME).payload)
        compression = CompressionSummary(
            selected_types=[t.value for t in plan.selected_types],
            strategy_names={t.value: name for t, name in plan.strategy_names.items()},
            target_ratio=plan.target_ratio,
            rationale=plan.rationale,
        )

    optimization: Optional[OptimizationSummary] = None
    if context.has(OPTIMIZER_STAGE_NAME):
        result: OptimizationResult = _unwrap(
            context.get(OPTIMIZER_STAGE_NAME).payload
        )
        optimization = OptimizationSummary(
            selected_configuration=dict(result.selected_configuration),
            objective_scores=dict(result.objective_scores),
            candidate_count=len(result.candidates),
            pareto_front_size=len(result.pareto_front),
            rationale=result.rationale,
        )

    memory_plan: Optional[MemoryPlanSummary] = None
    if context.has(MEMORY_OPTIMIZER_STAGE_NAME):
        plan_obj: MemoryPlan = _unwrap(
            context.get(MEMORY_OPTIMIZER_STAGE_NAME).payload
        )
        memory_plan = MemoryPlanSummary(
            buffer_arena_bytes=plan_obj.buffer_arena_bytes,
            peak_memory_bytes=plan_obj.peak_memory_bytes,
            static_weight_bytes=plan_obj.static_weight_bytes,
        )

    export: Optional[ExportSummary] = None
    if context.has(EXPORTER_STAGE_NAME):
        artifact: "DeploymentArtifact" = _unwrap(
            context.get(EXPORTER_STAGE_NAME).payload
        )
        export = ExportSummary(
            file_paths=list(artifact.file_paths),
            export_format=artifact.export_format.value,
            target_profile_id=artifact.target_profile_id,
            size_bytes=artifact.size_bytes,
            manifest_path=artifact.manifest_path,
            package_path=artifact.package_path,
        )

    evaluation: Optional[EvaluationResult] = None
    if context.has(EVALUATOR_STAGE_NAME):
        evaluation = context.get(EVALUATOR_STAGE_NAME).payload

    benchmark: Optional[BenchmarkResult] = None
    if context.has(BENCHMARKER_STAGE_NAME):
        benchmark = context.get(BENCHMARKER_STAGE_NAME).payload

    overall_success = all(summary.success for summary in stage_summaries)
    if evaluation is not None and not evaluation.passed:
        overall_success = False

    result = ReportResult(
        run_id=resolved_run_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        overall_success=overall_success,
        model=model,
        hardware=hardware,
        quantization=quantization,
        compression=compression,
        optimization=optimization,
        memory_plan=memory_plan,
        export=export,
        evaluation=evaluation,
        benchmark=benchmark,
        stage_summaries=stage_summaries,
        warnings=warnings,
    )

    if logger is not None:
        logger.info(
            "Report result assembled.",
            run_id=resolved_run_id,
            overall_success=overall_success,
            stage_count=len(stage_summaries),
        )

    return result
