"""``StageFactory`` — resolves and constructs the ordered
``List[PipelineStage]`` for one run.

Per ``09_Architecture_Lock.md`` §12, "``PipelineBuilder.build()`` is the
sole authority for materializing [the locked 19-stage] order into a
concrete ``List[PipelineStage]``." That class lives at
``uaqe.domain.pipeline.pipeline_builder`` in the locked folder
structure (``02_Folder_Structure.md`` §3). This package places that
same responsibility in the *application* layer instead, as
``StageFactory``, for two reasons specific to this codebase snapshot:

1. Stage *construction* (as opposed to stage *ordering*, which remains
   exactly the locked sequence) is fundamentally a wiring concern —
   every constructor below needs concrete collaborators
   (``IFrameworkAdapter`` instances, an ``IHardwareProfileRepository``,
   strategy maps, ``IExporterBackend``/``IReportRenderer`` instances)
   that only exist once ``uaqe.interface.composition_root`` has
   resolved them from ``uaqe.infrastructure``. Per the locked layering
   (``09_Architecture_Lock.md`` §8), ``application`` (not ``domain``)
   is the layer already permitted to sit adjacent to that composition
   boundary while still depending on interfaces alone — the same
   reasoning already documented for why ``HardwareManager`` accepts an
   injected ``IHardwareProfileRepository`` rather than constructing one
   itself.
2. This snapshot's actual first-party stages (``uaqe.model_loader``,
   ``uaqe.analyzer``, ``uaqe.hardware``, ``uaqe.quantization``,
   ``uaqe.compression``, ``uaqe.optimizer``, ``uaqe.exporter``,
   ``uaqe.evaluation``, ``uaqe.benchmark``, ``uaqe.reports``) are each
   already their own top-level package rather than living under
   ``uaqe.domain.<subpackage>`` as ``02_Folder_Structure.md`` §3
   originally laid out — the same precedent-setting deviation
   ``uaqe.optimizer``'s own ``__init__.py`` documents for itself. This
   factory is the single place that knows how to reach across all ten
   of those packages to assemble one run; no other application file
   imports them directly.

This module also absorbs the two seam-binding responsibilities the
locked design (``03_API_Specification.md`` §14.2, §14.3) assigns to
``ModelIngestionService``/``HardwareSelectionService``: handing
``WorkflowConfig.model_path`` to :class:`~uaqe.model_loader.
model_loader.ModelLoader` via its documented ``bind_source_path`` seam,
and ``WorkflowConfig.hardware_profile_id`` to
:class:`~uaqe.hardware.hardware_manager.HardwareManager`'s constructor
— both are one-line, no-branching-logic seams that do not warrant a
separate file in this package's ten-file decomposition.

Per the locked layering (``09_Architecture_Lock.md`` §8:
``application -> domain, common``), every import below is either
``uaqe.common`` (interfaces/value objects), ``uaqe.domain`` (the
abstract ``PipelineStage``), or one of this snapshot's first-party
stage packages listed above — never ``uaqe.infrastructure`` or
``uaqe.interface``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from uaqe.application.application_errors import StageResolutionError
from uaqe.application.workflow_config import WorkflowConfig
from uaqe.common.interfaces.i_compression_strategy import ICompressionStrategy
from uaqe.common.interfaces.i_exporter_backend import IExporterBackend
from uaqe.common.interfaces.i_framework_adapter import IFrameworkAdapter
from uaqe.common.interfaces.i_hardware_profile_repository import (
    IHardwareProfileRepository,
)
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.interfaces.i_quantization_strategy import IQuantizationStrategy
from uaqe.common.interfaces.i_report_renderer import IReportRenderer
from uaqe.common.value_objects import (
    CompressionConfig,
    OptimizationConfig,
    QuantizationConfig,
)
from uaqe.domain.pipeline_stage import PipelineStage
from uaqe.analyzer.model_analyzer import ModelAnalyzer
from uaqe.benchmark.benchmark import Benchmarker
from uaqe.benchmark.benchmark_planner import BenchmarkConfig
from uaqe.compression.compression_planner import CompressionPlanner
from uaqe.evaluation.evaluation_planner import EvaluationConfig
from uaqe.evaluation.evaluator import Evaluator
from uaqe.exporter.exporter import Exporter
from uaqe.hardware.hardware_manager import HardwareManager
from uaqe.model_loader.model_loader import ModelLoader
from uaqe.model_loader.model_validator import ModelValidator
from uaqe.optimizer.memory_optimizer import MemoryOptimizer
from uaqe.optimizer.optimizer import Optimizer
from uaqe.quantization.calibrator import Calibrator
from uaqe.quantization.precision_recommender import PrecisionRecommender
from uaqe.quantization.quantization_planner import QuantizationPlanner
from uaqe.quantization.sensitivity_analyzer import SensitivityAnalyzer
from uaqe.reports.report_generator import ReportGenerator
from uaqe.reports.report_planner import ReportConfig

#: This snapshot's implemented subset of ``09_Architecture_Lock.md``
#: §12's locked 19-stage canonical order, in the same relative order.
#: ``layer_compatibility_checker``, ``quantization_advisor``,
#: ``compression_advisor``, ``deployment_readiness_scorer``, and
#: ``optimization_advisor`` have no ``PipelineStage`` implementation on
#: disk in this snapshot and are therefore omitted here rather than
#: silently reordered — see :meth:`StageFactory.build_stage_sequence`'s
#: docstring for why omission (not a raised error) is this factory's
#: chosen handling. Entries are each stage's ``PipelineStage.name()``
#: value (the auto-derived ``snake_case`` of its class name), which is
#: also the ``PipelineContext`` key every downstream stage reads it
#: back under.
IMPLEMENTED_STAGE_ORDER: tuple = (
    "model_loader",
    "model_validator",
    "model_analyzer",
    "hardware_manager",
    "calibrator",
    "sensitivity_analyzer",
    "quantization_planner",
    "compression_planner",
    "optimizer",
    "memory_optimizer",
    "exporter",
    "evaluator",
    "benchmarker",
    "report_generator",
)


@dataclass
class StageFactoryDependencies:
    """Every collaborator :class:`StageFactory` needs to construct one
    run's stages, gathered into a single constructor-injected bundle.

    All of these are ordinarily resolved from ``uaqe.infrastructure``
    by ``uaqe.interface.composition_root`` and passed down as their
    ``uaqe.common.interfaces`` port types (or the frozen ``*Config``
    value objects an ``IConfigRepository`` implementation returns) —
    this factory never constructs a concrete infrastructure class
    itself (``11_Implementation_Rules.md`` §3.4/§3.6).

    Attributes:
        framework_adapters: Every ``IFrameworkAdapter`` available to
            :class:`~uaqe.model_loader.model_loader.ModelLoader`.
        hardware_repository: Resolves the run's ``HardwareProfile`` by
            id for :class:`~uaqe.hardware.hardware_manager.
            HardwareManager`.
        quantization_config: The run's ``QuantizationConfig``, sourced
            from ``IConfigRepository`` (with
            :meth:`~uaqe.application.workflow_config.WorkflowConfig.
            override` already applied by the caller).
        quantization_strategies: Every registered
            ``IQuantizationStrategy``, keyed by
            ``IQuantizationStrategy.name()``, for
            :class:`~uaqe.quantization.quantization_planner.
            QuantizationPlanner`.
        compression_config: The run's ``CompressionConfig``.
        compression_strategies: Every registered
            ``ICompressionStrategy``, keyed by
            ``ICompressionStrategy.name()``, for
            :class:`~uaqe.compression.compression_planner.
            CompressionPlanner`.
        optimization_config: The run's ``OptimizationConfig``, for
            :class:`~uaqe.optimizer.optimizer.Optimizer`.
        exporter_backends: Every registered ``IExporterBackend`` for
            :class:`~uaqe.exporter.exporter.Exporter`.
        report_renderers: Every registered ``IReportRenderer`` for
            :class:`~uaqe.reports.report_generator.ReportGenerator`.
        evaluation_config: Optional ``EvaluationConfig``; ``None``
            lets :class:`~uaqe.evaluation.evaluator.Evaluator` apply
            its own default (the conditional-construction rule in
            ``11_Implementation_Rules.md`` §3.4 binds *this factory*,
            not the already-implemented stage classes it merely
            forwards optional config through to).
        benchmark_config: Optional ``BenchmarkConfig``; ``None`` lets
            :class:`~uaqe.benchmark.benchmark.Benchmarker` apply its
            own default.
        report_config: Optional ``ReportConfig``; ``None`` lets
            :class:`~uaqe.reports.report_generator.ReportGenerator`
            apply its own default.
    """

    framework_adapters: List[IFrameworkAdapter]
    hardware_repository: IHardwareProfileRepository
    quantization_config: QuantizationConfig
    quantization_strategies: Dict[str, IQuantizationStrategy]
    compression_config: CompressionConfig
    compression_strategies: Dict[str, ICompressionStrategy]
    optimization_config: OptimizationConfig
    exporter_backends: List[IExporterBackend]
    report_renderers: List[IReportRenderer]
    evaluation_config: Optional[EvaluationConfig] = None
    benchmark_config: Optional[BenchmarkConfig] = None
    report_config: Optional[ReportConfig] = None


class StageFactory:
    """Constructs one run's ordered ``List[PipelineStage]``.

    Attributes:
        _logger: Structured logging sink, threaded into every
            constructed stage.
        _dependencies: The collaborator bundle every stage constructor
            below draws from.
    """

    def __init__(
        self, logger: ILogger, dependencies: StageFactoryDependencies
    ) -> None:
        """Initialize the factory for a single run.

        Args:
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
            dependencies: The collaborator bundle every stage
                constructor draws from.
        """
        self._logger: ILogger = logger
        self._dependencies: StageFactoryDependencies = dependencies

    def build_stage_sequence(self, config: WorkflowConfig) -> List[PipelineStage]:
        """Build this run's complete, ordered stage sequence.

        Constructs every stage named in :data:`IMPLEMENTED_STAGE_ORDER`
        via :meth:`create_stage`, then performs the two seam bindings
        described in this module's docstring: binding
        ``config.model_path`` to the constructed
        :class:`~uaqe.model_loader.model_loader.ModelLoader`, and
        ``config.hardware_profile_id`` into
        :class:`~uaqe.hardware.hardware_manager.HardwareManager`'s
        constructor.

        Per ``09_Architecture_Lock.md`` §12, "stage order is
        ``HardwareClass``-dependent only insofar as [this factory] may
        select different concrete ``IExporterBackend``/strategy
        implementations per class — the stage *sequence* itself does
        not vary by hardware class"; this method therefore always
        returns the same 14-entry sequence, with hardware-class-specific
        selection happening *inside* :class:`~uaqe.exporter.exporter.
        Exporter`'s own backend resolution, not here.

        Note on completeness: ``09_Architecture_Lock.md`` §12 locks a
        19-stage canonical order; this snapshot has concrete
        ``PipelineStage`` implementations for 14 of those 19 (see
        :data:`IMPLEMENTED_STAGE_ORDER`'s docstring for the five
        omitted names). The omitted stages are silently skipped rather
        than raising, since their absence is a known, already-documented
        gap in this codebase snapshot rather than a caller
        configuration error — raising here would make every run
        unusable until all five land, which is not this factory's call
        to make.

        Args:
            config: The run's validated ``WorkflowConfig``.

        Returns:
            The ordered stages for this run, ready to hand to
            :class:`~uaqe.application.pipeline_executor.
            PipelineExecutor`.

        Raises:
            StageResolutionError: If any named stage cannot be
                constructed (propagated from :meth:`create_stage`).
        """
        stages: List[PipelineStage] = [
            self.create_stage(name, config) for name in IMPLEMENTED_STAGE_ORDER
        ]

        for stage in stages:
            if isinstance(stage, ModelLoader):
                stage.bind_source_path(config.model_path)
            elif isinstance(stage, Calibrator):
                stage.bind_model_path(config.model_path)
                stage.bind_calibration_dataset_path(config.calibration_dataset_path)

        self._logger.info(
            "Stage sequence built.",
            stage_count=len(stages),
            stage_names=[stage.name() for stage in stages],
        )
        return stages

    def create_stage(self, name: str, config: WorkflowConfig) -> PipelineStage:
        """Construct a single stage by its registered name.

        Args:
            name: One of the entries in :data:`IMPLEMENTED_STAGE_ORDER`
                (equivalently, the value a constructed instance's
                ``PipelineStage.name()`` returns).
            config: The run's ``WorkflowConfig`` — only consulted here
                for ``hardware_profile_id``, needed at construction
                time by ``hardware_manager`` (unlike ``model_loader``,
                whose ``model_path`` is bound post-construction via its
                ``bind_source_path`` seam, per
                :meth:`build_stage_sequence`).

        Returns:
            The freshly constructed stage.

        Raises:
            StageResolutionError: If ``name`` is not a recognized
                stage, or a required collaborator for it was not
                supplied via this factory's ``StageFactoryDependencies``.
        """
        deps = self._dependencies
        
        # Run-isolated deep-copy and config overrides
        import copy
        import dataclasses
        from uaqe.common.types import Precision, CompressionType
        
        quant_config = copy.deepcopy(deps.quantization_config)
        comp_config = copy.deepcopy(deps.compression_config)
        opt_config = copy.deepcopy(deps.optimization_config)
        
        # QuantizationConfig dynamic overrides
        default_precision_val = quant_config.default_precision
        default_precision_str = default_precision_val.value if isinstance(default_precision_val, Precision) else default_precision_val
        default_precision_overridden = config.override("quantization.default_precision", default_precision_str)
        if isinstance(default_precision_overridden, str):
            default_precision = Precision[default_precision_overridden]
        else:
            default_precision = default_precision_overridden

        per_layer_overrides_raw = config.override("quantization.per_layer_overrides", quant_config.per_layer_overrides)
        per_layer_overrides = {}
        for k, v in per_layer_overrides_raw.items():
            if isinstance(v, str):
                per_layer_overrides[k] = Precision[v]
            else:
                per_layer_overrides[k] = v

        quant_config = dataclasses.replace(
            quant_config,
            default_precision=default_precision,
            per_layer_overrides=per_layer_overrides,
            calibration_batch_size=config.override("quantization.calibration_batch_size", quant_config.calibration_batch_size),
            sensitivity_threshold=config.override("quantization.sensitivity_threshold", quant_config.sensitivity_threshold),
        )

        # CompressionConfig dynamic overrides
        enabled_types_val = comp_config.enabled_types
        enabled_types_raw = [t.value if isinstance(t, CompressionType) else t for t in enabled_types_val]
        enabled_types_overridden = config.override("compression.enabled_types", enabled_types_raw)
        enabled_types = []
        for t in enabled_types_overridden:
            if isinstance(t, str):
                enabled_types.append(CompressionType[t])
            else:
                enabled_types.append(t)

        comp_config = dataclasses.replace(
            comp_config,
            enabled_types=enabled_types,
            target_ratio=config.override("compression.target_ratio", comp_config.target_ratio),
            pruning_sparsity=config.override("compression.pruning_sparsity", comp_config.pruning_sparsity),
            experimental_pruning_strategy=config.override("compression.experimental_pruning_strategy", comp_config.experimental_pruning_strategy),
        )

        # Compression strategies overrides (clustering_clusters)
        clustering_clusters = config.override("compression.clustering_clusters", 256)
        comp_strategies = dict(deps.compression_strategies)
        if clustering_clusters != 256 or "weight_clustering" in comp_strategies:
            from uaqe.compression.weight_cluster import WeightClusterCompressor
            comp_strategies["weight_clustering"] = WeightClusterCompressor(self._logger, num_clusters=clustering_clusters)

        builders: Dict[str, Callable[[], PipelineStage]] = {
            "model_loader": lambda: ModelLoader(
                deps.framework_adapters, self._logger
            ),
            "model_validator": lambda: ModelValidator(self._logger),
            "model_analyzer": lambda: ModelAnalyzer(self._logger),
            "hardware_manager": lambda: HardwareManager(
                deps.hardware_repository, self._logger, config.hardware_profile_id
            ),
            "calibrator": lambda: Calibrator(
                self._logger,
                model_path=config.model_path,
                calibration_dataset_path=config.calibration_dataset_path,
            ),
            "sensitivity_analyzer": lambda: SensitivityAnalyzer(
                self._logger, quant_config.sensitivity_threshold
            ),
            "quantization_planner": lambda: QuantizationPlanner(
                self._logger,
                PrecisionRecommender(self._logger),
                quant_config,
                deps.quantization_strategies,
            ),
            "compression_planner": lambda: CompressionPlanner(
                self._logger, comp_config, comp_strategies,
                calibration_dataset_path=config.calibration_dataset_path,
                model_path=config.model_path
            ),
            "optimizer": lambda: Optimizer(self._logger, opt_config),
            "memory_optimizer": lambda: MemoryOptimizer(self._logger),
            "exporter": lambda: Exporter(
                deps.exporter_backends,
                self._logger,
                requested_runtime=config.runtime,
                model_path=config.model_path,
                calibration_dataset_path=config.calibration_dataset_path,
                output_dir=config.override("exporter.output_dir", "outputs/exports"),
            ),
            "evaluator": lambda: Evaluator(self._logger, deps.evaluation_config),
            "benchmarker": lambda: Benchmarker(self._logger, deps.benchmark_config),
            "report_generator": lambda: ReportGenerator(
                deps.report_renderers, self._logger, config=deps.report_config
            ),
        }

        build = builders.get(name)
        if build is None:
            raise StageResolutionError(
                f"Unknown stage name {name!r}; not present in "
                "IMPLEMENTED_STAGE_ORDER.",
                code="STAGE_NAME_UNRECOGNIZED",
                remediation_hint=(
                    "Use one of: " + ", ".join(IMPLEMENTED_STAGE_ORDER)
                ),
            )

        try:
            return build()
        except TypeError as exc:
            raise StageResolutionError(
                f"Stage {name!r} could not be constructed: a required "
                "collaborator was missing or malformed in "
                "StageFactoryDependencies.",
                code="STAGE_DEPENDENCY_MISSING",
                remediation_hint=(
                    "Verify every field StageFactoryDependencies requires "
                    f"for {name!r} was supplied by CompositionRoot."
                ),
            ) from exc
