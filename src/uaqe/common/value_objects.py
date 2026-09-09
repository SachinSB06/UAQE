"""Immutable configuration value objects for the Universal AI Quantization
Engine.

Every class in this module is a frozen dataclass supplied to a
``PipelineStage`` (or an ``I*`` strategy/backend) via constructor
injection. Per ``10_Module_Development_Guide.md`` §0 rule 7, no module
reads a config file directly: these objects are always sourced from an
``IConfigRepository`` implementation by the ``CompositionRoot`` and
passed down, never constructed ad hoc inside domain logic.

This module has no dependencies on any other ``uaqe`` package beyond
``uaqe.common`` itself, per the layering rules in
``09_Architecture_Lock.md`` §8.

Locked contract: ``03_API_Specification.md`` §1.3.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from uaqe.common.types import CompressionType, HardwareClass, PipelineErrorPolicy, Precision


@dataclass(frozen=True)
class QuantizationConfig:
    """Configuration governing how ``QuantizationAdvisor`` and
    ``QuantizationEngine`` select and apply numeric precision.

    Attributes:
        default_precision: The precision applied to any layer without a
            more specific override.
        per_layer_overrides: Precision overrides keyed by ``IMRLayer.name``,
            taking priority over ``default_precision`` for the named
            layers.
        calibration_batch_size: The batch size ``CalibrationEngine`` uses
            when computing per-layer activation statistics.
        sensitivity_threshold: The maximum acceptable per-layer accuracy
            delta (as used by ``SensitivityAnalyzer``) before a layer is
            flagged for a higher-precision override.
    """

    default_precision: Precision
    per_layer_overrides: Dict[str, Precision] = field(default_factory=dict)
    calibration_batch_size: int = 1
    sensitivity_threshold: float = 0.0


@dataclass(frozen=True)
class CompressionConfig:
    """Configuration governing how ``CompressionAdvisor`` and
    ``CompressionEngine`` select and apply compression techniques.

    Attributes:
        enabled_types: The set of ``CompressionType`` values available
            for selection.
        target_ratio: The desired overall compression ratio (compressed
            size / original size) the engine should aim for.
        pruning_sparsity: The target fraction of pruned (zeroed) weights
            when ``CompressionType.PRUNING`` is enabled.
    """

    enabled_types: List[CompressionType] = field(default_factory=list)
    target_ratio: float = 1.0
    pruning_sparsity: float = 0.0
    # Experimental pruning options
    experimental_pruning_strategy: str = "magnitude_pruning"
    max_accuracy_drop: float = 0.01
    max_latency_regression: float = 0.05
    fine_tune_enabled: bool = False
    fine_tune_epochs: int = 5
    learning_rate: float = 1e-4
    optimizer: str = "Adam"
    scheduler: str = "CosineAnnealing"
    batch_size: int = 16
    weight_decay: float = 1e-4
    early_stopping: bool = True



@dataclass(frozen=True)
class HardwareConfig:
    """Configuration identifying the deployment target.

    Attributes:
        target_profile_id: The ``HardwareProfile.profile_id`` to resolve
            via ``IHardwareProfileRepository.get()``.
        hardware_class: The broad hardware category of the target,
            consumed by ``PipelineBuilder.build()`` to select the
            concrete pipeline stage sequence and backend implementations.
    """

    target_profile_id: str
    hardware_class: HardwareClass


@dataclass(frozen=True)
class OptimizationConfig:
    """Configuration governing ``OptimizationEngine.search()``'s
    multi-objective search.

    Attributes:
        objectives: The names of the objectives to optimize for, e.g.
            ``["accuracy", "latency", "memory"]``.
        objective_weights: The relative weight of each objective, keyed
            by the same names as ``objectives``.
        max_search_iterations: The maximum number of search iterations
            ``OptimizationEngine.search()`` may perform before returning
            its best-known result.
    """

    objectives: List[str] = field(default_factory=list)
    objective_weights: Dict[str, float] = field(default_factory=dict)
    max_search_iterations: int = 1


@dataclass(frozen=True)
class ExecutionConfig:
    """Configuration governing how ``PipelineOrchestrator`` and
    ``PipelineBuilder`` execute a run.

    Attributes:
        parallelism: Per-stage-name flags indicating whether a stage may
            run its internal work concurrently.
        large_model_threshold_mb: The model size, in megabytes, above
            which stages may switch to a large-model execution strategy.
        on_error: The policy ``PipelineOrchestrator.run()``/``resume()``
            applies when a stage raises a ``UAQEError``.
    """

    parallelism: Dict[str, bool] = field(default_factory=dict)
    large_model_threshold_mb: int = 0
    on_error: PipelineErrorPolicy = PipelineErrorPolicy.ABORT
