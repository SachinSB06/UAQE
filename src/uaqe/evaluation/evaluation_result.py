"""Result dataclasses produced by a completed evaluation run.

Per ``03_API_Specification.md`` §10.1, the locked ``EvaluationResult``
shape is exactly ``baseline_accuracy: float``, ``optimized_accuracy:
float``, ``accuracy_delta: float``, ``per_layer_error: Dict[str,
float]``. This module keeps those four fields and adds one sub-score
dataclass per evaluation dimension (latency, memory, power, size,
compatibility, regression) as a strict superset — the same
locked-resolution precedent :class:`~uaqe.optimizer.optimization_result.
OptimizationResult` and :class:`~uaqe.compression.compression_planner.
CompressionPlan` already establish for their own locked counterparts.

Every sub-score is ``Optional`` and defaults to ``None``: a caller who
only has a baseline/optimized ``IMR`` pair (no resolved
``HardwareProfile``, no ``DeploymentArtifact``, no prior-run baseline
metrics) still gets a fully populated accuracy score, with every other
dimension simply absent rather than the whole evaluation failing — see
:mod:`uaqe.evaluation.evaluation_planner` for how the set of dimensions
actually run for a given input is decided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AccuracyScore:
    """The outcome of :class:`~uaqe.evaluation.accuracy_evaluator.
    AccuracyEvaluator.evaluate`.

    Attributes:
        baseline_accuracy: A normalized ``[0, 1]`` accuracy-retention
            figure for the pre-optimization ``IMR``, always ``1.0`` by
            construction (see the module docstring of
            :mod:`uaqe.evaluation.accuracy_evaluator` for why the
            baseline is itself the reference point rather than a
            dataset-measured figure).
        optimized_accuracy: The same figure for the post-optimization
            ``IMR``, ``<= baseline_accuracy`` whenever optimization has
            introduced any precision loss.
        accuracy_delta: ``optimized_accuracy - baseline_accuracy``;
            negative values indicate an accuracy regression.
        per_layer_error: The estimated per-layer accuracy-retention
            loss, keyed by ``IMRLayer.name``, for every layer whose
            precision changed between the baseline and optimized IMR.
        sample_count: The number of evaluation-dataset samples folded
            into this score (``0`` if no ``dataset_path`` was supplied,
            in which case the score is derived from static per-layer
            precision/parameter analysis alone).
        assumptions: Human-readable notes on the estimation model used
            and its known limitations.
    """

    baseline_accuracy: float = 1.0
    optimized_accuracy: float = 1.0
    accuracy_delta: float = 0.0
    per_layer_error: Dict[str, float] = field(default_factory=dict)
    sample_count: int = 0
    assumptions: List[str] = field(default_factory=list)


@dataclass
class LatencyScore:
    """The outcome of :class:`~uaqe.evaluation.latency_evaluator.
    LatencyEvaluator.evaluate`.

    Attributes:
        baseline_latency_ms: The pre-optimization ``IMR``'s estimated
            total inference latency, from
            :class:`~uaqe.hardware.latency_estimator.LatencyEstimator`.
        optimized_latency_ms: The post-optimization ``IMR``'s estimated
            total inference latency.
        latency_delta_ms: ``optimized_latency_ms - baseline_latency_ms``;
            negative values indicate a speedup.
        speedup_ratio: ``baseline_latency_ms / optimized_latency_ms``;
            ``1.0`` means no change, ``> 1.0`` means faster.
    """

    baseline_latency_ms: float = 0.0
    optimized_latency_ms: float = 0.0
    latency_delta_ms: float = 0.0
    speedup_ratio: float = 1.0


@dataclass
class MemoryScore:
    """The outcome of :class:`~uaqe.evaluation.memory_evaluator.
    MemoryEvaluator.evaluate`.

    Attributes:
        baseline_peak_memory_bytes: The pre-optimization ``IMR``'s
            planned peak activation-arena usage, from
            :class:`~uaqe.hardware.memory_planner.MemoryPlanner`.
        optimized_peak_memory_bytes: The post-optimization ``IMR``'s
            planned peak activation-arena usage.
        memory_delta_bytes: ``optimized_peak_memory_bytes -
            baseline_peak_memory_bytes``; negative values indicate a
            reduction.
        reduction_ratio: ``baseline_peak_memory_bytes /
            optimized_peak_memory_bytes``; ``1.0`` means no change,
            ``> 1.0`` means less memory used.
        baseline_static_weight_bytes: The pre-optimization ``IMR``'s
            total static parameter-tensor size, for context.
        optimized_static_weight_bytes: The post-optimization ``IMR``'s
            total static parameter-tensor size, for context.
    """

    baseline_peak_memory_bytes: int = 0
    optimized_peak_memory_bytes: int = 0
    memory_delta_bytes: int = 0
    reduction_ratio: float = 1.0
    baseline_static_weight_bytes: int = 0
    optimized_static_weight_bytes: int = 0


@dataclass
class PowerScore:
    """The outcome of :class:`~uaqe.evaluation.power_evaluator.
    PowerEvaluator.evaluate`.

    Attributes:
        baseline_average_power_mw: The pre-optimization ``IMR``'s
            estimated average active-mode power draw, from
            :class:`~uaqe.hardware.power_estimator.PowerEstimator`.
        optimized_average_power_mw: The post-optimization ``IMR``'s
            estimated average active-mode power draw.
        baseline_energy_per_inference_mj: The pre-optimization ``IMR``'s
            estimated energy consumed per inference.
        optimized_energy_per_inference_mj: The post-optimization
            ``IMR``'s estimated energy consumed per inference.
        energy_reduction_ratio: ``baseline_energy_per_inference_mj /
            optimized_energy_per_inference_mj``; ``1.0`` means no
            change, ``> 1.0`` means less energy used per inference.
    """

    baseline_average_power_mw: float = 0.0
    optimized_average_power_mw: float = 0.0
    baseline_energy_per_inference_mj: float = 0.0
    optimized_energy_per_inference_mj: float = 0.0
    energy_reduction_ratio: float = 1.0


@dataclass
class SizeScore:
    """The outcome of :class:`~uaqe.evaluation.size_evaluator.
    SizeEvaluator.evaluate`.

    Attributes:
        baseline_size_bytes: The pre-optimization ``IMR``'s estimated
            serialized size.
        optimized_size_bytes: The post-optimization ``IMR``'s size —
            the actual ``DeploymentArtifact.size_bytes`` when the
            ``Exporter`` stage has already run this run, otherwise the
            same static estimation model used for
            ``baseline_size_bytes``.
        size_delta_bytes: ``optimized_size_bytes - baseline_size_bytes``;
            negative values indicate a reduction.
        compression_ratio: ``baseline_size_bytes / optimized_size_bytes``;
            ``1.0`` means no change, ``> 1.0`` means smaller.
        used_exported_artifact: Whether ``optimized_size_bytes`` came
            from an actual ``DeploymentArtifact`` (``True``) rather than
            the static estimation model (``False``).
    """

    baseline_size_bytes: int = 0
    optimized_size_bytes: int = 0
    size_delta_bytes: int = 0
    compression_ratio: float = 1.0
    used_exported_artifact: bool = False


@dataclass
class CompatibilityScore:
    """The outcome of :class:`~uaqe.evaluation.compatibility_evaluator.
    CompatibilityEvaluator.evaluate`.

    Attributes:
        compatible: Whether the final, post-optimization ``IMR`` is
            still fully compatible with the resolved ``HardwareProfile``
            at evaluation time — a final confirmation re-check, since
            ``09_Architecture_Lock.md`` §12 places this package after
            ``Exporter`` in pipeline order.
        unsupported_layers: Names of any layer that cannot run on the
            target, per ``CompatibilityReport.unsupported_layers``.
        constraint_violations: Human-readable non-layer-specific
            constraint violations, per
            ``CompatibilityReport.constraint_violations``.
    """

    compatible: bool = True
    unsupported_layers: List[str] = field(default_factory=list)
    constraint_violations: List[str] = field(default_factory=list)


@dataclass
class RegressionFinding:
    """One metric's comparison against a prior-run baseline value,
    produced by :class:`~uaqe.evaluation.regression_evaluator.
    RegressionEvaluator.check`.

    Attributes:
        metric_name: The evaluated metric's identifier, e.g.
            ``"accuracy_delta"``, ``"optimized_latency_ms"``.
        baseline_value: The value recorded for this metric in the
            supplied prior-run baseline.
        current_value: The value observed for this metric in the
            current run.
        threshold: The maximum acceptable relative change (a fraction,
            e.g. ``0.05`` for 5%) before this metric is flagged as
            regressed.
        regressed: Whether ``current_value`` exceeds ``threshold``'s
            allowance relative to ``baseline_value``, in the direction
            that is unfavorable for this metric (see
            :mod:`uaqe.evaluation.regression_evaluator` for the
            per-metric favorable-direction table).
        message: A human-readable explanation of the comparison.
    """

    metric_name: str
    baseline_value: float = 0.0
    current_value: float = 0.0
    threshold: float = 0.0
    regressed: bool = False
    message: str = ""


@dataclass
class EvaluationResult:
    """The complete outcome of one evaluation run.

    Attributes:
        baseline_accuracy: The locked minimal field
            (``03_API_Specification.md`` §10.1), mirrored from
            ``accuracy.baseline_accuracy``.
        optimized_accuracy: The locked minimal field, mirrored from
            ``accuracy.optimized_accuracy``.
        accuracy_delta: The locked minimal field, mirrored from
            ``accuracy.accuracy_delta``.
        per_layer_error: The locked minimal field, mirrored from
            ``accuracy.per_layer_error``.
        accuracy: The full :class:`AccuracyScore`, always populated.
        latency: The full :class:`LatencyScore`, or ``None`` if no
            ``HardwareProfile`` was available to evaluate against.
        memory: The full :class:`MemoryScore`, or ``None`` for the same
            reason as ``latency``.
        power: The full :class:`PowerScore`, or ``None`` for the same
            reason as ``latency``.
        size: The full :class:`SizeScore`, always populated (this
            dimension needs no ``HardwareProfile``).
        compatibility: The full :class:`CompatibilityScore`, or ``None``
            if no ``HardwareProfile`` was available to check against.
        regressions: Every :class:`RegressionFinding` produced by
            :class:`~uaqe.evaluation.regression_evaluator.
            RegressionEvaluator`, empty if no prior-run baseline metrics
            were supplied.
        passed: ``True`` unless at least one entry in ``regressions``
            has ``regressed=True`` or ``compatibility.compatible`` is
            ``False`` when ``compatibility`` is populated.
        warnings: Non-fatal warnings accumulated while building this
            result (e.g. a requested dimension that could not be
            evaluated for lack of an input).
    """

    baseline_accuracy: float = 1.0
    optimized_accuracy: float = 1.0
    accuracy_delta: float = 0.0
    per_layer_error: Dict[str, float] = field(default_factory=dict)
    accuracy: AccuracyScore = field(default_factory=AccuracyScore)
    latency: Optional[LatencyScore] = None
    memory: Optional[MemoryScore] = None
    power: Optional[PowerScore] = None
    size: Optional[SizeScore] = None
    compatibility: Optional[CompatibilityScore] = None
    regressions: List[RegressionFinding] = field(default_factory=list)
    passed: bool = True
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return this result as a plain, JSON-serializable ``dict``."""
        return {
            "baseline_accuracy": self.baseline_accuracy,
            "optimized_accuracy": self.optimized_accuracy,
            "accuracy_delta": self.accuracy_delta,
            "per_layer_error": dict(self.per_layer_error),
            "latency": (
                {
                    "baseline_latency_ms": self.latency.baseline_latency_ms,
                    "optimized_latency_ms": self.latency.optimized_latency_ms,
                    "latency_delta_ms": self.latency.latency_delta_ms,
                    "speedup_ratio": self.latency.speedup_ratio,
                }
                if self.latency is not None
                else None
            ),
            "memory": (
                {
                    "baseline_peak_memory_bytes": self.memory.baseline_peak_memory_bytes,
                    "optimized_peak_memory_bytes": self.memory.optimized_peak_memory_bytes,
                    "memory_delta_bytes": self.memory.memory_delta_bytes,
                    "reduction_ratio": self.memory.reduction_ratio,
                }
                if self.memory is not None
                else None
            ),
            "power": (
                {
                    "baseline_average_power_mw": self.power.baseline_average_power_mw,
                    "optimized_average_power_mw": self.power.optimized_average_power_mw,
                    "energy_reduction_ratio": self.power.energy_reduction_ratio,
                }
                if self.power is not None
                else None
            ),
            "size": (
                {
                    "baseline_size_bytes": self.size.baseline_size_bytes,
                    "optimized_size_bytes": self.size.optimized_size_bytes,
                    "compression_ratio": self.size.compression_ratio,
                    "used_exported_artifact": self.size.used_exported_artifact,
                }
                if self.size is not None
                else None
            ),
            "compatibility": (
                {
                    "compatible": self.compatibility.compatible,
                    "unsupported_layers": list(self.compatibility.unsupported_layers),
                    "constraint_violations": list(
                        self.compatibility.constraint_violations
                    ),
                }
                if self.compatibility is not None
                else None
            ),
            "regressions": [
                {
                    "metric_name": finding.metric_name,
                    "baseline_value": finding.baseline_value,
                    "current_value": finding.current_value,
                    "threshold": finding.threshold,
                    "regressed": finding.regressed,
                    "message": finding.message,
                }
                for finding in self.regressions
            ],
            "passed": self.passed,
            "warnings": list(self.warnings),
        }
