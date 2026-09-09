"""Result and report dataclasses shared across the Universal AI
Quantization Engine.

These are the value objects every ``PipelineStage.execute()`` returns
(``StageResult``), every ``PipelineOrchestrator.run()``/``resume()``
returns (``RunResult``), and the compatibility summary produced by
``HardwareManager``/``LayerCompatibilityChecker`` (``CompatibilityReport``).

This module has no dependencies on any other ``uaqe`` package beyond
``uaqe.common`` itself, per the layering rules in
``09_Architecture_Lock.md`` §8.

Locked contract: ``03_API_Specification.md`` §1.5.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from uaqe.common.exceptions import UAQEError


@dataclass
class StageResult:
    """The outcome of a single ``PipelineStage.execute()`` call.

    Per ``03_API_Specification.md`` §22 rule 3, every stage returns a
    ``StageResult`` — never ``None`` and never a raw domain object.

    Attributes:
        stage_name: The name of the stage that produced this result,
            matching ``PipelineStage.name()``.
        success: Whether the stage completed without a stage-halting
            error.
        payload: The stage's produced value object, per the locked
            mapping in ``09_Architecture_Lock.md`` §9.
        warnings: Non-fatal warning messages accumulated during
            execution.
        duration_ms: Wall-clock time spent in ``execute()``, in
            milliseconds.
    """

    stage_name: str
    success: bool
    payload: Any
    warnings: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class RunResult:
    """The outcome of a complete (or partially complete)
    ``PipelineOrchestrator`` run.

    Attributes:
        run_id: The unique identifier of the run, matching
            ``PipelineContext``'s ``run_id``.
        success: Whether every stage in the run completed successfully
            under the run's ``PipelineErrorPolicy``.
        stage_results: Every ``StageResult`` produced during the run,
            keyed by stage name.
        errors: Every ``UAQEError`` raised during the run.
        output_paths: Filesystem paths written to ``outputs/`` or
            ``reports/`` during the run.
    """

    run_id: str
    success: bool
    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    errors: List[UAQEError] = field(default_factory=list)
    output_paths: List[str] = field(default_factory=list)


@dataclass
class CompatibilityReport:
    """The outcome of a hardware/layer compatibility check.

    Attributes:
        compatible: Whether the checked ``IMR`` is fully compatible with
            the checked ``HardwareProfile``.
        unsupported_layers: Names of ``IMRLayer`` instances that cannot
            be executed on the target hardware.
        constraint_violations: Human-readable descriptions of any
            non-layer-specific constraint violations (e.g. model size
            exceeding available flash).
    """

    compatible: bool
    unsupported_layers: List[str] = field(default_factory=list)
    constraint_violations: List[str] = field(default_factory=list)
