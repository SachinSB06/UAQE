"""Decides which evaluation dimensions a run performs, and the
configuration those dimensions run under.

``EvaluationPlanner`` is an internal collaborator of
:class:`~uaqe.evaluation.evaluator.Evaluator` — the same role
:class:`~uaqe.optimizer.optimization_planner.OptimizationPlanner` plays
for :class:`~uaqe.optimizer.optimizer.Optimizer`: it does not itself
read/write ``PipelineContext`` or return a ``StageResult``; ``Evaluator``
owns that. Its one job is :meth:`build_plan`, which looks at which
optional inputs are actually available for a given run (a resolved
``HardwareProfile``, a ``DeploymentArtifact``, prior-run baseline
metrics) and decides which of the seven evaluation dimensions
(accuracy, latency, memory, power, size, compatibility, regression) can
run — accuracy and size never need a ``HardwareProfile`` and so always
run; latency/memory/power/compatibility need one and are skipped
(with a recorded rationale) when none is available; regression needs
prior-run baseline metrics and is skipped without them.

``EvaluationConfig`` is this package's own configuration value object.
Unlike ``QuantizationConfig``/``CompressionConfig``/``OptimizationConfig``
(``uaqe.common.value_objects``, each locked in
``03_API_Specification.md`` §1.3), no ``EvaluationConfig`` is named in
that locked inventory — the locked ``Evaluator.evaluate`` signature
(``03_API_Specification.md`` §10.1) takes its one input,
``dataset_path``, directly as a parameter rather than via a config
object. This module defines ``EvaluationConfig`` as an *additive*
value object (``09_Architecture_Lock.md`` §0 rule 2: "a new file/class
... that follows an existing pattern, with zero modification to
existing names") carrying ``dataset_path`` plus the additional
regression-detection inputs this package's extended scope needs,
sourced from ``IConfigRepository`` by the ``CompositionRoot`` exactly
like every locked config value object already is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from uaqe.common.interfaces.i_logger import ILogger

#: Every evaluation dimension this package can produce, in the order
#: they are run by :meth:`~uaqe.evaluation.evaluator.Evaluator.evaluate`.
ACCURACY = "accuracy"
LATENCY = "latency"
MEMORY = "memory"
POWER = "power"
SIZE = "size"
COMPATIBILITY = "compatibility"
REGRESSION = "regression"

#: Dimensions that always run: neither needs a resolved
#: ``HardwareProfile`` to produce a meaningful score.
_HARDWARE_FREE_DIMENSIONS = (ACCURACY, SIZE)

#: Dimensions gated on a resolved ``HardwareProfile`` being available.
_HARDWARE_DEPENDENT_DIMENSIONS = (LATENCY, MEMORY, POWER, COMPATIBILITY)


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration governing what :class:`~uaqe.evaluation.evaluator.
    Evaluator` evaluates and how strictly it judges the result.

    Attributes:
        dataset_path: Path to a representative evaluation dataset,
            forwarded to
            :meth:`~uaqe.evaluation.accuracy_evaluator.
            AccuracyEvaluator.evaluate` — the locked
            ``Evaluator.evaluate`` parameter of the same name
            (``03_API_Specification.md`` §10.1).
        regression_thresholds: Per-metric acceptable relative
            regression overrides passed to
            :meth:`~uaqe.evaluation.regression_evaluator.
            RegressionEvaluator.check`; unset metrics fall back to
            :data:`~uaqe.evaluation.regression_evaluator.
            DEFAULT_THRESHOLDS`.
        baseline_metrics: The prior run's metric values to compare
            against, sourced by the ``CompositionRoot`` from wherever
            prior-run results are persisted. Empty (the default) means
            no regression check runs for this evaluation.
        fail_on_incompatibility: Whether
            ``EvaluationResult.passed`` is set to ``False`` when the
            final compatibility re-check reports
            ``compatible=False``, in addition to any regression
            finding.
    """

    dataset_path: Optional[str] = None
    regression_thresholds: Dict[str, float] = field(default_factory=dict)
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    fail_on_incompatibility: bool = True


@dataclass(frozen=True)
class EvaluationPlan:
    """Which evaluation dimensions a run should attempt.

    Attributes:
        enabled_dimensions: The dimension names (see the module-level
            constants) this run will attempt, in evaluation order.
        rationale: A human-readable explanation of why any dimension
            was skipped.
    """

    enabled_dimensions: List[str] = field(
        default_factory=lambda: [ACCURACY, SIZE]
    )
    rationale: str = ""


class EvaluationPlanner:
    """Builds an :class:`EvaluationPlan` from the inputs available for a
    given run.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            planner operates silently.
    """

    def __init__(self, logger: Optional[ILogger] = None) -> None:
        """Initialize an ``EvaluationPlanner``.

        Args:
            logger: Optional structured logging sink.
        """
        self.logger: Optional[ILogger] = logger

    def build_plan(
        self,
        config: EvaluationConfig,
        *,
        has_hardware_profile: bool,
        has_baseline_metrics: bool,
    ) -> EvaluationPlan:
        """Decide which evaluation dimensions this run should attempt.

        Args:
            config: The run's ``EvaluationConfig``.
            has_hardware_profile: Whether a resolved ``HardwareProfile``
                is available this run (from ``\"hardware_manager\"``).
            has_baseline_metrics: Whether ``config.baseline_metrics`` is
                non-empty (checked by the caller rather than this
                method inspecting ``config`` twice, so the same
                emptiness rule is never defined in two places).

        Returns:
            The finalized :class:`EvaluationPlan`.
        """
        enabled = list(_HARDWARE_FREE_DIMENSIONS)
        skipped_notes: List[str] = []

        if has_hardware_profile:
            enabled.extend(_HARDWARE_DEPENDENT_DIMENSIONS)
        else:
            skipped_notes.append(
                f"no resolved HardwareProfile available; skipping "
                f"{list(_HARDWARE_DEPENDENT_DIMENSIONS)}."
            )

        if has_baseline_metrics:
            enabled.append(REGRESSION)
        else:
            skipped_notes.append(
                "EvaluationConfig.baseline_metrics is empty; skipping "
                f"{REGRESSION!r}."
            )

        rationale = (
            "All dimensions enabled."
            if not skipped_notes
            else " ".join(skipped_notes)
        )

        if self.logger is not None:
            self.logger.info(
                "Evaluation plan built.",
                enabled_dimensions=enabled,
            )

        return EvaluationPlan(enabled_dimensions=enabled, rationale=rationale)
