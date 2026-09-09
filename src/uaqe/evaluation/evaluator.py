"""Post-export correctness/quality evaluation — the locked
``uaqe.domain.evaluation.evaluator`` contract
(``03_API_Specification.md`` §10.1), decomposed into this package.

``Evaluator`` is the sole orchestrating ``PipelineStage`` for this
package (registered under context key ``\"evaluator\"``). Per
``09_Architecture_Lock.md`` §12, it runs after ``Exporter`` alongside
``Benchmarker`` — the one point in the pipeline where two stages may
execute independently before the next join point
(``deployment_readiness_scorer``). Per ``01_Project_Architecture.md``
§6's strict responsibility split, ``Evaluator`` never runs on-device
timing (that is ``Benchmarker``'s job) — the ``latency``/``power``
dimensions this package does produce are the same static estimation
models :mod:`uaqe.hardware` and :mod:`uaqe.optimizer` already use during
search, evaluated once more, before/after, for reporting; they are not
a substitute for ``Benchmarker.run_benchmark``'s measured
``BenchmarkResult``.

Evaluation (``evaluate`` -- locked minimal signature
``evaluate(baseline_imr, optimized_imr, dataset_path) -> EvaluationResult``,
extended here with optional ``profile``/``artifact``/``baseline_metrics``
keyword arguments the same way
:meth:`~uaqe.optimizer.optimizer.Optimizer.search` extends its own
locked signature):

1. :class:`~uaqe.evaluation.accuracy_evaluator.AccuracyEvaluator` always
   runs, scoring precision-retention between the two IMRs.
2. :class:`~uaqe.evaluation.size_evaluator.SizeEvaluator` always runs,
   comparing serialized (or actual exported) size.
3. When a resolved ``HardwareProfile`` is available:
   :class:`~uaqe.evaluation.latency_evaluator.LatencyEvaluator`,
   :class:`~uaqe.evaluation.memory_evaluator.MemoryEvaluator`,
   :class:`~uaqe.evaluation.power_evaluator.PowerEvaluator`, and
   :class:`~uaqe.evaluation.compatibility_evaluator.
   CompatibilityEvaluator` each run in turn.
4. When ``EvaluationConfig.baseline_metrics`` is non-empty:
   :class:`~uaqe.evaluation.regression_evaluator.RegressionEvaluator`
   compares this run's metrics against them.

:class:`~uaqe.evaluation.evaluation_planner.EvaluationPlanner` decides
steps 3-4's applicability before any sub-evaluator runs, so
``Evaluator`` itself never inspects ``config``/context availability
more than once.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.result_types import StageResult
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.domain.pipeline_stage import PipelineStage
from uaqe.evaluation.accuracy_evaluator import AccuracyEvaluator
from uaqe.evaluation.compatibility_evaluator import CompatibilityEvaluator
from uaqe.evaluation.evaluation_planner import (
    COMPATIBILITY,
    LATENCY,
    MEMORY,
    POWER,
    REGRESSION,
    EvaluationConfig,
    EvaluationPlanner,
)
from uaqe.evaluation.evaluation_result import EvaluationResult
from uaqe.evaluation.latency_evaluator import LatencyEvaluator
from uaqe.evaluation.memory_evaluator import MemoryEvaluator
from uaqe.evaluation.power_evaluator import PowerEvaluator
from uaqe.evaluation.regression_evaluator import RegressionEvaluator
from uaqe.evaluation.size_evaluator import SizeEvaluator

if TYPE_CHECKING:
    # Deferred: this stage only needs DeploymentArtifact as a type hint
    # for an optional input already produced elsewhere; importing
    # uaqe.exporter at module scope would otherwise pull in that
    # package's full __init__ for no runtime benefit here, mirroring
    # the same TYPE_CHECKING-guarded cross-package type hint pattern
    # uaqe.hardware already uses for HardwareProfile.
    from uaqe.exporter.exporter import DeploymentArtifact

#: The context key holding the pre-optimization baseline ``IMR``.
_MODEL_LOADER_STAGE_NAME = "model_loader"

#: The context key this stage reads the final ``IMR`` from when
#: present, per :class:`~uaqe.optimizer.memory_optimizer.MemoryOptimizer`'s
#: own registered stage name.
_MEMORY_OPTIMIZER_STAGE_NAME = "memory_optimizer"

#: Fallback context keys, in resolution order, when
#: ``"memory_optimizer"`` has not (yet) run — mirroring
#: ``Exporter``'s own fallback chain exactly, since this stage runs
#: immediately after it in the locked stage order.
_FALLBACK_STAGE_NAMES = (
    "optimizer",
    "compression_planner",
    "quantization_planner",
    _MODEL_LOADER_STAGE_NAME,
)

#: The ``HardwareProfile``-producing stage, read at most once per
#: ``09_Architecture_Lock.md`` §13 rule 3.
_HARDWARE_STAGE_NAME = "hardware_manager"

#: The ``DeploymentArtifact``-producing stage, consulted for its
#: measured ``size_bytes`` when available.
_EXPORTER_STAGE_NAME = "exporter"


class Evaluator(PipelineStage):
    """Evaluates accuracy, resource, and compatibility outcomes for a
    completed optimization run.

    Attributes:
        _logger: Structured logging sink.
        _config: The run's ``EvaluationConfig``, supplied via
            constructor injection.
        _planner: Decides which evaluation dimensions this run
            attempts.
        _accuracy_evaluator: Scores precision-retention.
        _latency_evaluator: Compares estimated latency.
        _memory_evaluator: Compares planned peak memory.
        _power_evaluator: Compares estimated power/energy.
        _size_evaluator: Compares serialized/exported size.
        _compatibility_evaluator: Re-checks final compatibility.
        _regression_evaluator: Compares against a prior-run baseline.
    """

    def __init__(
        self,
        logger: ILogger,
        config: Optional[EvaluationConfig] = None,
        planner: Optional[EvaluationPlanner] = None,
        accuracy_evaluator: Optional[AccuracyEvaluator] = None,
        latency_evaluator: Optional[LatencyEvaluator] = None,
        memory_evaluator: Optional[MemoryEvaluator] = None,
        power_evaluator: Optional[PowerEvaluator] = None,
        size_evaluator: Optional[SizeEvaluator] = None,
        compatibility_evaluator: Optional[CompatibilityEvaluator] = None,
        regression_evaluator: Optional[RegressionEvaluator] = None,
    ) -> None:
        """Initialize the ``Evaluator``.

        Args:
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
            config: The run's ``EvaluationConfig``, sourced from
                ``IConfigRepository`` by the ``CompositionRoot``. A
                default (dataset-free, regression-free) config is used
                if omitted.
            planner: Decides which dimensions this run attempts; a
                fresh
                :class:`~uaqe.evaluation.evaluation_planner.
                EvaluationPlanner` is constructed if omitted.
            accuracy_evaluator: Scores precision-retention; a fresh
                :class:`~uaqe.evaluation.accuracy_evaluator.
                AccuracyEvaluator` is constructed if omitted.
            latency_evaluator: Compares estimated latency; a fresh
                :class:`~uaqe.evaluation.latency_evaluator.
                LatencyEvaluator` is constructed if omitted.
            memory_evaluator: Compares planned peak memory; a fresh
                :class:`~uaqe.evaluation.memory_evaluator.
                MemoryEvaluator` is constructed if omitted.
            power_evaluator: Compares estimated power/energy; a fresh
                :class:`~uaqe.evaluation.power_evaluator.PowerEvaluator`
                is constructed if omitted.
            size_evaluator: Compares serialized/exported size; a fresh
                :class:`~uaqe.evaluation.size_evaluator.SizeEvaluator`
                is constructed if omitted.
            compatibility_evaluator: Re-checks final compatibility; a
                fresh
                :class:`~uaqe.evaluation.compatibility_evaluator.
                CompatibilityEvaluator` is constructed if omitted.
            regression_evaluator: Compares against a prior-run baseline;
                a fresh
                :class:`~uaqe.evaluation.regression_evaluator.
                RegressionEvaluator` is constructed if omitted.
        """
        self._logger = logger
        self._config = config or EvaluationConfig()
        self._planner = planner or EvaluationPlanner(logger)
        self._accuracy_evaluator = accuracy_evaluator or AccuracyEvaluator(logger)
        self._latency_evaluator = latency_evaluator or LatencyEvaluator(logger)
        self._memory_evaluator = memory_evaluator or MemoryEvaluator(logger)
        self._power_evaluator = power_evaluator or PowerEvaluator(logger)
        self._size_evaluator = size_evaluator or SizeEvaluator(logger)
        self._compatibility_evaluator = (
            compatibility_evaluator or CompatibilityEvaluator(logger)
        )
        self._regression_evaluator = regression_evaluator or RegressionEvaluator(logger)

    def execute(self, context: PipelineContext) -> StageResult:
        """Run this run's evaluation.

        Reads the baseline ``IMR`` from ``"model_loader"``, the final
        ``IMR`` per the resolution order documented in the module
        docstring, the resolved ``HardwareProfile`` from
        ``"hardware_manager"`` if present, and the
        ``DeploymentArtifact`` from ``"exporter"`` if present.

        Args:
            context: The current run's pipeline context.

        Returns:
            A ``StageResult`` whose ``payload`` is the finalized
            ``EvaluationResult``.
        """
        start = time.monotonic()
        baseline_imr = context.get(_MODEL_LOADER_STAGE_NAME).payload
        optimized_imr = self._resolve_current_imr(context)

        profile: Optional[HardwareProfile] = None
        if context.has(_HARDWARE_STAGE_NAME):
            profile = context.get(_HARDWARE_STAGE_NAME).payload

        artifact: Optional[DeploymentArtifact] = None
        if context.has(_EXPORTER_STAGE_NAME):
            exporter_payload = context.get(_EXPORTER_STAGE_NAME).payload
            artifact = (
                exporter_payload[0]
                if isinstance(exporter_payload, tuple)
                else exporter_payload
            )

        result = self.evaluate(
            baseline_imr,
            optimized_imr,
            self._config.dataset_path,
            profile=profile,
            artifact=artifact,
            baseline_metrics=self._config.baseline_metrics,
        )

        duration_ms = (time.monotonic() - start) * 1000.0
        self._logger.info(
            "Evaluation complete.",
            stage_name=self.name(),
            accuracy_delta=result.accuracy_delta,
            passed=result.passed,
            duration_ms=duration_ms,
        )
        return StageResult(
            stage_name=self.name(),
            success=True,
            payload=result,
            duration_ms=duration_ms,
        )

    def evaluate(
        self,
        baseline_imr: IMR,
        optimized_imr: IMR,
        dataset_path: Optional[str],
        *,
        profile: Optional[HardwareProfile] = None,
        artifact: Optional[DeploymentArtifact] = None,
        baseline_metrics: Optional[dict] = None,
    ) -> EvaluationResult:
        """Evaluate ``optimized_imr`` against ``baseline_imr``.

        Args:
            baseline_imr: The pre-optimization model.
            optimized_imr: The final, post-optimization model.
            dataset_path: Path to a representative evaluation dataset,
                forwarded to
                :meth:`~uaqe.evaluation.accuracy_evaluator.
                AccuracyEvaluator.evaluate` — the locked
                ``Evaluator.evaluate`` parameter of the same name.
            profile: The resolved deployment target, if available.
                Gates the ``latency``/``memory``/``power``/
                ``compatibility`` dimensions.
            artifact: The ``DeploymentArtifact`` from an already-run
                ``Exporter`` stage, if available. Used for an exact
                ``size`` figure in place of the static estimate.
            baseline_metrics: A prior run's metric values to check for
                regression against. Gates the ``regression`` dimension.

        Returns:
            The finalized :class:`~uaqe.evaluation.evaluation_result.
            EvaluationResult`.
        """
        baseline_metrics = baseline_metrics or {}
        plan = self._planner.build_plan(
            self._config,
            has_hardware_profile=profile is not None,
            has_baseline_metrics=bool(baseline_metrics),
        )

        warnings = [] if plan.rationale == "All dimensions enabled." else [plan.rationale]

        accuracy = self._accuracy_evaluator.evaluate(
            baseline_imr, optimized_imr, dataset_path
        )
        size = self._size_evaluator.evaluate(baseline_imr, optimized_imr, artifact)

        result = EvaluationResult(
            baseline_accuracy=accuracy.baseline_accuracy,
            optimized_accuracy=accuracy.optimized_accuracy,
            accuracy_delta=accuracy.accuracy_delta,
            per_layer_error=accuracy.per_layer_error,
            accuracy=accuracy,
            size=size,
            warnings=warnings,
        )

        if profile is not None:
            if LATENCY in plan.enabled_dimensions:
                result.latency = self._latency_evaluator.evaluate(
                    baseline_imr, optimized_imr, profile
                )
            if MEMORY in plan.enabled_dimensions:
                result.memory = self._memory_evaluator.evaluate(
                    baseline_imr, optimized_imr, profile
                )
            if POWER in plan.enabled_dimensions:
                result.power = self._power_evaluator.evaluate(
                    baseline_imr, optimized_imr, profile
                )
            if COMPATIBILITY in plan.enabled_dimensions:
                result.compatibility = self._compatibility_evaluator.evaluate(
                    optimized_imr, profile
                )

        if REGRESSION in plan.enabled_dimensions:
            result.regressions = self._regression_evaluator.check(
                result, baseline_metrics, self._config.regression_thresholds
            )

        result.passed = self._compute_passed(result)
        return result

    def _compute_passed(self, result: EvaluationResult) -> bool:
        """Determine ``EvaluationResult.passed`` from its populated
        sub-scores.

        Args:
            result: The in-progress result, with every applicable
                sub-score already populated.

        Returns:
            ``False`` if any regression finding regressed, or if
            ``EvaluationConfig.fail_on_incompatibility`` is ``True`` and
            ``result.compatibility.compatible`` is ``False``; ``True``
            otherwise.
        """
        if any(finding.regressed for finding in result.regressions):
            return False
        if (
            self._config.fail_on_incompatibility
            and result.compatibility is not None
            and not result.compatibility.compatible
        ):
            return False
        return True

    def _resolve_current_imr(self, context: PipelineContext) -> IMR:
        """Resolve the final ``IMR`` from ``context``.

        Args:
            context: The current run's pipeline context.

        Returns:
            The ``IMR`` produced by ``"memory_optimizer"`` if that
            stage has run, else the first of
            :data:`_FALLBACK_STAGE_NAMES` found in ``context``.

        Raises:
            KeyError: If none of ``"memory_optimizer"`` or
                :data:`_FALLBACK_STAGE_NAMES` has been recorded in
                ``context``.
        """
        if context.has(_MEMORY_OPTIMIZER_STAGE_NAME):
            payload = context.get(_MEMORY_OPTIMIZER_STAGE_NAME).payload
            return payload[1] if isinstance(payload, tuple) else payload

        for stage_name in _FALLBACK_STAGE_NAMES:
            if context.has(stage_name):
                payload = context.get(stage_name).payload
                return payload[1] if isinstance(payload, tuple) else payload

        raise KeyError(
            "No IMR-producing stage recorded in context; expected one "
            f"of {(_MEMORY_OPTIMIZER_STAGE_NAME,) + _FALLBACK_STAGE_NAMES!r}."
        )
