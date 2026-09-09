"""Baseline-vs-optimized latency comparison for a completed run.

``LatencyEvaluator`` does not itself estimate latency: it wraps
:class:`~uaqe.hardware.latency_estimator.LatencyEstimator` (see that
module's docstring for the underlying MAC-proxy estimation model and
its documented limitations) and calls it once for the pre-optimization
``IMR`` and once for the final, post-optimization ``IMR``, reducing
both estimates to a single before/after :class:`~uaqe.evaluation.
evaluation_result.LatencyScore`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.evaluation.evaluation_result import LatencyScore
from uaqe.hardware.latency_estimator import LatencyEstimator

if TYPE_CHECKING:
    # Deferred to break the hardware_manager <-> latency_estimator
    # import cycle already documented in that module; this evaluator
    # only needs HardwareProfile as a type hint.
    from uaqe.domain.hardware_manager import HardwareProfile


class LatencyEvaluator:
    """Compares estimated inference latency before and after
    optimization.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            evaluator operates silently.
    """

    def __init__(
        self,
        logger: Optional[ILogger] = None,
        estimator: Optional[LatencyEstimator] = None,
    ) -> None:
        """Initialize a ``LatencyEvaluator``.

        Args:
            logger: Optional structured logging sink.
            estimator: The underlying estimator to delegate to; a
                fresh :class:`~uaqe.hardware.latency_estimator.
                LatencyEstimator` is constructed if omitted.
        """
        self.logger: Optional[ILogger] = logger
        self._estimator = estimator or LatencyEstimator(logger)

    def evaluate(
        self, baseline_imr: IMR, optimized_imr: IMR, profile: "HardwareProfile"
    ) -> LatencyScore:
        """Compare ``baseline_imr`` and ``optimized_imr`` latency on
        ``profile``.

        Args:
            baseline_imr: The pre-optimization model.
            optimized_imr: The final, post-optimization model.
            profile: The resolved deployment target both models are
                estimated against.

        Returns:
            The resulting :class:`~uaqe.evaluation.evaluation_result.
            LatencyScore`.
        """
        baseline_estimate = self._estimator.estimate(baseline_imr, profile)
        optimized_estimate = self._estimator.estimate(optimized_imr, profile)

        baseline_latency_ms = baseline_estimate.total_latency_ms
        optimized_latency_ms = optimized_estimate.total_latency_ms
        speedup_ratio = baseline_latency_ms / max(optimized_latency_ms, 1e-9)

        if self.logger is not None:
            self.logger.info(
                "Latency evaluated.",
                baseline_latency_ms=baseline_latency_ms,
                optimized_latency_ms=optimized_latency_ms,
                speedup_ratio=speedup_ratio,
            )

        return LatencyScore(
            baseline_latency_ms=baseline_latency_ms,
            optimized_latency_ms=optimized_latency_ms,
            latency_delta_ms=optimized_latency_ms - baseline_latency_ms,
            speedup_ratio=speedup_ratio,
        )
