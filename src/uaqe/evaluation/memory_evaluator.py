"""Baseline-vs-optimized activation-arena memory comparison.

``MemoryEvaluator`` wraps :class:`~uaqe.hardware.memory_planner.
MemoryPlanner` (see that module's docstring for the ping-pong
buffer-arena planning model) and calls it, non-strictly, for both the
pre-optimization and the final, post-optimization ``IMR`` against the
same ``HardwareProfile``. Unlike
:class:`~uaqe.optimizer.memory_optimizer.MemoryOptimizer`, which plans
strictly (raising on a ceiling breach since it is the last checkpoint
before ``Exporter``), this evaluator always plans with ``strict=False``
so that a baseline model that would not itself fit on the target — the
common case being evaluated, since the whole point of optimization is
to make an otherwise-oversized model fit — does not prevent the
comparison from completing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.evaluation.evaluation_result import MemoryScore
from uaqe.hardware.memory_planner import MemoryPlanner

if TYPE_CHECKING:
    # Deferred to break the hardware_manager <-> memory_planner import
    # cycle already documented in that module; this evaluator only
    # needs HardwareProfile as a type hint.
    from uaqe.domain.hardware_manager import HardwareProfile


class MemoryEvaluator:
    """Compares planned peak activation-arena memory usage before and
    after optimization.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            evaluator operates silently.
    """

    def __init__(
        self,
        logger: Optional[ILogger] = None,
        planner: Optional[MemoryPlanner] = None,
    ) -> None:
        """Initialize a ``MemoryEvaluator``.

        Args:
            logger: Optional structured logging sink.
            planner: The underlying activation-arena planner to
                delegate to; a fresh
                :class:`~uaqe.hardware.memory_planner.MemoryPlanner` is
                constructed if omitted.
        """
        self.logger: Optional[ILogger] = logger
        self._planner = planner or MemoryPlanner(logger)

    def evaluate(
        self, baseline_imr: IMR, optimized_imr: IMR, profile: "HardwareProfile"
    ) -> MemoryScore:
        """Compare ``baseline_imr`` and ``optimized_imr`` peak memory on
        ``profile``.

        Args:
            baseline_imr: The pre-optimization model.
            optimized_imr: The final, post-optimization model.
            profile: The resolved deployment target both models are
                planned against.

        Returns:
            The resulting :class:`~uaqe.evaluation.evaluation_result.
            MemoryScore`.
        """
        baseline_plan = self._planner.plan(baseline_imr, profile, strict=False)
        optimized_plan = self._planner.plan(optimized_imr, profile, strict=False)

        baseline_peak = baseline_plan.peak_memory_bytes
        optimized_peak = optimized_plan.peak_memory_bytes
        reduction_ratio = baseline_peak / max(optimized_peak, 1)

        if self.logger is not None:
            self.logger.info(
                "Memory evaluated.",
                baseline_peak_memory_bytes=baseline_peak,
                optimized_peak_memory_bytes=optimized_peak,
                reduction_ratio=reduction_ratio,
            )

        return MemoryScore(
            baseline_peak_memory_bytes=baseline_peak,
            optimized_peak_memory_bytes=optimized_peak,
            memory_delta_bytes=optimized_peak - baseline_peak,
            reduction_ratio=reduction_ratio,
            baseline_static_weight_bytes=baseline_plan.static_weight_bytes,
            optimized_static_weight_bytes=optimized_plan.static_weight_bytes,
        )
