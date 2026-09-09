"""Baseline-vs-optimized power/energy comparison.

``PowerEvaluator`` wraps :class:`~uaqe.hardware.power_estimator.
PowerEstimator`, mirroring how
:class:`~uaqe.optimizer.power_optimizer.PowerOptimizer` uses it during
the multi-objective search: each side (baseline, optimized) needs its
own :class:`~uaqe.hardware.latency_estimator.LatencyEstimate` first,
since ``PowerEstimator.estimate`` derives per-inference energy from
latency. Rather than re-estimating latency itself, this evaluator
accepts an already-computed
:class:`~uaqe.evaluation.latency_evaluator.LatencyEvaluator` and reuses
its collaborator, keeping the latency-estimation model in exactly one
place per :meth:`~uaqe.evaluation.evaluator.Evaluator.evaluate`'s
composition.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Optional

from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.types import Precision
from uaqe.evaluation.evaluation_result import PowerScore
from uaqe.hardware.latency_estimator import LatencyEstimator
from uaqe.hardware.power_estimator import PowerEstimator

if TYPE_CHECKING:
    # Deferred to break the hardware_manager <-> power_estimator import
    # cycle already documented in that module; this evaluator only
    # needs HardwareProfile as a type hint.
    from uaqe.domain.hardware_manager import HardwareProfile


class PowerEvaluator:
    """Compares estimated power draw and per-inference energy before
    and after optimization.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            evaluator operates silently.
    """

    def __init__(
        self,
        logger: Optional[ILogger] = None,
        estimator: Optional[PowerEstimator] = None,
        latency_estimator: Optional[LatencyEstimator] = None,
    ) -> None:
        """Initialize a ``PowerEvaluator``.

        Args:
            logger: Optional structured logging sink.
            estimator: The underlying power estimator to delegate to; a
                fresh :class:`~uaqe.hardware.power_estimator.
                PowerEstimator` is constructed if omitted.
            latency_estimator: The latency estimator this evaluator
                needs internally to derive per-inference energy; a
                fresh :class:`~uaqe.hardware.latency_estimator.
                LatencyEstimator` is constructed if omitted.
        """
        self.logger: Optional[ILogger] = logger
        self._estimator = estimator or PowerEstimator(logger)
        self._latency_estimator = latency_estimator or LatencyEstimator(logger)

    def evaluate(
        self, baseline_imr: IMR, optimized_imr: IMR, profile: "HardwareProfile"
    ) -> PowerScore:
        """Compare ``baseline_imr`` and ``optimized_imr`` power draw on
        ``profile``.

        Args:
            baseline_imr: The pre-optimization model.
            optimized_imr: The final, post-optimization model.
            profile: The resolved deployment target both models are
                estimated against.

        Returns:
            The resulting :class:`~uaqe.evaluation.evaluation_result.
            PowerScore`.
        """
        baseline_latency = self._latency_estimator.estimate(baseline_imr, profile)
        optimized_latency = self._latency_estimator.estimate(optimized_imr, profile)

        baseline_power = self._estimator.estimate(
            profile,
            baseline_latency,
            dominant_precision=self._dominant_precision(baseline_imr),
        )
        optimized_power = self._estimator.estimate(
            profile,
            optimized_latency,
            dominant_precision=self._dominant_precision(optimized_imr),
        )

        energy_reduction_ratio = baseline_power.energy_per_inference_mj / max(
            optimized_power.energy_per_inference_mj, 1e-9
        )

        if self.logger is not None:
            self.logger.info(
                "Power evaluated.",
                baseline_average_power_mw=baseline_power.average_power_mw,
                optimized_average_power_mw=optimized_power.average_power_mw,
                energy_reduction_ratio=energy_reduction_ratio,
            )

        return PowerScore(
            baseline_average_power_mw=baseline_power.average_power_mw,
            optimized_average_power_mw=optimized_power.average_power_mw,
            baseline_energy_per_inference_mj=baseline_power.energy_per_inference_mj,
            optimized_energy_per_inference_mj=optimized_power.energy_per_inference_mj,
            energy_reduction_ratio=energy_reduction_ratio,
        )

    def _dominant_precision(self, imr: IMR) -> Precision:
        """Determine the most common ``Precision`` across ``imr``'s
        layers, mirroring
        :meth:`~uaqe.optimizer.power_optimizer.PowerOptimizer.
        _dominant_precision`.

        Args:
            imr: The model to inspect.

        Returns:
            The most frequently occurring ``Precision`` among
            ``imr.layers``, or ``Precision.FP32`` for an empty model.
        """
        if not imr.layers:
            return Precision.FP32
        counts = Counter(layer.precision for layer in imr.layers)
        return counts.most_common(1)[0][0]
