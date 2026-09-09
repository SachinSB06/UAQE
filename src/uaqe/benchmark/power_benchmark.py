"""Power draw and per-inference energy estimation for a benchmark run.

``PowerBenchmark`` wraps :class:`~uaqe.hardware.power_estimator.
PowerEstimator`, the same collaborator
:class:`~uaqe.evaluation.power_evaluator.PowerEvaluator` and
:class:`~uaqe.optimizer.power_optimizer.PowerOptimizer` already wrap
for their own estimation needs — but where those two collaborators
each derive their own :class:`~uaqe.hardware.latency_estimator.
LatencyEstimate` first (since ``PowerEstimator.estimate`` needs one to
derive per-inference energy), this benchmark does not: a benchmark run
has already *measured* per-trial latency via
:class:`~uaqe.benchmark.runtime_benchmark.RuntimeBenchmark` and reduced
it to a :class:`~uaqe.benchmark.benchmark_result.LatencyBenchmarkScore`,
so re-deriving a second, independent latency estimate here would only
disagree with the figure this same run already reported as
``BenchmarkResult.latency_ms_p50``. Instead, this benchmark wraps the
already-measured ``latency_ms_p50`` in a synthetic
:class:`~uaqe.hardware.latency_estimator.LatencyEstimate` (with
``used_clock_model=True``, since the figure is measured rather than
derived from ``PowerEstimator``'s own fallback-model caveat) purely to
satisfy ``PowerEstimator.estimate``'s parameter shape.

Per ``10_Module_Development_Guide.md`` §10's "additional metrics [...]
are additive fields on ``BenchmarkResult``, never a breaking rename",
:meth:`PowerBenchmark.measure` returns ``None`` — not a zero-filled
score — when no dominant precision can be resolved (an empty or
unavailable ``IMR``), so ``BenchmarkResult.power`` stays the
``Optional`` additive field its own docstring already describes,
rather than a misleading all-zero reading.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Optional

from uaqe.benchmark.benchmark_result import (
    LatencyBenchmarkScore,
    PowerBenchmarkScore,
)
from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.types import Precision
from uaqe.hardware.latency_estimator import LatencyEstimate
from uaqe.hardware.power_estimator import PowerEstimator

if TYPE_CHECKING:
    from uaqe.domain.hardware_manager import HardwareProfile


class PowerBenchmark:
    """Estimates power draw and per-inference energy from a run's
    measured latency.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            benchmark operates silently.
    """

    def __init__(
        self,
        logger: Optional[ILogger] = None,
        estimator: Optional[PowerEstimator] = None,
    ) -> None:
        """Initialize a ``PowerBenchmark``.

        Args:
            logger: Optional structured logging sink.
            estimator: The underlying power estimator to delegate to; a
                fresh :class:`~uaqe.hardware.power_estimator.
                PowerEstimator` is constructed if omitted.
        """
        self.logger: Optional[ILogger] = logger
        self._estimator = estimator or PowerEstimator(logger)

    def measure(
        self,
        imr: Optional[IMR],
        profile: "HardwareProfile",
        latency: LatencyBenchmarkScore,
    ) -> Optional[PowerBenchmarkScore]:
        """Estimate power/energy for ``profile`` from ``latency``'s
        measured ``latency_ms_p50``.

        Args:
            imr: The model the benchmarked artifact was exported from,
                if available; used only to resolve a dominant
                ``Precision`` to scale the reference power figure.
            profile: The resolved deployment target.
            latency: This run's already-measured
                :class:`~uaqe.benchmark.benchmark_result.
                LatencyBenchmarkScore`.

        Returns:
            The resulting :class:`~uaqe.benchmark.benchmark_result.
            PowerBenchmarkScore`, or ``None`` if ``imr`` is ``None`` or
            has no layers to resolve a dominant precision from.
        """
        if imr is None or not imr.layers:
            if self.logger is not None:
                self.logger.warning(
                    "No IMR (or an empty IMR) available this benchmark "
                    "run; PowerBenchmark cannot resolve a dominant "
                    "precision, so BenchmarkResult.power stays None."
                )
            return None

        measured_latency = LatencyEstimate(
            total_latency_ms=latency.latency_ms_p50,
            used_clock_model=True,
        )
        dominant_precision = self._dominant_precision(imr)
        estimate = self._estimator.estimate(
            profile, measured_latency, dominant_precision=dominant_precision
        )

        assumptions = list(estimate.assumptions)
        assumptions.append(
            "energy_per_inference_mj is derived from this run's "
            "measured latency_ms_p50, not a re-estimated latency "
            "figure; average_power_mw/peak_power_mw remain a "
            "reference-table power estimate (see PowerEstimator)."
        )

        if self.logger is not None:
            self.logger.info(
                "Power benchmarked.",
                profile_id=profile.profile_id,
                average_power_mw=estimate.average_power_mw,
                energy_per_inference_mj=estimate.energy_per_inference_mj,
                dominant_precision=dominant_precision.value,
            )

        return PowerBenchmarkScore(
            average_power_mw=estimate.average_power_mw,
            peak_power_mw=estimate.peak_power_mw,
            energy_per_inference_mj=estimate.energy_per_inference_mj,
            assumptions=assumptions,
        )

    def _dominant_precision(self, imr: IMR) -> Precision:
        """Determine the most common ``Precision`` across ``imr``'s
        layers, mirroring
        :meth:`~uaqe.evaluation.power_evaluator.PowerEvaluator.
        _dominant_precision` and
        :meth:`~uaqe.optimizer.power_optimizer.PowerOptimizer.
        _dominant_precision`.

        Args:
            imr: The model to inspect; guaranteed non-empty by
                :meth:`measure`'s own guard.

        Returns:
            The most frequently occurring ``Precision`` among
            ``imr.layers``.
        """
        counts = Counter(layer.precision for layer in imr.layers)
        return counts.most_common(1)[0][0]
