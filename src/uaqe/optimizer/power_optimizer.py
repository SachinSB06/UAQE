"""Power-objective scoring and advisory for the multi-objective search.

``PowerOptimizer`` is the ``"power"`` objective's scorer, invoked by
:class:`~uaqe.optimizer.optimizer.Optimizer` once per candidate
configuration alongside
:class:`~uaqe.optimizer.latency_optimizer.LatencyOptimizer`. It wraps
:class:`~uaqe.hardware.power_estimator.PowerEstimator` (see that
module's docstring for the underlying reference-table estimation model
and its documented limitations), supplying it with the candidate's
dominant precision — the single input ``PowerEstimator.estimate``
needs beyond a previously computed
:class:`~uaqe.hardware.latency_estimator.LatencyEstimate` — and adds a
non-binding recommendation when that dominant precision still has
power-reduction headroom.

Like ``LatencyOptimizer``, this module does not itself rewrite the
``IMR``; it only scores a candidate already materialized by
:class:`~uaqe.optimizer.optimization_planner.OptimizationPlanner`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.types import Precision
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.hardware.latency_estimator import LatencyEstimate
from uaqe.hardware.power_estimator import PowerEstimate, PowerEstimator

#: Precisions considered "reducible" for the purpose of a power
#: recommendation — mirrors
#: ``uaqe.optimizer.latency_optimizer._REDUCIBLE_PRECISIONS``.
_REDUCIBLE_PRECISIONS = frozenset({Precision.FP32, Precision.FP16})


@dataclass
class PowerObjectiveScore:
    """The outcome of one :meth:`PowerOptimizer.evaluate` call.

    Attributes:
        estimate: The underlying
            :class:`~uaqe.hardware.power_estimator.PowerEstimate` this
            score was derived from.
        dominant_precision: The precision most layers in the evaluated
            ``IMR`` currently hold, used to scale ``estimate``.
        recommendations: Human-readable, non-binding suggestions for
            reducing power draw further.
    """

    estimate: PowerEstimate
    dominant_precision: Precision
    recommendations: List[str] = field(default_factory=list)


class PowerOptimizer:
    """Scores a candidate ``IMR``/``HardwareProfile`` pair on the
    ``"power"`` objective and advises on further reduction.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            optimizer operates silently.
    """

    def __init__(
        self,
        logger: Optional[ILogger] = None,
        estimator: Optional[PowerEstimator] = None,
    ) -> None:
        """Initialize a ``PowerOptimizer``.

        Args:
            logger: Optional structured logging sink.
            estimator: The underlying estimator to delegate to; a
                fresh :class:`~uaqe.hardware.power_estimator.
                PowerEstimator` is constructed if omitted.
        """
        self.logger: Optional[ILogger] = logger
        self._estimator = estimator or PowerEstimator(logger)

    def evaluate(
        self, imr: IMR, profile: HardwareProfile, latency: LatencyEstimate
    ) -> PowerObjectiveScore:
        """Estimate power draw for ``imr`` on ``profile``.

        Args:
            imr: The candidate model to score.
            profile: The candidate deployment target.
            latency: A previously computed ``LatencyEstimate`` for the
                same candidate (typically from
                :meth:`~uaqe.optimizer.latency_optimizer.
                LatencyOptimizer.evaluate` on the same candidate),
                used to derive per-inference energy.

        Returns:
            The resulting :class:`PowerObjectiveScore`.
        """
        dominant_precision = self._dominant_precision(imr)
        estimate = self._estimator.estimate(
            profile, latency, dominant_precision=dominant_precision
        )
        recommendations = self._recommend(dominant_precision)

        if self.logger is not None:
            self.logger.info(
                "Power objective evaluated.",
                average_power_mw=estimate.average_power_mw,
                dominant_precision=dominant_precision.value,
            )

        return PowerObjectiveScore(
            estimate=estimate,
            dominant_precision=dominant_precision,
            recommendations=recommendations,
        )

    def _dominant_precision(self, imr: IMR) -> Precision:
        """Determine the most common ``Precision`` across ``imr``'s
        layers.

        Args:
            imr: The model to inspect.

        Returns:
            The most frequently occurring ``Precision`` among
            ``imr.layers``, or ``Precision.FP32`` for an empty model
            (a conservative default matching
            ``PowerEstimator.estimate``'s own default).
        """
        if not imr.layers:
            return Precision.FP32
        counts = Counter(layer.precision for layer in imr.layers)
        return counts.most_common(1)[0][0]

    def _recommend(self, dominant_precision: Precision) -> List[str]:
        """Build a non-binding power-reduction suggestion when
        ``dominant_precision`` still has headroom.

        Args:
            dominant_precision: The candidate's dominant precision, as
                computed by :meth:`_dominant_precision`.

        Returns:
            A single-entry list with a recommendation, or an empty
            list if ``dominant_precision`` is already at ``INT8``/
            ``INT4``.
        """
        if dominant_precision in _REDUCIBLE_PRECISIONS:
            return [
                f"Dominant precision is {dominant_precision.value}; "
                "lowering QuantizationConfig.default_precision would "
                "reduce average_power_mw for this profile."
            ]
        return []
