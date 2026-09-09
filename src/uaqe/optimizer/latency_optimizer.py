"""Latency-objective scoring and advisory for the multi-objective search.

``LatencyOptimizer`` is the ``"latency"`` objective's scorer, invoked by
:class:`~uaqe.optimizer.optimizer.Optimizer` once per candidate
configuration during :meth:`~uaqe.optimizer.optimizer.Optimizer.search`.
It does not itself rewrite the ``IMR`` — that is
:class:`~uaqe.optimizer.graph_optimizer.GraphOptimizer`'s and
:class:`~uaqe.optimizer.fusion_optimizer.FusionOptimizer`'s job, applied
upstream by :class:`~uaqe.optimizer.optimization_planner.
OptimizationPlanner`. Instead it wraps
:class:`~uaqe.hardware.latency_estimator.LatencyEstimator` (the same
order-of-magnitude MAC-proxy model documented there) to produce a
comparable per-candidate figure, and additionally flags which layers
are driving that figure so a caller can see *why* one candidate scored
better than another, not just that it did.

This module intentionally does not duplicate
``uaqe.hardware.latency_estimator``'s estimation model — see that
module's docstring for the model itself and its documented limitations
(order-of-magnitude, not a measured ``Benchmarker`` result).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.hardware.latency_estimator import LatencyEstimate, LatencyEstimator

#: Number of highest-latency layers surfaced in
#: :attr:`LatencyObjectiveScore.top_contributors`/``recommendations``.
_DEFAULT_TOP_N = 3

#: Precisions considered "reducible" for the purpose of a latency
#: recommendation — a layer already at ``INT8``/``INT4`` has little
#: further precision-driven latency headroom left to suggest.
_REDUCIBLE_PRECISIONS = frozenset({"FP32", "FP16"})


@dataclass
class LatencyObjectiveScore:
    """The outcome of one :meth:`LatencyOptimizer.evaluate` call.

    Attributes:
        estimate: The underlying
            :class:`~uaqe.hardware.latency_estimator.LatencyEstimate`
            this score was derived from.
        top_contributors: The names of the highest-latency layers, most
            expensive first, up to the requested ``top_n``.
        recommendations: Human-readable, non-binding suggestions for
            reducing latency further (e.g. a precision override for a
            top contributor still at ``FP32``/``FP16``).
    """

    estimate: LatencyEstimate
    top_contributors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class LatencyOptimizer:
    """Scores a candidate ``IMR``/``HardwareProfile`` pair on the
    ``"latency"`` objective and advises on further reduction.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            optimizer operates silently.
    """

    def __init__(
        self,
        logger: Optional[ILogger] = None,
        estimator: Optional[LatencyEstimator] = None,
    ) -> None:
        """Initialize a ``LatencyOptimizer``.

        Args:
            logger: Optional structured logging sink.
            estimator: The underlying estimator to delegate to; a
                fresh :class:`~uaqe.hardware.latency_estimator.
                LatencyEstimator` is constructed if omitted.
        """
        self.logger: Optional[ILogger] = logger
        self._estimator = estimator or LatencyEstimator(logger)

    def evaluate(
        self, imr: IMR, profile: HardwareProfile, top_n: int = _DEFAULT_TOP_N
    ) -> LatencyObjectiveScore:
        """Estimate latency for ``imr`` on ``profile`` and identify its
        top contributors.

        Args:
            imr: The candidate model to score.
            profile: The candidate deployment target.
            top_n: The number of highest-latency layers to surface.

        Returns:
            The resulting :class:`LatencyObjectiveScore`.
        """
        estimate = self._estimator.estimate(imr, profile)
        ranked = sorted(
            estimate.per_layer_latency_ms.items(), key=lambda item: item[1], reverse=True
        )
        top_contributors = [name for name, _ in ranked[:top_n]]
        recommendations = self._recommend(imr, top_contributors)

        if self.logger is not None:
            self.logger.info(
                "Latency objective evaluated.",
                total_latency_ms=estimate.total_latency_ms,
                top_contributors=top_contributors,
            )

        return LatencyObjectiveScore(
            estimate=estimate,
            top_contributors=top_contributors,
            recommendations=recommendations,
        )

    def _recommend(self, imr: IMR, top_contributors: List[str]) -> List[str]:
        """Build non-binding latency-reduction suggestions for the
        current top contributors.

        Args:
            imr: The model the contributors were drawn from.
            top_contributors: Layer names from :meth:`evaluate`, most
                expensive first.

        Returns:
            A list of human-readable recommendations, one per
            contributor still at a reducible precision.
        """
        recommendations: List[str] = []
        for layer_name in top_contributors:
            try:
                layer = imr.get_layer(layer_name)
            except KeyError:
                continue
            if layer.precision.value in _REDUCIBLE_PRECISIONS:
                recommendations.append(
                    f"Layer {layer_name!r} ({layer.op_type}) is a top "
                    f"latency contributor at {layer.precision.value}; "
                    "consider a QuantizationConfig.per_layer_overrides "
                    "entry, or confirm it is part of a fusion-eligible "
                    "chain the FusionOptimizer can collapse."
                )
        return recommendations
