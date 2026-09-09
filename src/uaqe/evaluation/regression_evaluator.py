"""Prior-run baseline comparison for regression detection.

``RegressionEvaluator`` never touches the filesystem itself (per
``10_Module_Development_Guide.md`` §0 rule 7, no domain module reads a
config or state file directly): it compares an already-loaded mapping
of a prior run's metric values (``baseline_metrics``, typically sourced
from a previous run's persisted
:meth:`~uaqe.evaluation.evaluation_result.EvaluationResult.to_dict` by
an ``IConfigRepository``/``IFilesystemRepository`` implementation and
threaded in by the caller) against the current run's metrics, flagging
any metric that moved unfavorably beyond its configured threshold.

Threshold semantics: every threshold in :data:`DEFAULT_THRESHOLDS` is a
fraction of the baseline value (e.g. ``0.05`` = "up to a 5% unfavorable
move is tolerated"). Which direction counts as "unfavorable" is
metric-specific — :data:`_METRIC_DIRECTIONS` records it once so the
comparison logic itself never special-cases individual metric names.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from uaqe.common.interfaces.i_logger import ILogger
from uaqe.evaluation.evaluation_result import EvaluationResult, RegressionFinding

#: Default acceptable relative regression, as a fraction of the
#: baseline value, per metric name. A caller may override any subset of
#: these via :meth:`RegressionEvaluator.check`'s ``thresholds``
#: argument; entries not present in ``thresholds`` fall back to these
#: defaults.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "accuracy_delta": 0.05,
    "optimized_latency_ms": 0.20,
    "optimized_peak_memory_bytes": 0.10,
    "optimized_average_power_mw": 0.20,
    "optimized_size_bytes": 0.10,
}

#: Whether a larger current value is the unfavorable direction
#: (``"higher"``) or a smaller one is (``"lower"``) for each metric —
#: e.g. latency/memory/power/size regressions are "got bigger", while
#: an accuracy regression is "got smaller" (more negative).
_METRIC_DIRECTIONS: Dict[str, str] = {
    "accuracy_delta": "lower",
    "optimized_latency_ms": "higher",
    "optimized_peak_memory_bytes": "higher",
    "optimized_average_power_mw": "higher",
    "optimized_size_bytes": "higher",
}


class RegressionEvaluator:
    """Compares a run's metrics against a prior-run baseline.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            evaluator operates silently.
    """

    def __init__(self, logger: Optional[ILogger] = None) -> None:
        """Initialize a ``RegressionEvaluator``.

        Args:
            logger: Optional structured logging sink.
        """
        self.logger: Optional[ILogger] = logger

    def check(
        self,
        result: EvaluationResult,
        baseline_metrics: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> List[RegressionFinding]:
        """Compare ``result``'s metrics against ``baseline_metrics``.

        Args:
            result: The current run's in-progress
                :class:`~uaqe.evaluation.evaluation_result.
                EvaluationResult` (already populated with whichever of
                ``accuracy``/``latency``/``memory``/``power``/``size``
                are available).
            baseline_metrics: The prior run's metric values, keyed by
                the same names as :data:`DEFAULT_THRESHOLDS`. A metric
                absent from this mapping (e.g. a prior run that had no
                resolved ``HardwareProfile`` and so never computed
                ``optimized_latency_ms``) is skipped, not treated as a
                regression.
            thresholds: Per-metric acceptable relative regression
                overrides; unset metrics fall back to
                :data:`DEFAULT_THRESHOLDS`.

        Returns:
            One :class:`~uaqe.evaluation.evaluation_result.
            RegressionFinding` per metric present in both
            ``baseline_metrics`` and the current run's populated
            sub-scores.
        """
        effective_thresholds = dict(DEFAULT_THRESHOLDS)
        if thresholds:
            effective_thresholds.update(thresholds)

        current_metrics = self._current_metrics(result)
        findings: List[RegressionFinding] = []

        for metric_name, baseline_value in baseline_metrics.items():
            if metric_name not in current_metrics:
                continue
            current_value = current_metrics[metric_name]
            threshold = effective_thresholds.get(metric_name, 0.0)
            finding = self._compare(metric_name, baseline_value, current_value, threshold)
            findings.append(finding)

        if self.logger is not None:
            self.logger.info(
                "Regression check complete.",
                metrics_compared=len(findings),
                regressed_count=sum(1 for finding in findings if finding.regressed),
            )

        return findings

    def _current_metrics(self, result: EvaluationResult) -> Dict[str, float]:
        """Flatten ``result``'s populated sub-scores into the same flat
        metric-name space as :data:`DEFAULT_THRESHOLDS`.

        Args:
            result: The current run's ``EvaluationResult``.

        Returns:
            A mapping of metric name to current value, containing only
            the metrics whose backing sub-score is populated.
        """
        metrics: Dict[str, float] = {"accuracy_delta": result.accuracy_delta}
        if result.latency is not None:
            metrics["optimized_latency_ms"] = result.latency.optimized_latency_ms
        if result.memory is not None:
            metrics["optimized_peak_memory_bytes"] = float(
                result.memory.optimized_peak_memory_bytes
            )
        if result.power is not None:
            metrics["optimized_average_power_mw"] = result.power.optimized_average_power_mw
        if result.size is not None:
            metrics["optimized_size_bytes"] = float(result.size.optimized_size_bytes)
        return metrics

    def _compare(
        self,
        metric_name: str,
        baseline_value: float,
        current_value: float,
        threshold: float,
    ) -> RegressionFinding:
        """Build one :class:`RegressionFinding` for a single metric.

        Args:
            metric_name: The metric's identifier.
            baseline_value: The prior run's value for this metric.
            current_value: The current run's value for this metric.
            threshold: The acceptable relative regression fraction.

        Returns:
            The resulting :class:`~uaqe.evaluation.evaluation_result.
            RegressionFinding`.
        """
        direction = _METRIC_DIRECTIONS.get(metric_name, "higher")
        delta = current_value - baseline_value
        denominator = abs(baseline_value) if baseline_value != 0 else 1.0
        relative_change = delta / denominator

        if direction == "higher":
            regressed = relative_change > threshold
        else:
            regressed = relative_change < -threshold

        message = (
            f"{metric_name}: baseline={baseline_value:.6g}, "
            f"current={current_value:.6g}, "
            f"relative_change={relative_change:+.2%}, "
            f"threshold={threshold:.2%} "
            f"({'REGRESSED' if regressed else 'within tolerance'})."
        )

        return RegressionFinding(
            metric_name=metric_name,
            baseline_value=baseline_value,
            current_value=current_value,
            threshold=threshold,
            regressed=regressed,
            message=message,
        )
