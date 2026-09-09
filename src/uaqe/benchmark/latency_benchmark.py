"""Percentile/summary statistics over a completed run's raw per-trial
latency samples.

``LatencyBenchmark`` does not itself execute or simulate any trial:
it reduces the raw ``per_trial_latency_ms`` list already produced by
:class:`~uaqe.benchmark.runtime_benchmark.RuntimeBenchmark` (with
``warmup_trials`` already excluded — see that module) into the locked
``latency_ms_p50``/``latency_ms_p99`` figures plus the mean/min/max
this package additionally reports, exactly the same
statistics-over-an-already-collected-series role
:class:`~uaqe.infrastructure.metrics.metrics_collector.
MetricsCollector.summary` plays for arbitrary named series in general.

Percentile method: the nearest-rank method over the sorted sample
list (``rank = ceil(p * n)``, 1-indexed, clamped to ``[1, n]``) — a
simple, dependency-free definition that needs no interpolation and
matches exactly for ``n`` a multiple of 100, which is true for this
package's default ``trials=20`` only at ``p50`` but is otherwise a
reasonable, auditable approximation for smaller ``n``.
"""

from __future__ import annotations

import math
from typing import List, Optional

from uaqe.benchmark.benchmark_result import LatencyBenchmarkScore
from uaqe.common.interfaces.i_logger import ILogger


class LatencyBenchmark:
    """Computes percentile/summary latency statistics from a completed
    run's raw per-trial samples.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            benchmark operates silently.
    """

    def __init__(self, logger: Optional[ILogger] = None) -> None:
        """Initialize a ``LatencyBenchmark``.

        Args:
            logger: Optional structured logging sink.
        """
        self.logger: Optional[ILogger] = logger

    def measure(self, per_trial_latency_ms: List[float]) -> LatencyBenchmarkScore:
        """Reduce ``per_trial_latency_ms`` to a
        :class:`~uaqe.benchmark.benchmark_result.LatencyBenchmarkScore`.

        Args:
            per_trial_latency_ms: The non-warmup trial latencies, in
                execution order, from
                :attr:`~uaqe.benchmark.benchmark_result.
                RuntimeBenchmarkScore.per_trial_latency_ms`.

        Returns:
            The resulting :class:`~uaqe.benchmark.benchmark_result.
            LatencyBenchmarkScore`; a zero-filled score with
            ``sample_count=0`` if ``per_trial_latency_ms`` is empty.
        """
        if not per_trial_latency_ms:
            return LatencyBenchmarkScore()

        sorted_samples = sorted(per_trial_latency_ms)
        sample_count = len(sorted_samples)

        score = LatencyBenchmarkScore(
            latency_ms_p50=self._percentile(sorted_samples, 0.50),
            latency_ms_p99=self._percentile(sorted_samples, 0.99),
            latency_ms_mean=sum(sorted_samples) / sample_count,
            latency_ms_min=sorted_samples[0],
            latency_ms_max=sorted_samples[-1],
            sample_count=sample_count,
        )

        if self.logger is not None:
            self.logger.info(
                "Latency benchmarked.",
                latency_ms_p50=score.latency_ms_p50,
                latency_ms_p99=score.latency_ms_p99,
                sample_count=sample_count,
            )

        return score

    def _percentile(self, sorted_samples: List[float], p: float) -> float:
        """Compute the ``p``-th percentile of an already-sorted sample
        list via the nearest-rank method.

        Args:
            sorted_samples: The samples, sorted ascending.
            p: The percentile to compute, in ``[0, 1]``.

        Returns:
            The value at the computed rank.
        """
        rank = max(1, min(len(sorted_samples), math.ceil(p * len(sorted_samples))))
        return sorted_samples[rank - 1]
