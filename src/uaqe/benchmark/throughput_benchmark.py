"""Single-stream throughput derivation from a measured latency score.

``ThroughputBenchmark`` does not run any trial of its own: the locked
``BenchmarkResult.throughput_inferences_per_sec`` field
(``03_API_Specification.md`` §11.1) is fully determined by the
already-measured latency distribution, so this class is a pure
reduction over
:class:`~uaqe.benchmark.benchmark_result.LatencyBenchmarkScore`.

Derivation: ``throughput_inferences_per_sec = 1000.0 /
latency_ms_p50``. The median, rather than the mean, is used
deliberately — a single slow outlier trial (a scheduling hiccup on a
shared CPU, a GC pause) should not drag down the reported steady-state
throughput figure the way it would if the mean latency were used
instead; ``latency_ms_p99`` is reserved for tail-latency reporting via
:class:`~uaqe.benchmark.benchmark_result.LatencyBenchmarkScore` itself,
not folded into this figure.
"""

from __future__ import annotations

from typing import Optional

from uaqe.benchmark.benchmark_result import (
    LatencyBenchmarkScore,
    ThroughputBenchmarkScore,
)
from uaqe.common.interfaces.i_logger import ILogger

#: Milliseconds per second, used to convert a per-inference latency
#: figure into an inferences-per-second throughput figure.
_MS_PER_SECOND: float = 1000.0


class ThroughputBenchmark:
    """Derives single-stream throughput from a measured latency score.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            benchmark operates silently.
    """

    def __init__(self, logger: Optional[ILogger] = None) -> None:
        """Initialize a ``ThroughputBenchmark``.

        Args:
            logger: Optional structured logging sink.
        """
        self.logger: Optional[ILogger] = logger

    def measure(self, latency: LatencyBenchmarkScore) -> ThroughputBenchmarkScore:
        """Derive throughput from ``latency``.

        Args:
            latency: The run's already-computed
                :class:`~uaqe.benchmark.benchmark_result.
                LatencyBenchmarkScore`.

        Returns:
            The resulting :class:`~uaqe.benchmark.benchmark_result.
            ThroughputBenchmarkScore`; ``throughput_inferences_per_sec``
            is ``0.0`` if ``latency.latency_ms_p50`` is non-positive
            (e.g. no trials were recorded).
        """
        if latency.latency_ms_p50 <= 0.0:
            return ThroughputBenchmarkScore(throughput_inferences_per_sec=0.0)

        throughput = _MS_PER_SECOND / latency.latency_ms_p50

        if self.logger is not None:
            self.logger.info(
                "Throughput benchmarked.",
                throughput_inferences_per_sec=throughput,
            )

        return ThroughputBenchmarkScore(throughput_inferences_per_sec=throughput)
