"""In-memory numeric metrics aggregation for a single run.

``MetricsCollector`` accumulates arbitrary named float samples (e.g.
per-trial benchmark latencies, per-layer compression ratios) during a
run and reduces each named series to summary statistics on demand.
It performs no I/O and has no dependency on any other ``uaqe`` package,
per ``09_Architecture_Lock.md`` §8.

Locked contract: ``03_API_Specification.md`` §19.1.
"""

from __future__ import annotations

import threading
from typing import Dict, List


class MetricsCollector:
    """Thread-safe collector of named numeric sample series.

    Attributes:
        metrics: Every recorded sample, keyed by metric name, in the
            order it was recorded.
    """

    def __init__(self) -> None:
        """Initialize an empty ``MetricsCollector``."""
        self.metrics: Dict[str, List[float]] = {}
        self._lock: threading.Lock = threading.Lock()

    def record(self, name: str, value: float) -> None:
        """Record one sample for the named metric series.

        Args:
            name: The metric series name (e.g. ``"benchmark.latency_ms"``).
            value: The sample value to append to that series.
        """
        with self._lock:
            self.metrics.setdefault(name, []).append(value)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Reduce every recorded series to its min/max/avg statistics.

        Returns:
            A mapping of metric name to a ``{"min", "max", "avg"}``
            statistics mapping computed over every sample recorded for
            that name. A series with no recorded samples is omitted.
        """
        with self._lock:
            return {
                name: {
                    "min": min(samples),
                    "max": max(samples),
                    "avg": sum(samples) / len(samples),
                }
                for name, samples in self.metrics.items()
                if samples
            }
