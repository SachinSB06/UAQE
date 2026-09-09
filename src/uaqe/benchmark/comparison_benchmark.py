"""Cross-hardware-target benchmark comparison.

``ComparisonBenchmark`` is the capability
``01_Project_Architecture.md`` §"Scalability Strategy" describes:
"``Benchmarker`` [...] designed to consume N hardware-target results
per model in a single pass, avoiding O(N²) recomputation when comparing
multiple targets for the same model." It is a plain collaborator, not
a ``PipelineStage`` — a single pipeline run benchmarks exactly one
resolved ``HardwareProfile`` (per ``09_Architecture_Lock.md`` §12's
``{evaluator, benchmarker}`` join point, itself downstream of a single
``Exporter`` output); comparing across *N* targets for the same model
is a ``BatchRunner``-level concern (``01_Project_Architecture.md``
§18's "horizontal scalability" fan-out), which calls this class
directly over the ``N`` independently-produced
:class:`~uaqe.benchmark.benchmark_result.BenchmarkResult` instances
once every run has completed — mirroring how
:class:`~uaqe.benchmark.comparison_benchmark` sits alongside, not
inside, :class:`~uaqe.benchmark.benchmark.Benchmarker` itself.

Ranking model: each compared target is scored on four ratio-based
sub-objectives — latency (lower ``latency_ms_p50`` is better), memory
(lower ``peak_memory_bytes`` is better), size (lower
``size.artifact_size_bytes`` is better), and throughput (higher
``throughput_inferences_per_sec`` is better) — each normalized against
the *best* value observed across the compared set for that dimension,
so the best performer in each dimension always scores ``1.0`` and every
other target scores proportionally lower, the same normalize-against-
the-best precedent :class:`~uaqe.optimizer.optimizer.Optimizer`'s
own multi-objective search already establishes (there, against a
``\"baseline\"`` candidate rather than the best of the set, since a
comparison has no baseline to normalize against — only competing
targets). ``normalized_score`` is the unweighted mean of the four
sub-scores; ties are broken by compared-list order. This is a single
``O(N)`` pass over ``results`` for the four per-dimension minimums/
maximum plus one further ``O(N)`` pass to score each entry — never
``O(N^2)``.
"""

from __future__ import annotations

from typing import List, Optional

from uaqe.benchmark.benchmark_result import (
    BenchmarkResult,
    ComparisonBenchmarkResult,
    ComparisonEntry,
)
from uaqe.common.interfaces.i_logger import ILogger

#: Floor value substituted for a zero or negative denominator when
#: computing a ratio-based sub-score, so a degenerate (e.g. simulated-
#: with-zero-latency) reading never produces a division error.
_MIN_DENOMINATOR: float = 1e-9


class ComparisonBenchmark:
    """Ranks ``N`` ``BenchmarkResult`` instances for the same model
    across different ``HardwareProfile`` targets.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            benchmark operates silently.
    """

    def __init__(self, logger: Optional[ILogger] = None) -> None:
        """Initialize a ``ComparisonBenchmark``.

        Args:
            logger: Optional structured logging sink.
        """
        self.logger: Optional[ILogger] = logger

    def compare(
        self, results: List[BenchmarkResult]
    ) -> ComparisonBenchmarkResult:
        """Rank ``results`` against one another.

        Args:
            results: One already-produced ``BenchmarkResult`` per
                compared ``HardwareProfile`` target, for the same
                model.

        Returns:
            The resulting :class:`~uaqe.benchmark.benchmark_result.
            ComparisonBenchmarkResult`; an empty, rationale-only result
            if ``results`` is empty.
        """
        if not results:
            return ComparisonBenchmarkResult(
                rationale="No BenchmarkResult supplied; nothing to compare."
            )

        if len(results) == 1:
            only = results[0]
            entry = ComparisonEntry(
                profile_id=only.hardware_profile_id,
                result=only,
                normalized_score=1.0,
                rank=1,
            )
            if self.logger is not None:
                self.logger.info(
                    "Benchmark comparison complete.",
                    compared_profile_count=1,
                    best_overall_profile_id=only.hardware_profile_id,
                )
            return ComparisonBenchmarkResult(
                entries=[entry],
                fastest_profile_id=only.hardware_profile_id,
                lowest_memory_profile_id=only.hardware_profile_id,
                smallest_size_profile_id=only.hardware_profile_id,
                best_overall_profile_id=only.hardware_profile_id,
                rationale="Only one target supplied; trivially ranked first.",
            )

        best_latency_ms = min(result.latency_ms_p50 for result in results)
        best_memory_bytes = min(result.peak_memory_bytes for result in results)
        best_size_bytes = min(
            result.size.artifact_size_bytes for result in results
        )
        best_throughput = max(
            result.throughput_inferences_per_sec for result in results
        )

        entries = [
            ComparisonEntry(
                profile_id=result.hardware_profile_id,
                result=result,
                normalized_score=self._normalized_score(
                    result,
                    best_latency_ms=best_latency_ms,
                    best_memory_bytes=best_memory_bytes,
                    best_size_bytes=best_size_bytes,
                    best_throughput=best_throughput,
                ),
            )
            for result in results
        ]
        entries.sort(key=lambda entry: entry.normalized_score, reverse=True)
        for rank, entry in enumerate(entries, start=1):
            entry.rank = rank

        fastest = min(results, key=lambda result: result.latency_ms_p50)
        lowest_memory = min(results, key=lambda result: result.peak_memory_bytes)
        smallest_size = min(
            results, key=lambda result: result.size.artifact_size_bytes
        )
        best_overall = entries[0]

        rationale = (
            f"Ranked {len(results)} hardware targets by the unweighted mean "
            "of four normalized sub-scores (latency, memory, size, "
            f"throughput), each relative to the best value observed across "
            f"the compared set; {best_overall.profile_id!r} ranked first "
            f"with normalized_score={best_overall.normalized_score:.4f}."
        )

        if self.logger is not None:
            self.logger.info(
                "Benchmark comparison complete.",
                compared_profile_count=len(results),
                best_overall_profile_id=best_overall.profile_id,
                fastest_profile_id=fastest.hardware_profile_id,
            )

        return ComparisonBenchmarkResult(
            entries=entries,
            fastest_profile_id=fastest.hardware_profile_id,
            lowest_memory_profile_id=lowest_memory.hardware_profile_id,
            smallest_size_profile_id=smallest_size.hardware_profile_id,
            best_overall_profile_id=best_overall.profile_id,
            rationale=rationale,
        )

    def _normalized_score(
        self,
        result: BenchmarkResult,
        *,
        best_latency_ms: float,
        best_memory_bytes: int,
        best_size_bytes: int,
        best_throughput: float,
    ) -> float:
        """Compute ``result``'s unweighted-mean normalized score.

        Args:
            result: The target's ``BenchmarkResult``.
            best_latency_ms: The lowest ``latency_ms_p50`` observed
                across the compared set.
            best_memory_bytes: The lowest ``peak_memory_bytes``
                observed across the compared set.
            best_size_bytes: The lowest ``size.artifact_size_bytes``
                observed across the compared set.
            best_throughput: The highest
                ``throughput_inferences_per_sec`` observed across the
                compared set.

        Returns:
            The mean of the four sub-scores, each in ``(0, 1]``, with
            ``1.0`` meaning ``result`` matches the best observed value
            for that dimension.
        """
        latency_score = best_latency_ms / max(result.latency_ms_p50, _MIN_DENOMINATOR)
        memory_score = best_memory_bytes / max(
            result.peak_memory_bytes, _MIN_DENOMINATOR
        )
        size_score = best_size_bytes / max(
            result.size.artifact_size_bytes, _MIN_DENOMINATOR
        )
        throughput_score = result.throughput_inferences_per_sec / max(
            best_throughput, _MIN_DENOMINATOR
        )
        return (latency_score + memory_score + size_score + throughput_score) / 4.0
