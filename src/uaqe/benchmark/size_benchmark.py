"""Exported artifact size measurement for a benchmark run.

``SizeBenchmark`` does not itself estimate or compute anything: unlike
:class:`~uaqe.evaluation.size_evaluator.SizeEvaluator` (which falls
back to a pre-export static estimate when no
:class:`~uaqe.exporter.exporter.DeploymentArtifact` is available yet),
a benchmark run always has one — the locked ``run_benchmark(artifact,
profile, trials)`` signature (``03_API_Specification.md`` §11.1)
requires it — so this benchmark is a pure reduction over the already-
produced, already-measured ``artifact.size_bytes``, the same
already-collected-figure role
:class:`~uaqe.benchmark.throughput_benchmark.ThroughputBenchmark` plays
for ``throughput_inferences_per_sec``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from uaqe.benchmark.benchmark_result import SizeBenchmarkScore
from uaqe.common.interfaces.i_logger import ILogger

if TYPE_CHECKING:
    from uaqe.exporter.exporter import DeploymentArtifact


class SizeBenchmark:
    """Reduces a benchmarked ``DeploymentArtifact`` to its size figures.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            benchmark operates silently.
    """

    def __init__(self, logger: Optional[ILogger] = None) -> None:
        """Initialize a ``SizeBenchmark``.

        Args:
            logger: Optional structured logging sink.
        """
        self.logger: Optional[ILogger] = logger

    def measure(self, artifact: "DeploymentArtifact") -> SizeBenchmarkScore:
        """Reduce ``artifact`` to a
        :class:`~uaqe.benchmark.benchmark_result.SizeBenchmarkScore`.

        Args:
            artifact: The ``DeploymentArtifact`` under benchmark.

        Returns:
            The resulting :class:`~uaqe.benchmark.benchmark_result.
            SizeBenchmarkScore`.
        """
        score = SizeBenchmarkScore(
            artifact_size_bytes=artifact.size_bytes,
            export_format=artifact.export_format.value,
            target_profile_id=artifact.target_profile_id,
        )

        if self.logger is not None:
            self.logger.info(
                "Size benchmarked.",
                artifact_size_bytes=score.artifact_size_bytes,
                export_format=score.export_format,
                target_profile_id=score.target_profile_id,
            )

        return score
