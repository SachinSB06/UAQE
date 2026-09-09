"""Qualitative and quantitative model-complexity scoring for an
analyzed ``IMR``.

Aggregates the outputs of ``ParameterCounter``, ``FlopsEstimator``, and
``MemoryAnalyzer`` into a single complexity classification consumed by
``AnalysisReport`` (``analysis_report.py``) and, downstream, by
``OptimizationAdvisor``/``DeploymentReadinessScorer`` when reasoning
about how aggressively a model likely needs to be quantized/compressed
for a given hardware class. Internal helper for ``uaqe.analyzer``; not
part of the locked class inventory (``09_Architecture_Lock.md`` §11).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# (upper bound, exclusive, on parameter_count) -> classification label,
# checked in ascending order; parameter counts at or above the last
# entry's bound fall through to _CATCH_ALL_CLASSIFICATION.
_PARAMETER_COUNT_THRESHOLDS: Tuple[Tuple[int, str], ...] = (
    (100_000, "tiny"),
    (1_000_000, "small"),
    (10_000_000, "medium"),
    (100_000_000, "large"),
)
_CATCH_ALL_CLASSIFICATION = "extra_large"


@dataclass(frozen=True)
class ComplexityProfile:
    """The aggregate complexity summary for one analyzed ``IMR``.

    Attributes:
        classification: A coarse size label (``"tiny"``, ``"small"``,
            ``"medium"``, ``"large"``, or ``"extra_large"``) derived
            from ``parameter_count``.
        flops_per_parameter: ``estimated_flops / parameter_count``
            (``0.0`` if ``parameter_count`` is ``0``), a rough proxy
            for how compute- (vs. memory-) bound the model is.
        bytes_per_parameter: ``estimated_memory_bytes / parameter_count``
            (``0.0`` if ``parameter_count`` is ``0``), reflecting the
            average memory overhead per parameter.
        graph_depth: The longest dependency chain length, as computed
            by ``LayerAnalyzer.graph_depth``.
    """

    classification: str
    flops_per_parameter: float
    bytes_per_parameter: float
    graph_depth: int


class ComplexityAnalyzer:
    """Classifies an already-computed set of analysis metrics by complexity."""

    def classify_by_parameter_count(self, parameter_count: int) -> str:
        """Map a total parameter count to a coarse size classification.

        Args:
            parameter_count: The model's total learnable-parameter
                element count, e.g. from ``ParameterCounter.count_total``.

        Returns:
            One of ``"tiny"``, ``"small"``, ``"medium"``, ``"large"``,
            or ``"extra_large"``.
        """
        for upper_bound, label in _PARAMETER_COUNT_THRESHOLDS:
            if parameter_count < upper_bound:
                return label
        return _CATCH_ALL_CLASSIFICATION

    def build_profile(
        self,
        parameter_count: int,
        estimated_flops: int,
        estimated_memory_bytes: int,
        graph_depth: int,
    ) -> ComplexityProfile:
        """Aggregate already-computed metrics into a ``ComplexityProfile``.

        Args:
            parameter_count: Total parameter element count.
            estimated_flops: Total estimated FLOPs.
            estimated_memory_bytes: Total estimated memory footprint,
                in bytes.
            graph_depth: The layer graph's longest dependency chain
                length.

        Returns:
            The aggregated ``ComplexityProfile``.
        """
        flops_per_parameter = (
            estimated_flops / parameter_count if parameter_count else 0.0
        )
        bytes_per_parameter = (
            estimated_memory_bytes / parameter_count if parameter_count else 0.0
        )
        return ComplexityProfile(
            classification=self.classify_by_parameter_count(parameter_count),
            flops_per_parameter=flops_per_parameter,
            bytes_per_parameter=bytes_per_parameter,
            graph_depth=graph_depth,
        )
