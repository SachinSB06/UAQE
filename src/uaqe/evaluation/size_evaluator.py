"""Baseline-vs-optimized serialized model size comparison.

Model size is estimated the same way
:class:`~uaqe.hardware.compatibility_checker.CompatibilityChecker`
already does for its ``max_model_size_bytes`` constraint check: the sum
of every layer's parameter tensor raw byte length — see
:meth:`SizeEvaluator._estimate_serialized_size`, kept identical to
``CompatibilityChecker._estimate_serialized_size`` (a lower-bound
estimate that ignores runtime/container framing overhead) so the two
packages never disagree on what "size" means for the same ``IMR``.

When a :class:`~uaqe.exporter.exporter.DeploymentArtifact` from an
already-run ``Exporter`` stage is supplied, its measured
``size_bytes`` is used for ``optimized_size_bytes`` instead of the
static estimate — an actual on-disk artifact size is strictly more
accurate than any pre-export proxy, whenever it is available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.evaluation.evaluation_result import SizeScore

if TYPE_CHECKING:
    from uaqe.exporter.exporter import DeploymentArtifact

class SizeEvaluator:
    """Compares estimated (or actual, post-export) serialized model
    size before and after optimization.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            evaluator operates silently.
    """

    def __init__(self, logger: Optional[ILogger] = None) -> None:
        """Initialize a ``SizeEvaluator``.

        Args:
            logger: Optional structured logging sink.
        """
        self.logger: Optional[ILogger] = logger

    def evaluate(
        self,
        baseline_imr: IMR,
        optimized_imr: IMR,
        artifact: Optional["DeploymentArtifact"] = None,
    ) -> SizeScore:
        """Compare ``baseline_imr`` and ``optimized_imr`` serialized
        size.

        Args:
            baseline_imr: The pre-optimization model.
            optimized_imr: The final, post-optimization model.
            artifact: The ``DeploymentArtifact`` produced by an
                already-run ``Exporter`` stage this run, if any. When
                supplied, ``artifact.size_bytes`` is used for
                ``optimized_size_bytes`` in place of the static
                estimate.

        Returns:
            The resulting :class:`~uaqe.evaluation.evaluation_result.
            SizeScore`.
        """
        baseline_size_bytes = self._estimate_serialized_size(baseline_imr)
        used_exported_artifact = artifact is not None
        optimized_size_bytes = (
            artifact.size_bytes
            if artifact is not None
            else self._estimate_serialized_size(optimized_imr)
        )
        compression_ratio = baseline_size_bytes / max(optimized_size_bytes, 1)

        if self.logger is not None:
            self.logger.info(
                "Size evaluated.",
                baseline_size_bytes=baseline_size_bytes,
                optimized_size_bytes=optimized_size_bytes,
                compression_ratio=compression_ratio,
                used_exported_artifact=used_exported_artifact,
            )

        return SizeScore(
            baseline_size_bytes=baseline_size_bytes,
            optimized_size_bytes=optimized_size_bytes,
            size_delta_bytes=optimized_size_bytes - baseline_size_bytes,
            compression_ratio=compression_ratio,
            used_exported_artifact=used_exported_artifact,
        )

    def _estimate_serialized_size(self, imr: IMR) -> int:
        """Estimate ``imr``'s total serialized size, in bytes.

        Args:
            imr: The model to measure.

        Returns:
            The sum of every layer's parameter tensor raw byte lengths.
        """
        total = 0
        for layer in imr.layers:
            for tensor in layer.parameters.values():
                total += len(tensor.data)
        return total
