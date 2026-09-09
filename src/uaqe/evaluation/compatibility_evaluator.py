"""Final-checkpoint hardware/layer compatibility re-verification.

``HardwareManager``/``LayerCompatibilityChecker`` already checks
compatibility earlier in the pipeline (``09_Architecture_Lock.md`` §12:
stage 6 of 19), before quantization, compression, and optimization have
run. Per that same locked stage ordering, this package's ``evaluator``
stage runs *after* ``Exporter`` — the last IMR-mutation-adjacent stage
is ``memory_optimizer``, several transformation stages upstream of the
original check. ``CompatibilityEvaluator`` re-runs the identical check
against the *final* IMR so a caller sees whether every intervening
transformation preserved compatibility, without duplicating the check's
logic: it wraps
:class:`~uaqe.hardware.compatibility_checker.CompatibilityChecker`, the
same class ``HardwareManager`` itself composes over.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.evaluation.evaluation_result import CompatibilityScore
from uaqe.hardware.compatibility_checker import CompatibilityChecker

if TYPE_CHECKING:
    from uaqe.domain.hardware_manager import HardwareProfile


class CompatibilityEvaluator:
    """Re-checks a final ``IMR``'s compatibility with a
    ``HardwareProfile``.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            evaluator operates silently.
    """

    def __init__(
        self,
        logger: Optional[ILogger] = None,
        checker: Optional[CompatibilityChecker] = None,
    ) -> None:
        """Initialize a ``CompatibilityEvaluator``.

        Args:
            logger: Optional structured logging sink.
            checker: The underlying compatibility checker to delegate
                to; a fresh
                :class:`~uaqe.hardware.compatibility_checker.
                CompatibilityChecker` is constructed if omitted.
        """
        self.logger: Optional[ILogger] = logger
        self._checker = checker or CompatibilityChecker(logger)

    def evaluate(
        self, optimized_imr: IMR, profile: "HardwareProfile"
    ) -> CompatibilityScore:
        """Re-check ``optimized_imr`` against ``profile``.

        Args:
            optimized_imr: The final, post-optimization model.
            profile: The resolved deployment target.

        Returns:
            The resulting :class:`~uaqe.evaluation.evaluation_result.
            CompatibilityScore`. Never raises: this is a diagnostic
            re-check, so it always runs non-strictly and reports
            ``compatible=False`` rather than propagating
            ``HardwareIncompatibilityError``.
        """
        report = self._checker.check(optimized_imr, profile, strict=False)

        if self.logger is not None:
            self.logger.info(
                "Final compatibility re-checked.",
                compatible=report.compatible,
                unsupported_layer_count=len(report.unsupported_layers),
            )

        return CompatibilityScore(
            compatible=report.compatible,
            unsupported_layers=list(report.unsupported_layers),
            constraint_violations=list(report.constraint_violations),
        )
