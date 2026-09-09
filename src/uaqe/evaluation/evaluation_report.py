"""Human- and machine-readable reporting for a completed evaluation run.

``EvaluationReportRenderer`` summarizes an
:class:`~uaqe.evaluation.evaluation_result.EvaluationResult` into an
:class:`EvaluationReportDocument` — a Markdown- and dict-renderable
summary suitable for ``uaqe.domain.reporting.report_generator.
ReportGenerator`` to fold into a run's overall report bundle, mirroring
:class:`~uaqe.optimizer.optimization_report.OptimizationReportRenderer`'s
and :class:`~uaqe.compression.compression_report.
CompressionReportRenderer`'s role for their own packages.

Like those two renderers, this class is deliberately *structurally*
compatible with ``IReportRenderer`` (``render(...) -> ReportDocument``,
``report_type() -> str``) rather than formally inheriting from it: the
locked ``ReportDocument`` type is not yet implemented anywhere in this
codebase, and this package should not take a hard import dependency on
a module that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from uaqe.evaluation.evaluation_result import EvaluationResult

#: This renderer's report-type identifier, returned by
#: :meth:`EvaluationReportRenderer.report_type`.
REPORT_TYPE = "evaluation"


@dataclass(frozen=True)
class EvaluationReportDocument:
    """A rendered summary of one run's evaluation outcome.

    Attributes:
        passed: Whether the run's evaluation passed overall, mirrored
            from ``EvaluationResult.passed``.
        accuracy_delta: The accuracy dimension's headline figure.
        speedup_ratio: The latency dimension's headline figure, or
            ``None`` if latency was not evaluated this run.
        memory_reduction_ratio: The memory dimension's headline figure,
            or ``None`` if memory was not evaluated this run.
        energy_reduction_ratio: The power dimension's headline figure,
            or ``None`` if power was not evaluated this run.
        compression_ratio: The size dimension's headline figure.
        compatible: The compatibility dimension's headline figure, or
            ``None`` if compatibility was not evaluated this run.
        regressed_metrics: The metric names of every regression finding
            that regressed.
        per_layer_error: The accuracy dimension's per-layer error
            breakdown, carried through unchanged.
        warnings: Every non-fatal warning accumulated while building
            this report.
    """

    passed: bool = True
    accuracy_delta: float = 0.0
    speedup_ratio: Any = None
    memory_reduction_ratio: Any = None
    energy_reduction_ratio: Any = None
    compression_ratio: float = 1.0
    compatible: Any = None
    regressed_metrics: List[str] = field(default_factory=list)
    per_layer_error: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return this document as a plain, JSON-serializable ``dict``."""
        return {
            "passed": self.passed,
            "accuracy_delta": self.accuracy_delta,
            "speedup_ratio": self.speedup_ratio,
            "memory_reduction_ratio": self.memory_reduction_ratio,
            "energy_reduction_ratio": self.energy_reduction_ratio,
            "compression_ratio": self.compression_ratio,
            "compatible": self.compatible,
            "regressed_metrics": list(self.regressed_metrics),
            "per_layer_error": dict(self.per_layer_error),
            "warnings": list(self.warnings),
        }

    def to_markdown(self) -> str:
        """Render this document as a Markdown summary.

        Returns:
            A Markdown string: an overall pass/fail line, one bullet per
            evaluated dimension, the per-layer error table (if
            non-empty), and any warnings/regressions.
        """
        lines: List[str] = [
            "# Evaluation Report",
            "",
            f"- **Overall result:** {'PASSED' if self.passed else 'FAILED'}",
            f"- **Accuracy delta:** {self.accuracy_delta:+.4f}",
        ]
        if self.speedup_ratio is not None:
            lines.append(f"- **Speedup:** {self.speedup_ratio:.2f}x")
        if self.memory_reduction_ratio is not None:
            lines.append(f"- **Memory reduction:** {self.memory_reduction_ratio:.2f}x")
        if self.energy_reduction_ratio is not None:
            lines.append(f"- **Energy reduction:** {self.energy_reduction_ratio:.2f}x")
        lines.append(f"- **Compression ratio:** {self.compression_ratio:.2f}x")
        if self.compatible is not None:
            lines.append(
                f"- **Final compatibility:** {'OK' if self.compatible else 'FAILED'}"
            )

        if self.per_layer_error:
            lines.extend(["", "## Per-Layer Accuracy Error", "", "| Layer | Error |", "|---|---|"])
            for layer_name in sorted(self.per_layer_error):
                lines.append(f"| {layer_name} | {self.per_layer_error[layer_name]:.4f} |")

        if self.regressed_metrics:
            lines.extend(["", "## Regressions", ""])
            lines.extend(f"- {metric_name}" for metric_name in self.regressed_metrics)

        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)

        return "\n".join(lines)


class EvaluationReportRenderer:
    """Builds an :class:`EvaluationReportDocument` from a completed
    evaluation run's result.

    Structurally compatible with ``IReportRenderer`` — see the module
    docstring for why this class does not formally inherit it yet.
    """

    def render(self, result: EvaluationResult) -> EvaluationReportDocument:
        """Render a completed evaluation result into a report document.

        Args:
            result: The finalized
                :class:`~uaqe.evaluation.evaluation_result.
                EvaluationResult` to summarize.

        Returns:
            The rendered :class:`EvaluationReportDocument`.
        """
        regressed_metrics = [
            finding.metric_name for finding in result.regressions if finding.regressed
        ]

        return EvaluationReportDocument(
            passed=result.passed,
            accuracy_delta=result.accuracy_delta,
            speedup_ratio=(
                result.latency.speedup_ratio if result.latency is not None else None
            ),
            memory_reduction_ratio=(
                result.memory.reduction_ratio if result.memory is not None else None
            ),
            energy_reduction_ratio=(
                result.power.energy_reduction_ratio if result.power is not None else None
            ),
            compression_ratio=(
                result.size.compression_ratio if result.size is not None else 1.0
            ),
            compatible=(
                result.compatibility.compatible
                if result.compatibility is not None
                else None
            ),
            regressed_metrics=regressed_metrics,
            per_layer_error=dict(result.per_layer_error),
            warnings=list(result.warnings),
        )

    def report_type(self) -> str:
        """Return this renderer's unique registration name."""
        return REPORT_TYPE
