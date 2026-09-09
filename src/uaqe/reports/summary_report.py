"""Condensed, decision-oriented summary of a run —
:class:`~uaqe.report.report_result.ReportResult` distilled down to a
headline readiness score, category, and a short list of actionable
recommendations.

This is this codebase's counterpart to the locked "Summary Report"
(``04_Data_Flow.md`` §12, sourced from "all prior ``StageResult``s")
and to the readiness/advisory role ``09_Architecture_Lock.md`` §3
describes for ``uaqe.domain.advisory`` (``ReadinessScore``,
``DeploymentReadinessScorer``, ``Recommendation``,
``OptimizationAdvisor``) — none of which exist as implemented packages
in this codebase yet. Rather than depend on packages that do not
exist, :class:`SummaryReportRenderer` computes a self-contained,
transparent readiness heuristic directly from the same
``ReportResult`` every other renderer in this package already
consumes; every deduction it makes is named in
:class:`SummaryReportDocument.readiness_notes`, so the score is always
traceable back to a concrete cause rather than an opaque number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from uaqe.common.interfaces.i_report_renderer import IReportRenderer
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.reports.report_document import ReportDocument
from uaqe.reports.report_result import ReportResult, build_report_result

#: This renderer's report-type identifier, returned by
#: :meth:`SummaryReportRenderer.report_type`.
REPORT_TYPE = "summary"

#: Readiness score thresholds, inclusive lower bounds, mapped to a
#: human-readable category — mirroring the three-tier
#: ``\"Ready\"``/``\"Marginal\"``/``\"Not Recommended\"`` categorization
#: named for ``ReadinessScore.category`` in
#: ``03_API_Specification.md`` §12.1.
_READY_THRESHOLD = 80.0
_MARGINAL_THRESHOLD = 50.0

#: Score deductions applied by :func:`_score_readiness`, named so every
#: deduction's magnitude is documented in exactly one place.
_DEDUCTION_STAGE_FAILURE = 15.0
_DEDUCTION_EVALUATION_FAILED = 25.0
_DEDUCTION_ACCURACY_REGRESSION = 15.0
_DEDUCTION_INCOMPATIBLE = 30.0
_DEDUCTION_MEMORY_OVER_BUDGET = 20.0
_DEDUCTION_MODEL_OVER_BUDGET = 20.0
_DEDUCTION_PER_WARNING = 1.0
_MAX_WARNING_DEDUCTION = 10.0


@dataclass(frozen=True)
class SummaryReportDocument:
    """A condensed, decision-oriented view of one run.

    Attributes:
        run_id: The run this summary covers.
        generated_at: The UTC ISO-8601 timestamp this summary was
            built at.
        overall_success: Mirrored from ``ReportResult.overall_success``.
        readiness_score: A ``[0, 100]`` deployment-readiness figure,
            per :func:`_score_readiness`.
        readiness_category: ``\"Ready\"`` (``>= 80``), ``\"Marginal\"``
            (``>= 50``), or ``\"Not Recommended\"`` (``< 50``).
        readiness_notes: Human-readable notes on every deduction
            applied while computing ``readiness_score``.
        headline_metrics: The single most decision-relevant figure from
            each populated module (accuracy delta, compression ratio,
            latency, artifact size, and so on), keyed by metric name.
        recommendations: Short, actionable follow-up suggestions
            derived from ``headline_metrics``/``readiness_notes``.
        warnings: Every warning carried through from the underlying
            ``ReportResult``.
    """

    run_id: str = ""
    generated_at: str = ""
    overall_success: bool = True
    readiness_score: float = 100.0
    readiness_category: str = "Ready"
    readiness_notes: List[str] = field(default_factory=list)
    headline_metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return this document as a plain, JSON-serializable ``dict``."""
        return {
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "overall_success": self.overall_success,
            "readiness_score": self.readiness_score,
            "readiness_category": self.readiness_category,
            "readiness_notes": list(self.readiness_notes),
            "headline_metrics": dict(self.headline_metrics),
            "recommendations": list(self.recommendations),
            "warnings": list(self.warnings),
        }

    def to_markdown(self) -> str:
        """Render this document as a short Markdown summary.

        Returns:
            A Markdown string: readiness verdict, headline metrics
            table, recommendations, and readiness notes.
        """
        lines: List[str] = [
            "# Run Summary",
            "",
            f"- **Run ID:** {self.run_id}",
            f"- **Readiness:** {self.readiness_category} "
            f"({self.readiness_score:.0f}/100)",
            f"- **Overall result:** {'PASSED' if self.overall_success else 'FAILED'}",
        ]

        if self.headline_metrics:
            lines.extend(["", "## Headline Metrics", "", "| Metric | Value |", "|---|---|"])
            for metric_name in sorted(self.headline_metrics):
                lines.append(f"| {metric_name} | {self.headline_metrics[metric_name]} |")

        if self.recommendations:
            lines.extend(["", "## Recommendations", ""])
            lines.extend(f"- {rec}" for rec in self.recommendations)

        if self.readiness_notes:
            lines.extend(["", "## Readiness Notes", ""])
            lines.extend(f"- {note}" for note in self.readiness_notes)

        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)

        return "\n".join(lines) + "\n"


def _score_readiness(result: ReportResult) -> "tuple[float, List[str]]":
    """Compute a transparent ``[0, 100]`` readiness score for ``result``.

    Every deduction applied is appended to the returned notes list, so
    the final score is always traceable to a concrete, named cause
    rather than an opaque figure.

    Args:
        result: The run summary to score.

    Returns:
        The ``(score, notes)`` pair.
    """
    score = 100.0
    notes: List[str] = []

    failed_stages = [s.stage_name for s in result.stage_summaries if not s.success]
    if failed_stages:
        deduction = _DEDUCTION_STAGE_FAILURE * len(failed_stages)
        score -= deduction
        notes.append(
            f"-{deduction:.0f}: {len(failed_stages)} stage(s) failed: "
            f"{', '.join(failed_stages)}."
        )

    if result.evaluation is not None:
        if not result.evaluation.passed:
            score -= _DEDUCTION_EVALUATION_FAILED
            notes.append(f"-{_DEDUCTION_EVALUATION_FAILED:.0f}: evaluation did not pass.")
        if result.evaluation.accuracy_delta < -0.01:
            score -= _DEDUCTION_ACCURACY_REGRESSION
            notes.append(
                f"-{_DEDUCTION_ACCURACY_REGRESSION:.0f}: accuracy regressed by "
                f"{-result.evaluation.accuracy_delta:.4f}."
            )
        if (
            result.evaluation.compatibility is not None
            and not result.evaluation.compatibility.compatible
        ):
            score -= _DEDUCTION_INCOMPATIBLE
            notes.append(
                f"-{_DEDUCTION_INCOMPATIBLE:.0f}: final compatibility check failed."
            )

    if result.memory_plan is not None and result.hardware is not None:
        if result.memory_plan.peak_memory_bytes > result.hardware.tensor_memory_bytes:
            score -= _DEDUCTION_MEMORY_OVER_BUDGET
            notes.append(
                f"-{_DEDUCTION_MEMORY_OVER_BUDGET:.0f}: peak activation memory "
                f"({result.memory_plan.peak_memory_bytes:,} bytes) exceeds the "
                f"target's tensor memory budget "
                f"({result.hardware.tensor_memory_bytes:,} bytes)."
            )

    if result.export is not None and result.hardware is not None:
        if result.export.size_bytes > result.hardware.max_model_size_bytes > 0:
            score -= _DEDUCTION_MODEL_OVER_BUDGET
            notes.append(
                f"-{_DEDUCTION_MODEL_OVER_BUDGET:.0f}: exported artifact "
                f"({result.export.size_bytes:,} bytes) exceeds the target's "
                f"maximum model size "
                f"({result.hardware.max_model_size_bytes:,} bytes)."
            )

    if result.warnings:
        warning_deduction = min(
            len(result.warnings) * _DEDUCTION_PER_WARNING, _MAX_WARNING_DEDUCTION
        )
        score -= warning_deduction
        notes.append(
            f"-{warning_deduction:.0f}: {len(result.warnings)} accumulated warning(s)."
        )

    score = max(0.0, min(100.0, score))
    if not notes:
        notes.append("No deductions; every checked signal was within range.")
    return score, notes


def _categorize(score: float) -> str:
    """Map a readiness score to its category label.

    Args:
        score: A ``[0, 100]`` readiness score.

    Returns:
        ``\"Ready\"``, ``\"Marginal\"``, or ``\"Not Recommended\"``.
    """
    if score >= _READY_THRESHOLD:
        return "Ready"
    if score >= _MARGINAL_THRESHOLD:
        return "Marginal"
    return "Not Recommended"


def _headline_metrics(result: ReportResult) -> Dict[str, Any]:
    """Extract the single most decision-relevant figure from each
    populated module in ``result``.

    Args:
        result: The run summary to extract headline metrics from.

    Returns:
        A flat ``{metric_name: value}`` mapping.
    """
    metrics: Dict[str, Any] = {}
    if result.model is not None:
        metrics["parameter_count"] = result.model.parameter_count
    if result.compression is not None:
        metrics["compression_target_ratio"] = result.compression.target_ratio
    if result.evaluation is not None:
        metrics["accuracy_delta"] = round(result.evaluation.accuracy_delta, 4)
        if result.evaluation.size is not None:
            metrics["compression_ratio"] = round(
                result.evaluation.size.compression_ratio, 4
            )
    if result.export is not None:
        metrics["artifact_size_bytes"] = result.export.size_bytes
    if result.benchmark is not None:
        metrics["latency_ms_p50"] = round(result.benchmark.latency_ms_p50, 3)
        metrics["throughput_inferences_per_sec"] = round(
            result.benchmark.throughput_inferences_per_sec, 3
        )
    return metrics


def _recommendations(result: ReportResult, notes: List[str]) -> List[str]:
    """Derive short, actionable recommendations from ``result`` and the
    readiness deduction notes already computed for it.

    Args:
        result: The run summary being summarized.
        notes: The readiness notes from :func:`_score_readiness`.

    Returns:
        Zero or more short, human-readable recommendation strings.
    """
    recommendations: List[str] = []

    if any("stage(s) failed" in note for note in notes):
        recommendations.append(
            "Re-run the pipeline after addressing the failed stage(s) above; "
            "downstream results were computed against a degraded run."
        )
    if any("accuracy regressed" in note for note in notes):
        recommendations.append(
            "Consider a lower-aggressiveness quantization/compression plan, or "
            "raise QuantizationConfig.sensitivity_threshold, to recover accuracy."
        )
    if any("tensor memory budget" in note for note in notes):
        recommendations.append(
            "Peak activation memory exceeds the target's budget; enable or "
            "tighten MemoryOptimizer's tensor-reuse planning, or select a "
            "hardware profile with a larger tensor memory budget."
        )
    if any("maximum model size" in note for note in notes):
        recommendations.append(
            "Exported artifact exceeds the target's maximum model size; "
            "increase CompressionConfig.target_ratio or select a different "
            "precision plan."
        )
    if any("compatibility check failed" in note for note in notes):
        recommendations.append(
            "Resolve the flagged compatibility violations before deploying "
            "this artifact to the target hardware."
        )
    if not recommendations and result.overall_success:
        recommendations.append(
            "No corrective action identified; this run is ready for deployment "
            "review."
        )
    return recommendations


def summarize(result: ReportResult) -> SummaryReportDocument:
    """Build a :class:`SummaryReportDocument` from a completed
    :class:`~uaqe.report.report_result.ReportResult`.

    Args:
        result: The run summary to condense.

    Returns:
        The rendered :class:`SummaryReportDocument`.
    """
    score, notes = _score_readiness(result)
    return SummaryReportDocument(
        run_id=result.run_id,
        generated_at=result.generated_at,
        overall_success=result.overall_success,
        readiness_score=score,
        readiness_category=_categorize(score),
        readiness_notes=notes,
        headline_metrics=_headline_metrics(result),
        recommendations=_recommendations(result, notes),
        warnings=list(result.warnings),
    )


class SummaryReportRenderer(IReportRenderer):
    """Renders a run's ``PipelineContext`` into a condensed summary
    :class:`~uaqe.report.report_document.ReportDocument`.

    Formally implements ``IReportRenderer`` — see
    :mod:`uaqe.report.report_document`'s module docstring.
    """

    def render(self, context: PipelineContext) -> ReportDocument:
        """Render ``context`` into a summary ``ReportDocument``.

        Args:
            context: The pipeline context of the run being reported on.

        Returns:
            The rendered ``ReportDocument``, with ``content`` set to
            the summary's Markdown rendering and ``file_path`` set to
            ``reports/<run_id>/summary.md``.
        """
        result = build_report_result(context)
        document = summarize(result)
        return ReportDocument(
            report_type=REPORT_TYPE,
            content=document.to_markdown(),
            file_path=f"{result.run_id}/summary.md",
            mime_type="text/markdown",
            run_id=result.run_id,
            warnings=list(document.warnings),
        )

    def report_type(self) -> str:
        """Return this renderer's unique registration name."""
        return REPORT_TYPE
