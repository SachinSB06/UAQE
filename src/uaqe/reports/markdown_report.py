"""Markdown rendering of a run's :class:`~uaqe.report.report_result.ReportResult`.

``MarkdownReportRenderer`` is the reference renderer for this package:
:func:`render_markdown` is the single place the run's narrative
structure (title, run summary, per-module sections, warnings) is
defined, and :mod:`uaqe.report.html_report` and
:mod:`uaqe.report.pdf_report` both build on its section ordering
(HTML re-renders the same sections as marked-up HTML; PDF paginates
the same section text as plain lines) so the three human-readable
formats never drift out of sync with one another.
"""

from __future__ import annotations

from typing import List

from uaqe.common.interfaces.i_report_renderer import IReportRenderer
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.reports.report_document import ReportDocument
from uaqe.reports.report_result import ReportResult, build_report_result

#: This renderer's report-type identifier, returned by
#: :meth:`MarkdownReportRenderer.report_type`.
REPORT_TYPE = "markdown"


def render_markdown(result: ReportResult) -> str:
    """Render ``result`` into a complete Markdown document.

    Args:
        result: The run summary to render.

    Returns:
        A Markdown string covering the run's overall status, model
        and hardware summary, every module section present in
        ``result``, the stage-execution table, and any accumulated
        warnings.
    """
    lines: List[str] = [
        "# Universal AI Quantization Engine — Run Report",
        "",
        f"- **Run ID:** {result.run_id}",
        f"- **Generated at:** {result.generated_at}",
        f"- **Overall result:** {'PASSED' if result.overall_success else 'FAILED'}",
    ]

    if result.model is not None:
        lines.extend(
            [
                "",
                "## Model",
                "",
                f"- **Parameters:** {result.model.parameter_count:,}",
                f"- **Estimated FLOPs:** {result.model.estimated_flops:,}",
                f"- **Estimated memory:** {result.model.estimated_memory_bytes:,} bytes",
            ]
        )
        if result.model.op_type_histogram:
            lines.extend(["", "| Op type | Count |", "|---|---|"])
            for op_type in sorted(result.model.op_type_histogram):
                lines.append(f"| {op_type} | {result.model.op_type_histogram[op_type]} |")

    if result.hardware is not None:
        lines.extend(
            [
                "",
                "## Hardware Target",
                "",
                f"- **Profile:** {result.hardware.display_name} "
                f"(`{result.hardware.profile_id}`)",
                f"- **Class:** {result.hardware.hardware_class}",
                f"- **Runtime:** {result.hardware.runtime}",
                f"- **Tensor memory budget:** {result.hardware.tensor_memory_bytes:,} bytes",
                f"- **Max model size:** {result.hardware.max_model_size_bytes:,} bytes",
            ]
        )

    if result.quantization is not None:
        lines.extend(
            [
                "",
                "## Quantization",
                "",
                f"- **Strategy:** {result.quantization.selected_strategy_name}",
            ]
        )
        if result.quantization.precision_distribution:
            lines.extend(["", "| Precision | Layer count |", "|---|---|"])
            for precision in sorted(result.quantization.precision_distribution):
                lines.append(
                    f"| {precision} | "
                    f"{result.quantization.precision_distribution[precision]} |"
                )

    if result.compression is not None:
        lines.extend(
            [
                "",
                "## Compression",
                "",
                f"- **Applied types:** "
                f"{', '.join(result.compression.selected_types) or 'none'}",
                f"- **Target ratio:** {result.compression.target_ratio:.4f}",
                f"- **Rationale:** {result.compression.rationale}",
            ]
        )

    if result.optimization is not None:
        lines.extend(
            [
                "",
                "## Optimization",
                "",
                f"- **Candidates evaluated:** {result.optimization.candidate_count}",
                f"- **Pareto front size:** {result.optimization.pareto_front_size}",
                f"- **Rationale:** {result.optimization.rationale}",
            ]
        )
        if result.optimization.objective_scores:
            lines.extend(["", "| Objective | Score |", "|---|---|"])
            for objective in sorted(result.optimization.objective_scores):
                lines.append(
                    f"| {objective} | "
                    f"{result.optimization.objective_scores[objective]:.4f} |"
                )

    if result.memory_plan is not None:
        lines.extend(
            [
                "",
                "## Memory Plan",
                "",
                f"- **Arena size:** {result.memory_plan.buffer_arena_bytes:,} bytes",
                f"- **Peak usage:** {result.memory_plan.peak_memory_bytes:,} bytes",
                f"- **Static weights:** {result.memory_plan.static_weight_bytes:,} bytes",
            ]
        )

    if result.export is not None:
        lines.extend(
            [
                "",
                "## Export",
                "",
                f"- **Target:** {result.export.target_profile_id}",
                f"- **Format:** {result.export.export_format}",
                f"- **Artifact size:** {result.export.size_bytes:,} bytes",
                f"- **Files:** {', '.join(result.export.file_paths) or 'none'}",
            ]
        )

    if result.evaluation is not None:
        evaluation = result.evaluation
        lines.extend(
            [
                "",
                "## Evaluation",
                "",
                f"- **Result:** {'PASSED' if evaluation.passed else 'FAILED'}",
                f"- **Accuracy delta:** {evaluation.accuracy_delta:+.4f}",
            ]
        )
        if evaluation.latency is not None:
            lines.append(f"- **Speedup:** {evaluation.latency.speedup_ratio:.2f}x")
        if evaluation.size is not None:
            lines.append(
                f"- **Compression ratio:** {evaluation.size.compression_ratio:.2f}x"
            )

    if result.benchmark is not None:
        benchmark = result.benchmark
        lines.extend(
            [
                "",
                "## Benchmark",
                "",
                f"- **Hardware:** {benchmark.hardware_profile_id} "
                f"({'real device' if benchmark.used_real_hardware else 'simulated'})",
                f"- **Latency (p50 / p99):** "
                f"{benchmark.latency_ms_p50:.2f} ms / {benchmark.latency_ms_p99:.2f} ms",
                f"- **Throughput:** "
                f"{benchmark.throughput_inferences_per_sec:.2f} inferences/sec",
                f"- **Peak memory:** {benchmark.peak_memory_bytes:,} bytes",
            ]
        )

    if result.stage_summaries:
        lines.extend(
            [
                "",
                "## Stage Execution",
                "",
                "| Stage | Status | Duration (ms) | Warnings |",
                "|---|---|---|---|",
            ]
        )
        for stage in result.stage_summaries:
            status = "OK" if stage.success else "FAILED"
            lines.append(
                f"| {stage.stage_name} | {status} | {stage.duration_ms:.2f} | "
                f"{stage.warning_count} |"
            )

    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)

    return "\n".join(lines) + "\n"


class MarkdownReportRenderer(IReportRenderer):
    """Renders a run's ``PipelineContext`` into a Markdown
    :class:`~uaqe.report.report_document.ReportDocument`.

    Formally implements ``IReportRenderer``
    (``uaqe.common.interfaces.i_report_renderer``) — see
    :mod:`uaqe.report.report_document`'s module docstring for why this
    package is the first to do so.
    """

    def render(self, context: PipelineContext) -> ReportDocument:
        """Render ``context`` into a Markdown ``ReportDocument``.

        Args:
            context: The pipeline context of the run being reported on.

        Returns:
            The rendered ``ReportDocument``, with ``content`` set to
            the full Markdown report and ``file_path`` set to
            ``reports/<run_id>/report.md``.
        """
        result = build_report_result(context)
        content = render_markdown(result)
        return ReportDocument(
            report_type=REPORT_TYPE,
            content=content,
            file_path=f"{result.run_id}/report.md",
            mime_type="text/markdown",
            run_id=result.run_id,
            warnings=list(result.warnings),
        )

    def report_type(self) -> str:
        """Return this renderer's unique registration name."""
        return REPORT_TYPE
