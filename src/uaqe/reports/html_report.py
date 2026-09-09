"""HTML rendering of a run's :class:`~uaqe.report.report_result.ReportResult`.

``HtmlReportRenderer`` renders the exact same section set and ordering
as :mod:`uaqe.report.markdown_report` (see that module's docstring for
why the human-readable renderers are kept structurally in sync), as a
single self-contained, dependency-free HTML document — no external
Markdown-to-HTML conversion library, no external stylesheet or script
reference, so the rendered file is viewable standalone (e.g. attached
to an email or opened from a local ``reports/<run_id>/`` directory
without network access).
"""

from __future__ import annotations

from html import escape
from typing import List

from uaqe.common.interfaces.i_report_renderer import IReportRenderer
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.reports.report_document import ReportDocument
from uaqe.reports.report_result import ReportResult, build_report_result

#: This renderer's report-type identifier, returned by
#: :meth:`HtmlReportRenderer.report_type`.
REPORT_TYPE = "html"

_STYLE = """
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
       max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
h1 { border-bottom: 2px solid #333; padding-bottom: 0.3rem; }
h2 { border-bottom: 1px solid #ccc; padding-bottom: 0.2rem; margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; margin: 0.75rem 0; }
th, td { border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }
th { background-color: #f2f2f2; }
.status-pass { color: #0a7d28; font-weight: bold; }
.status-fail { color: #b00020; font-weight: bold; }
ul.warnings { color: #8a6d00; }
code { background-color: #f4f4f4; padding: 0.1rem 0.3rem; border-radius: 3px; }
"""


def _status_span(passed: bool) -> str:
    """Render a pass/fail label as a styled ``<span>``.

    Args:
        passed: Whether the outcome being labeled succeeded.

    Returns:
        A ``<span>`` element with the appropriate status class.
    """
    label = "PASSED" if passed else "FAILED"
    css_class = "status-pass" if passed else "status-fail"
    return f'<span class="{css_class}">{label}</span>'


def _kv_list(pairs: List[str]) -> str:
    """Render a list of pre-formatted ``key: value`` strings as an
    unordered list.

    Args:
        pairs: Already-escaped ``\"Label: value\"`` strings.

    Returns:
        A ``<ul>`` HTML fragment.
    """
    items = "".join(f"<li>{pair}</li>" for pair in pairs)
    return f"<ul>{items}</ul>"


def _table(headers: List[str], rows: List[List[str]]) -> str:
    """Render a simple HTML table.

    Args:
        headers: Column headers, already escaped.
        rows: Row cell values, already escaped.

    Returns:
        A ``<table>`` HTML fragment, or ``\"\"`` if ``rows`` is empty.
    """
    if not rows:
        return ""
    header_html = "".join(f"<th>{h}</th>" for h in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>"


def render_html(result: ReportResult) -> str:
    """Render ``result`` into a complete, self-contained HTML document.

    Args:
        result: The run summary to render.

    Returns:
        A full ``<html>...</html>`` document string.
    """
    sections: List[str] = []

    sections.append(
        "<h1>Universal AI Quantization Engine &mdash; Run Report</h1>"
        + _kv_list(
            [
                f"<strong>Run ID:</strong> <code>{escape(result.run_id)}</code>",
                f"<strong>Generated at:</strong> {escape(result.generated_at)}",
                f"<strong>Overall result:</strong> "
                f"{_status_span(result.overall_success)}",
            ]
        )
    )

    if result.model is not None:
        rows = [
            [escape(op_type), str(count)]
            for op_type, count in sorted(result.model.op_type_histogram.items())
        ]
        sections.append(
            "<h2>Model</h2>"
            + _kv_list(
                [
                    f"<strong>Parameters:</strong> {result.model.parameter_count:,}",
                    f"<strong>Estimated FLOPs:</strong> "
                    f"{result.model.estimated_flops:,}",
                    f"<strong>Estimated memory:</strong> "
                    f"{result.model.estimated_memory_bytes:,} bytes",
                ]
            )
            + _table(["Op type", "Count"], rows)
        )

    if result.hardware is not None:
        sections.append(
            "<h2>Hardware Target</h2>"
            + _kv_list(
                [
                    f"<strong>Profile:</strong> {escape(result.hardware.display_name)} "
                    f"(<code>{escape(result.hardware.profile_id)}</code>)",
                    f"<strong>Class:</strong> {escape(result.hardware.hardware_class)}",
                    f"<strong>Runtime:</strong> {escape(result.hardware.runtime)}",
                    f"<strong>Tensor memory budget:</strong> "
                    f"{result.hardware.tensor_memory_bytes:,} bytes",
                    f"<strong>Max model size:</strong> "
                    f"{result.hardware.max_model_size_bytes:,} bytes",
                ]
            )
        )

    if result.quantization is not None:
        rows = [
            [escape(precision), str(count)]
            for precision, count in sorted(
                result.quantization.precision_distribution.items()
            )
        ]
        sections.append(
            "<h2>Quantization</h2>"
            + _kv_list(
                [
                    f"<strong>Strategy:</strong> "
                    f"{escape(result.quantization.selected_strategy_name)}"
                ]
            )
            + _table(["Precision", "Layer count"], rows)
        )

    if result.compression is not None:
        sections.append(
            "<h2>Compression</h2>"
            + _kv_list(
                [
                    f"<strong>Applied types:</strong> "
                    f"{escape(', '.join(result.compression.selected_types) or 'none')}",
                    f"<strong>Target ratio:</strong> "
                    f"{result.compression.target_ratio:.4f}",
                    f"<strong>Rationale:</strong> "
                    f"{escape(result.compression.rationale)}",
                ]
            )
        )

    if result.optimization is not None:
        rows = [
            [escape(objective), f"{score:.4f}"]
            for objective, score in sorted(result.optimization.objective_scores.items())
        ]
        sections.append(
            "<h2>Optimization</h2>"
            + _kv_list(
                [
                    f"<strong>Candidates evaluated:</strong> "
                    f"{result.optimization.candidate_count}",
                    f"<strong>Pareto front size:</strong> "
                    f"{result.optimization.pareto_front_size}",
                    f"<strong>Rationale:</strong> "
                    f"{escape(result.optimization.rationale)}",
                ]
            )
            + _table(["Objective", "Score"], rows)
        )

    if result.memory_plan is not None:
        sections.append(
            "<h2>Memory Plan</h2>"
            + _kv_list(
                [
                    f"<strong>Arena size:</strong> "
                    f"{result.memory_plan.buffer_arena_bytes:,} bytes",
                    f"<strong>Peak usage:</strong> "
                    f"{result.memory_plan.peak_memory_bytes:,} bytes",
                    f"<strong>Static weights:</strong> "
                    f"{result.memory_plan.static_weight_bytes:,} bytes",
                ]
            )
        )

    if result.export is not None:
        sections.append(
            "<h2>Export</h2>"
            + _kv_list(
                [
                    f"<strong>Target:</strong> "
                    f"{escape(result.export.target_profile_id)}",
                    f"<strong>Format:</strong> {escape(result.export.export_format)}",
                    f"<strong>Artifact size:</strong> "
                    f"{result.export.size_bytes:,} bytes",
                    f"<strong>Files:</strong> "
                    f"{escape(', '.join(result.export.file_paths) or 'none')}",
                ]
            )
        )

    if result.evaluation is not None:
        evaluation = result.evaluation
        items = [
            f"<strong>Result:</strong> {_status_span(evaluation.passed)}",
            f"<strong>Accuracy delta:</strong> {evaluation.accuracy_delta:+.4f}",
        ]
        if evaluation.latency is not None:
            items.append(
                f"<strong>Speedup:</strong> {evaluation.latency.speedup_ratio:.2f}x"
            )
        if evaluation.size is not None:
            items.append(
                f"<strong>Compression ratio:</strong> "
                f"{evaluation.size.compression_ratio:.2f}x"
            )
        sections.append("<h2>Evaluation</h2>" + _kv_list(items))

    if result.benchmark is not None:
        benchmark = result.benchmark
        sections.append(
            "<h2>Benchmark</h2>"
            + _kv_list(
                [
                    f"<strong>Hardware:</strong> "
                    f"{escape(benchmark.hardware_profile_id)} "
                    f"({'real device' if benchmark.used_real_hardware else 'simulated'})",
                    f"<strong>Latency (p50 / p99):</strong> "
                    f"{benchmark.latency_ms_p50:.2f} ms / "
                    f"{benchmark.latency_ms_p99:.2f} ms",
                    f"<strong>Throughput:</strong> "
                    f"{benchmark.throughput_inferences_per_sec:.2f} inferences/sec",
                    f"<strong>Peak memory:</strong> "
                    f"{benchmark.peak_memory_bytes:,} bytes",
                ]
            )
        )

    if result.stage_summaries:
        rows = [
            [
                escape(stage.stage_name),
                _status_span(stage.success),
                f"{stage.duration_ms:.2f}",
                str(stage.warning_count),
            ]
            for stage in result.stage_summaries
        ]
        sections.append(
            "<h2>Stage Execution</h2>"
            + _table(["Stage", "Status", "Duration (ms)", "Warnings"], rows)
        )

    if result.warnings:
        items = "".join(f"<li>{escape(w)}</li>" for w in result.warnings)
        sections.append(f'<h2>Warnings</h2><ul class="warnings">{items}</ul>')

    body = "\n".join(sections)
    title = f"UAQE Run Report — {escape(result.run_id)}"
    return (
        "<!DOCTYPE html>"
        f'<html lang="en"><head><meta charset="utf-8">'
        f"<title>{title}</title><style>{_STYLE}</style></head>"
        f"<body>{body}</body></html>"
    )


class HtmlReportRenderer(IReportRenderer):
    """Renders a run's ``PipelineContext`` into an HTML
    :class:`~uaqe.report.report_document.ReportDocument`.

    Formally implements ``IReportRenderer`` — see
    :mod:`uaqe.report.report_document`'s module docstring.
    """

    def render(self, context: PipelineContext) -> ReportDocument:
        """Render ``context`` into an HTML ``ReportDocument``.

        Args:
            context: The pipeline context of the run being reported on.

        Returns:
            The rendered ``ReportDocument``, with ``content`` set to
            the full HTML document and ``file_path`` set to
            ``reports/<run_id>/report.html``.
        """
        result = build_report_result(context)
        content = render_html(result)
        return ReportDocument(
            report_type=REPORT_TYPE,
            content=content,
            file_path=f"{result.run_id}/report.html",
            mime_type="text/html",
            run_id=result.run_id,
            warnings=list(result.warnings),
        )

    def report_type(self) -> str:
        """Return this renderer's unique registration name."""
        return REPORT_TYPE
