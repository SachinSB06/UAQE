"""CSV rendering of a run's :class:`~uaqe.report.report_result.ReportResult`.

Unlike the narrative Markdown/HTML formats or the fully structured JSON
format, CSV has no native way to represent this result's nested shape
(a plan, a hardware profile, an evaluation result, and so on, each with
their own fields). ``CsvReportRenderer`` therefore flattens
``ReportResult`` into a single ``metric,value`` table — one row per
scalar metric across every populated section — which is the
representation most spreadsheet tools and downstream analytics
pipelines expect from a CSV export, and mirrors the same
flatten-to-rows approach ``uaqe.benchmark.comparison_benchmark``
already uses for its own ranking tables.
"""

from __future__ import annotations

import csv
import io
from typing import List, Tuple

from uaqe.common.interfaces.i_report_renderer import IReportRenderer
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.reports.report_document import ReportDocument
from uaqe.reports.report_result import ReportResult, build_report_result

#: This renderer's report-type identifier, returned by
#: :meth:`CsvReportRenderer.report_type`.
REPORT_TYPE = "csv"

#: The fixed column header row every document from this renderer opens
#: with.
_HEADER = ("section", "metric", "value")


def _flatten(result: ReportResult) -> List[Tuple[str, str, str]]:
    """Flatten ``result`` into ``(section, metric, value)`` rows.

    Args:
        result: The run summary to flatten.

    Returns:
        One row per scalar metric, grouped by section in the same
        order as :func:`~uaqe.report.markdown_report.render_markdown`'s
        sections.
    """
    rows: List[Tuple[str, str, str]] = [
        ("run", "run_id", result.run_id),
        ("run", "generated_at", result.generated_at),
        ("run", "overall_success", str(result.overall_success)),
    ]

    if result.model is not None:
        rows.extend(
            [
                ("model", "parameter_count", str(result.model.parameter_count)),
                ("model", "estimated_flops", str(result.model.estimated_flops)),
                (
                    "model",
                    "estimated_memory_bytes",
                    str(result.model.estimated_memory_bytes),
                ),
            ]
        )

    if result.hardware is not None:
        rows.extend(
            [
                ("hardware", "profile_id", result.hardware.profile_id),
                ("hardware", "hardware_class", result.hardware.hardware_class),
                ("hardware", "runtime", result.hardware.runtime),
                (
                    "hardware",
                    "tensor_memory_bytes",
                    str(result.hardware.tensor_memory_bytes),
                ),
                (
                    "hardware",
                    "max_model_size_bytes",
                    str(result.hardware.max_model_size_bytes),
                ),
            ]
        )

    if result.quantization is not None:
        rows.append(
            (
                "quantization",
                "selected_strategy_name",
                result.quantization.selected_strategy_name,
            )
        )
        for precision, count in sorted(
            result.quantization.precision_distribution.items()
        ):
            rows.append(("quantization", f"precision_count.{precision}", str(count)))

    if result.compression is not None:
        rows.extend(
            [
                (
                    "compression",
                    "selected_types",
                    "|".join(result.compression.selected_types),
                ),
                ("compression", "target_ratio", f"{result.compression.target_ratio:.6f}"),
            ]
        )

    if result.optimization is not None:
        rows.extend(
            [
                (
                    "optimization",
                    "candidate_count",
                    str(result.optimization.candidate_count),
                ),
                (
                    "optimization",
                    "pareto_front_size",
                    str(result.optimization.pareto_front_size),
                ),
            ]
        )
        for objective, score in sorted(result.optimization.objective_scores.items()):
            rows.append(("optimization", f"objective_score.{objective}", f"{score:.6f}"))

    if result.memory_plan is not None:
        rows.extend(
            [
                (
                    "memory_plan",
                    "buffer_arena_bytes",
                    str(result.memory_plan.buffer_arena_bytes),
                ),
                (
                    "memory_plan",
                    "peak_memory_bytes",
                    str(result.memory_plan.peak_memory_bytes),
                ),
                (
                    "memory_plan",
                    "static_weight_bytes",
                    str(result.memory_plan.static_weight_bytes),
                ),
            ]
        )

    if result.export is not None:
        rows.extend(
            [
                ("export", "target_profile_id", result.export.target_profile_id),
                ("export", "export_format", result.export.export_format),
                ("export", "size_bytes", str(result.export.size_bytes)),
            ]
        )

    if result.evaluation is not None:
        evaluation = result.evaluation
        rows.extend(
            [
                ("evaluation", "passed", str(evaluation.passed)),
                ("evaluation", "accuracy_delta", f"{evaluation.accuracy_delta:.6f}"),
            ]
        )
        if evaluation.latency is not None:
            rows.append(
                ("evaluation", "speedup_ratio", f"{evaluation.latency.speedup_ratio:.6f}")
            )
        if evaluation.size is not None:
            rows.append(
                (
                    "evaluation",
                    "compression_ratio",
                    f"{evaluation.size.compression_ratio:.6f}",
                )
            )

    if result.benchmark is not None:
        benchmark = result.benchmark
        rows.extend(
            [
                ("benchmark", "hardware_profile_id", benchmark.hardware_profile_id),
                ("benchmark", "latency_ms_p50", f"{benchmark.latency_ms_p50:.6f}"),
                ("benchmark", "latency_ms_p99", f"{benchmark.latency_ms_p99:.6f}"),
                (
                    "benchmark",
                    "throughput_inferences_per_sec",
                    f"{benchmark.throughput_inferences_per_sec:.6f}",
                ),
                ("benchmark", "peak_memory_bytes", str(benchmark.peak_memory_bytes)),
            ]
        )

    for stage in result.stage_summaries:
        rows.append(("stage", f"{stage.stage_name}.success", str(stage.success)))
        rows.append(
            ("stage", f"{stage.stage_name}.duration_ms", f"{stage.duration_ms:.4f}")
        )

    rows.append(("warnings", "count", str(len(result.warnings))))
    return rows


def render_csv(result: ReportResult) -> str:
    """Render ``result`` into a flat ``section,metric,value`` CSV
    document.

    Args:
        result: The run summary to render.

    Returns:
        A CSV string, header row plus one row per metric, using
        ``\\r\\n`` line endings per RFC 4180 (as produced by the
        standard library ``csv`` module's default dialect).
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_HEADER)
    writer.writerows(_flatten(result))
    return buffer.getvalue()


class CsvReportRenderer(IReportRenderer):
    """Renders a run's ``PipelineContext`` into a CSV
    :class:`~uaqe.report.report_document.ReportDocument`.

    Formally implements ``IReportRenderer`` — see
    :mod:`uaqe.report.report_document`'s module docstring.
    """

    def render(self, context: PipelineContext) -> ReportDocument:
        """Render ``context`` into a CSV ``ReportDocument``.

        Args:
            context: The pipeline context of the run being reported on.

        Returns:
            The rendered ``ReportDocument``, with ``content`` set to
            the full CSV document and ``file_path`` set to
            ``reports/<run_id>/report.csv``.
        """
        result = build_report_result(context)
        content = render_csv(result)
        return ReportDocument(
            report_type=REPORT_TYPE,
            content=content,
            file_path=f"{result.run_id}/report.csv",
            mime_type="text/csv",
            run_id=result.run_id,
            warnings=list(result.warnings),
        )

    def report_type(self) -> str:
        """Return this renderer's unique registration name."""
        return REPORT_TYPE
