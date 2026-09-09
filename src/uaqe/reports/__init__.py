"""``uaqe.report`` — assembles and renders a completed run's report
bundle, for the Universal AI Quantization Engine.

This package covers the same responsibilities as the locked
``uaqe.domain.reporting`` module described in
``03_API_Specification.md`` §13 and ``09_Architecture_Lock.md`` §3/§9/
§12 (aggregate every stage's ``StageResult`` into one summary, render
it into every enabled output format, write the results to
``reports/<run_id>/``), decomposed into nine single-responsibility
files rather than the doc's one — following the same precedent
``uaqe.quantization``, ``uaqe.compression``, ``uaqe.optimizer``, and
``uaqe.exporter`` each already set for their own locked counterparts
(see those packages' ``__init__`` docstrings). Per the same "location
deviation only" note ``uaqe.report.report_document``'s own docstring
documents, this package's flat ``uaqe.report.*`` layout replaces the
doc's originally nested ``uaqe.domain.reporting.*`` path; every locked
name, signature, and contract is otherwise unchanged.

Pipeline (run in this order, entirely within the terminal
``\"report_generator\"`` ``PipelineStage``):

1. :func:`~uaqe.report.report_result.build_report_result` flattens the
   run's ``PipelineContext`` into one renderer-agnostic
   :class:`~uaqe.report.report_result.ReportResult` — the single place
   "what did this run produce" is answered, consumed by every renderer
   below rather than each re-reading ``PipelineContext`` itself.
2. :class:`~uaqe.report.report_planner.ReportPlanner` decides which of
   the renderers supplied to :class:`~uaqe.report.report_generator.
   ReportGenerator` actually run this call, against the run's
   :class:`~uaqe.report.report_planner.ReportConfig`
   (``enabled_reports``, mirroring ``reports.json``).
3. :class:`~uaqe.report.report_generator.ReportGenerator` — the sole
   orchestrating stage, registered under context key
   ``\"report_generator\"``. Invokes each selected renderer's
   ``render(context) -> ReportDocument`` and writes the result to
   ``reports/<run_id>/...``.

Renderers (each an ``IReportRenderer``, invoked by ``ReportGenerator``
per the plan built above — never called directly outside of
composition-root wiring or a batch/comparison caller):

- :class:`~uaqe.report.markdown_report.MarkdownReportRenderer` —
  ``\"markdown\"``: the reference renderer; every other human-readable
  format's section ordering follows it exactly.
- :class:`~uaqe.report.html_report.HtmlReportRenderer` — ``\"html\"``:
  the same sections, re-rendered as one self-contained HTML document.
- :class:`~uaqe.report.json_report.JsonReportRenderer` — ``\"json\"``:
  ``ReportResult.to_dict()``, pretty-printed.
- :class:`~uaqe.report.csv_report.CsvReportRenderer` — ``\"csv\"``:
  ``ReportResult`` flattened to one ``section,metric,value`` row per
  scalar metric.
- :class:`~uaqe.report.pdf_report.PdfReportRenderer` — ``\"pdf\"``: the
  same section text, hand-paginated into a minimal, dependency-free
  PDF 1.4 document (see that module's docstring for why no PDF library
  dependency was added).
- :class:`~uaqe.report.summary_report.SummaryReportRenderer` —
  ``\"summary\"``: a condensed, decision-oriented readiness score,
  category, and short recommendation list, distilled from the same
  ``ReportResult``.

Cross-run comparison:

- :class:`~uaqe.report.comparison_report.ComparisonReport` ranks ``N``
  already-completed runs' ``ReportResult`` summaries against one
  another. Like :class:`~uaqe.benchmark.comparison_benchmark.
  ComparisonBenchmark`, it is a plain collaborator, not a
  ``PipelineStage`` or an ``IReportRenderer`` — a single pipeline run
  produces exactly one ``ReportResult``; comparing across many
  completed runs is a ``BatchRunner``-level concern that calls this
  class directly once every compared run has completed.

Per ``09_Architecture_Lock.md`` §8, this package depends only on
``uaqe.common`` and ``uaqe.domain`` (for ``PipelineStage``,
``PipelineContext``) — never on ``uaqe.infrastructure`` or
``uaqe.interface``. Every ``IReportRenderer`` implementation here is
registered with ``uaqe.infrastructure.plugins.PluginRegistry`` the same
way a third-party plugin renderer would be
(``10_Module_Development_Guide.md`` §18), so ``ReportGenerator``
depends on them only through the ``IReportRenderer`` port supplied to
its constructor, never by importing a concrete renderer class
directly.
"""

from uaqe.reports.comparison_report import (
    REPORT_TYPE as COMPARISON_REPORT_TYPE,
    ComparisonEntry,
    ComparisonReport,
    ComparisonReportDocument,
)
from uaqe.reports.csv_report import CsvReportRenderer
from uaqe.reports.html_report import HtmlReportRenderer
from uaqe.reports.json_report import JsonReportRenderer
from uaqe.reports.markdown_report import MarkdownReportRenderer
from uaqe.reports.pdf_report import PdfReportRenderer
from uaqe.reports.report_document import ReportDocument
from uaqe.reports.report_generator import ReportGenerator
from uaqe.reports.report_planner import ReportConfig, ReportPlan, ReportPlanner
from uaqe.reports.report_result import (
    CompressionSummary,
    ExportSummary,
    HardwareSummary,
    MemoryPlanSummary,
    ModelSummary,
    OptimizationSummary,
    QuantizationSummary,
    ReportResult,
    StageExecutionSummary,
    build_report_result,
    resolve_run_id,
)
from uaqe.reports.summary_report import (
    SummaryReportDocument,
    SummaryReportRenderer,
    summarize,
)

__all__ = [
    # report_document.py
    "ReportDocument",
    # report_result.py
    "ReportResult",
    "ModelSummary",
    "HardwareSummary",
    "QuantizationSummary",
    "CompressionSummary",
    "OptimizationSummary",
    "MemoryPlanSummary",
    "ExportSummary",
    "StageExecutionSummary",
    "build_report_result",
    "resolve_run_id",
    # report_planner.py
    "ReportPlanner",
    "ReportPlan",
    "ReportConfig",
    # report_generator.py
    "ReportGenerator",
    # markdown_report.py
    "MarkdownReportRenderer",
    # html_report.py
    "HtmlReportRenderer",
    # json_report.py
    "JsonReportRenderer",
    # csv_report.py
    "CsvReportRenderer",
    # pdf_report.py
    "PdfReportRenderer",
    # summary_report.py
    "SummaryReportRenderer",
    "SummaryReportDocument",
    "summarize",
    # comparison_report.py
    "ComparisonReport",
    "ComparisonReportDocument",
    "ComparisonEntry",
    "COMPARISON_REPORT_TYPE",
]
