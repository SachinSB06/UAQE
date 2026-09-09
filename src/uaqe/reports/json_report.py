"""JSON rendering of a run's :class:`~uaqe.report.report_result.ReportResult`.

The machine-readable counterpart to
:mod:`uaqe.report.markdown_report`/:mod:`uaqe.report.html_report`:
``JsonReportRenderer`` serializes the exact same
:class:`~uaqe.report.report_result.ReportResult` (via its own
``to_dict()``, which every nested dataclass already implements) rather
than re-deriving a parallel field set, so the JSON, Markdown, and HTML
reports for a given run are always describing identical underlying
data.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from uaqe.common.interfaces.i_report_renderer import IReportRenderer
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.reports.report_document import ReportDocument
from uaqe.reports.report_result import ReportResult, build_report_result

#: This renderer's report-type identifier, returned by
#: :meth:`JsonReportRenderer.report_type`.
REPORT_TYPE = "json"

#: The ``json.dumps`` indent width used for every document this
#: renderer produces, fixed so output is diff-friendly across runs.
_INDENT = 2


def render_json(result: ReportResult) -> str:
    """Render ``result`` into a formatted JSON document.

    Args:
        result: The run summary to render.

    Returns:
        A pretty-printed JSON string of ``result.to_dict()``.
    """
    payload: Dict[str, Any] = result.to_dict()
    return json.dumps(payload, indent=_INDENT, sort_keys=True, default=str)


class JsonReportRenderer(IReportRenderer):
    """Renders a run's ``PipelineContext`` into a JSON
    :class:`~uaqe.report.report_document.ReportDocument`.

    Formally implements ``IReportRenderer`` — see
    :mod:`uaqe.report.report_document`'s module docstring.
    """

    def render(self, context: PipelineContext) -> ReportDocument:
        """Render ``context`` into a JSON ``ReportDocument``.

        Args:
            context: The pipeline context of the run being reported on.

        Returns:
            The rendered ``ReportDocument``, with ``content`` set to
            the full JSON document and ``file_path`` set to
            ``reports/<run_id>/report.json``.
        """
        result = build_report_result(context)
        content = render_json(result)
        return ReportDocument(
            report_type=REPORT_TYPE,
            content=content,
            file_path=f"{result.run_id}/report.json",
            mime_type="application/json",
            run_id=result.run_id,
            warnings=list(result.warnings),
        )

    def report_type(self) -> str:
        """Return this renderer's unique registration name."""
        return REPORT_TYPE
