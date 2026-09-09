"""Cross-run comparison of multiple completed runs'
:class:`~uaqe.report.report_result.ReportResult` summaries.

``ComparisonReport`` mirrors
:class:`~uaqe.benchmark.comparison_benchmark.ComparisonBenchmark`'s own
role and reasoning almost exactly, one level up: where
``ComparisonBenchmark`` ranks ``N`` ``BenchmarkResult`` instances for
*the same model* across ``N`` different ``HardwareProfile`` targets,
``ComparisonReport`` ranks ``N`` whole-run ``ReportResult`` summaries
against one another — different models, different hardware targets,
different quantization/compression plans, or simply different
attempts at the same run, compared side by side. Like
``ComparisonBenchmark``, it is a plain collaborator, not a
``PipelineStage``: a single pipeline run produces exactly one
``ReportResult`` (via ``\"report_generator\"``, per
``09_Architecture_Lock.md`` §12's terminal stage); comparing across
``N`` already-completed runs is a ``BatchRunner``-level concern
(``01_Project_Architecture.md`` §18's "horizontal scalability"
fan-out), which calls this class directly over the ``N``
independently-produced ``ReportResult`` instances once every run has
completed. For the same reason, this module deliberately does not
implement ``IReportRenderer`` (whose locked ``render(context) ->
ReportDocument`` signature takes exactly one run's
``PipelineContext`` — there is no single ``PipelineContext`` a
cross-run comparison could be rendered from).

Ranking model: rather than re-deriving a second scoring heuristic,
:meth:`ComparisonReport.compare` reuses
:func:`~uaqe.report.summary_report.summarize`'s existing, already-named
``readiness_score``/``headline_metrics`` for each compared
``ReportResult`` — the same "consume the existing artifact rather than
duplicate its logic" precedent every renderer in this package already
follows for ``ReportResult`` itself. Each entry's ``normalized_score``
is simply ``readiness_score / 100.0`` (already a ``[0, 1]``-normalized,
multi-factor figure per :mod:`uaqe.report.summary_report`'s own
deduction model, so no further per-dimension normalization step is
needed to rank runs against each other); ties are broken by compared-
list order, the same tie-break precedent ``ComparisonBenchmark`` sets.
Beyond the overall ranking, :meth:`ComparisonReport.compare` also
surfaces the same four *per-dimension* winners ``ComparisonBenchmark``
does — fastest, lowest-memory/smallest-artifact, and least accuracy
regression — computed only across the subset of compared runs whose
``ReportResult`` actually populated that dimension (e.g. a run with no
``benchmark`` module cannot contend for "fastest").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from uaqe.common.interfaces.i_logger import ILogger
from uaqe.reports.report_result import ReportResult
from uaqe.reports.summary_report import summarize

#: This module's report-type identifier. Not an ``IReportRenderer``
#: registration (see the module docstring for why), but kept for
#: parity with every other renderer's ``REPORT_TYPE`` constant and for
#: use as this document's own ``report_type``-shaped field.
REPORT_TYPE = "comparison"


@dataclass
class ComparisonEntry:
    """One compared run's rank and scoring detail.

    Deliberately mutable (unlike most value objects in this package),
    mirroring :class:`~uaqe.benchmark.benchmark_result.ComparisonEntry`'s
    own precedent: every entry's ``normalized_score`` is known at
    construction, but ``rank`` is only assigned after every entry in
    the compared set has been built and sorted.

    Attributes:
        run_id: The compared ``ReportResult.run_id``.
        overall_success: Mirrored from the compared ``ReportResult``.
        readiness_score: The compared run's ``[0, 100]``
            deployment-readiness figure, per
            :func:`~uaqe.report.summary_report.summarize`.
        readiness_category: The compared run's ``\"Ready\"``/
            ``\"Marginal\"``/``\"Not Recommended\"`` category, per the
            same function.
        headline_metrics: The compared run's headline metrics, per
            :func:`~uaqe.report.summary_report._headline_metrics`
            (accessed here only through ``summarize``'s public
            result).
        normalized_score: ``readiness_score / 100.0``; the value this
            entry was ranked by.
        rank: This entry's 1-indexed rank within the compared set,
            ``1`` being the highest ``normalized_score``.
    """

    run_id: str = ""
    overall_success: bool = True
    readiness_score: float = 0.0
    readiness_category: str = "Not Recommended"
    headline_metrics: Dict[str, Any] = field(default_factory=dict)
    normalized_score: float = 0.0
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return this entry as a plain, JSON-serializable ``dict``."""
        return {
            "run_id": self.run_id,
            "overall_success": self.overall_success,
            "readiness_score": self.readiness_score,
            "readiness_category": self.readiness_category,
            "headline_metrics": dict(self.headline_metrics),
            "normalized_score": self.normalized_score,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class ComparisonReportDocument:
    """The complete outcome of ranking ``N`` runs against one another.

    Attributes:
        generated_at: The UTC ISO-8601 timestamp this comparison was
            built at.
        entries: One :class:`ComparisonEntry` per compared run, ordered
            by ``rank`` ascending (best first).
        best_overall_run_id: ``entries[0].run_id``, or ``None`` if no
            runs were compared.
        fastest_run_id: The compared run with the lowest
            ``latency_ms_p50``, or ``None`` if no compared run has a
            populated ``benchmark`` module.
        lowest_memory_run_id: The compared run with the lowest
            ``peak_memory_bytes``, or ``None`` for the same reason.
        smallest_artifact_run_id: The compared run with the lowest
            ``artifact_size_bytes``, or ``None`` if no compared run has
            a populated ``export`` module.
        least_regression_run_id: The compared run with the highest
            (least negative) ``accuracy_delta``, or ``None`` if no
            compared run has a populated ``evaluation`` module.
        rationale: A short, human-readable summary of the comparison.
    """

    generated_at: str = ""
    entries: List[ComparisonEntry] = field(default_factory=list)
    best_overall_run_id: Optional[str] = None
    fastest_run_id: Optional[str] = None
    lowest_memory_run_id: Optional[str] = None
    smallest_artifact_run_id: Optional[str] = None
    least_regression_run_id: Optional[str] = None
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return this document as a plain, JSON-serializable ``dict``."""
        return {
            "generated_at": self.generated_at,
            "entries": [entry.to_dict() for entry in self.entries],
            "best_overall_run_id": self.best_overall_run_id,
            "fastest_run_id": self.fastest_run_id,
            "lowest_memory_run_id": self.lowest_memory_run_id,
            "smallest_artifact_run_id": self.smallest_artifact_run_id,
            "least_regression_run_id": self.least_regression_run_id,
            "rationale": self.rationale,
        }

    def to_markdown(self) -> str:
        """Render this document as a Markdown comparison table.

        Returns:
            A Markdown string: a ranked table of every compared run
            plus the per-dimension winners and rationale.
        """
        lines: List[str] = [
            "# Run Comparison",
            "",
            f"- **Generated at:** {self.generated_at}",
            f"- **Runs compared:** {len(self.entries)}",
        ]
        if self.best_overall_run_id is not None:
            lines.append(f"- **Best overall:** {self.best_overall_run_id}")
        if self.fastest_run_id is not None:
            lines.append(f"- **Fastest:** {self.fastest_run_id}")
        if self.lowest_memory_run_id is not None:
            lines.append(f"- **Lowest peak memory:** {self.lowest_memory_run_id}")
        if self.smallest_artifact_run_id is not None:
            lines.append(f"- **Smallest artifact:** {self.smallest_artifact_run_id}")
        if self.least_regression_run_id is not None:
            lines.append(
                f"- **Least accuracy regression:** {self.least_regression_run_id}"
            )

        if self.entries:
            lines.extend(
                [
                    "",
                    "## Ranking",
                    "",
                    "| Rank | Run ID | Readiness | Score | Result |",
                    "|---|---|---|---|---|",
                ]
            )
            for entry in self.entries:
                result_label = "PASSED" if entry.overall_success else "FAILED"
                lines.append(
                    f"| {entry.rank} | {entry.run_id} | "
                    f"{entry.readiness_category} ({entry.readiness_score:.0f}/100) | "
                    f"{entry.normalized_score:.4f} | {result_label} |"
                )

        if self.rationale:
            lines.extend(["", "## Rationale", "", self.rationale])

        return "\n".join(lines) + "\n"


class ComparisonReport:
    """Ranks ``N`` ``ReportResult`` instances against one another.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            comparison operates silently.
    """

    def __init__(self, logger: Optional[ILogger] = None) -> None:
        """Initialize a ``ComparisonReport``.

        Args:
            logger: Optional structured logging sink.
        """
        self.logger: Optional[ILogger] = logger

    def compare(self, results: List[ReportResult]) -> ComparisonReportDocument:
        """Rank ``results`` against one another.

        Args:
            results: One already-produced ``ReportResult`` per compared
                run (e.g. gathered by a ``BatchRunner`` across several
                completed ``PipelineOrchestrator`` runs).

        Returns:
            The resulting :class:`ComparisonReportDocument`; an empty,
            rationale-only document if ``results`` is empty.
        """
        generated_at = datetime.now(timezone.utc).isoformat()

        if not results:
            return ComparisonReportDocument(
                generated_at=generated_at,
                rationale="No ReportResult supplied; nothing to compare.",
            )

        summaries = [summarize(result) for result in results]

        entries = [
            ComparisonEntry(
                run_id=result.run_id,
                overall_success=result.overall_success,
                readiness_score=summary.readiness_score,
                readiness_category=summary.readiness_category,
                headline_metrics=dict(summary.headline_metrics),
                normalized_score=summary.readiness_score / 100.0,
            )
            for result, summary in zip(results, summaries)
        ]
        entries.sort(key=lambda entry: entry.normalized_score, reverse=True)
        for rank, entry in enumerate(entries, start=1):
            entry.rank = rank

        fastest_run_id = self._best_run_id(
            results, key=lambda r: r.benchmark.latency_ms_p50 if r.benchmark else None
        )
        lowest_memory_run_id = self._best_run_id(
            results,
            key=lambda r: r.benchmark.peak_memory_bytes if r.benchmark else None,
        )
        smallest_artifact_run_id = self._best_run_id(
            results, key=lambda r: r.export.size_bytes if r.export else None
        )
        least_regression_run_id = self._best_run_id(
            results,
            key=lambda r: (
                -r.evaluation.accuracy_delta if r.evaluation is not None else None
            ),
        )

        best_overall = entries[0]

        rationale = (
            f"Ranked {len(results)} run(s) by deployment-readiness score "
            "(each run's own summary_report.summarize() readiness_score, "
            f"normalized to [0, 1]); {best_overall.run_id!r} ranked first "
            f"with normalized_score={best_overall.normalized_score:.4f}."
        )

        if self.logger is not None:
            self.logger.info(
                "Report comparison complete.",
                compared_run_count=len(results),
                best_overall_run_id=best_overall.run_id,
            )

        return ComparisonReportDocument(
            generated_at=generated_at,
            entries=entries,
            best_overall_run_id=best_overall.run_id,
            fastest_run_id=fastest_run_id,
            lowest_memory_run_id=lowest_memory_run_id,
            smallest_artifact_run_id=smallest_artifact_run_id,
            least_regression_run_id=least_regression_run_id,
            rationale=rationale,
        )

    @staticmethod
    def _best_run_id(
        results: List[ReportResult], *, key: Callable[[ReportResult], Optional[float]]
    ) -> Optional[str]:
        """Return the ``run_id`` of the lowest-``key`` entry in
        ``results``, ignoring any entry ``key`` maps to ``None``.

        Args:
            results: The compared runs.
            key: A callable mapping a ``ReportResult`` to a
                lower-is-better numeric value, or ``None`` if that run
                does not populate the relevant module.

        Returns:
            The winning run's ``run_id``, or ``None`` if no compared
            run populates the relevant module.
        """
        candidates = [(key(result), result.run_id) for result in results]
        candidates = [c for c in candidates if c[0] is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda c: c[0])[1]
