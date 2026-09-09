"""Decides which registered :class:`~uaqe.common.interfaces.i_report_renderer.
IReportRenderer` instances a single run's
:class:`~uaqe.report.report_generator.ReportGenerator` should invoke.

``ReportPlanner`` is an internal collaborator of ``ReportGenerator`` —
the same role :class:`~uaqe.benchmark.benchmark_planner.BenchmarkPlanner`
plays for :class:`~uaqe.benchmark.benchmark.Benchmarker` and
:class:`~uaqe.evaluation.evaluation_planner.EvaluationPlanner` plays
for :class:`~uaqe.evaluation.evaluator.Evaluator`: it does not itself
read ``PipelineContext``, render anything, or write a file;
``ReportGenerator`` owns that. Its one job is :meth:`ReportPlanner.
build_plan`, which resolves *which* of the renderers supplied to
``ReportGenerator`` actually run this call, against the run's
:class:`ReportConfig`, before any renderer's ``render()`` is invoked.

``ReportConfig`` mirrors ``reports.json`` (``06_Config_Spec.md`` §7,
locked field-for-field by ``09_Architecture_Lock.md`` §5:
``schema_version``, ``enabled_reports``, ``output_format``,
``include_raw_metrics_appendix``) — the same "config dataclass mirrors
its JSON file field-for-field" precedent
:class:`~uaqe.benchmark.benchmark_planner.BenchmarkConfig` already
sets for ``benchmark.json``.

Deviation note on ``enabled_reports``: ``06_Config_Spec.md`` §7's
example value is ``[\"accuracy\", \"benchmark\", \"compression\",
\"deployment\", \"summary\"]`` — subsystem-shaped names from the
original, undecomposed reporting design. This package instead ships
one renderer per *output format* (:mod:`uaqe.report.markdown_report`,
:mod:`uaqe.report.html_report`, :mod:`uaqe.report.json_report`,
:mod:`uaqe.report.csv_report`, :mod:`uaqe.report.pdf_report`,
:mod:`uaqe.report.summary_report`), each registering under its own
``IReportRenderer.report_type()`` string (``\"markdown\"``,
``\"html\"``, ``\"json\"``, ``\"csv\"``, ``\"pdf\"``, ``\"summary\"``) —
the same "decomposed package, format-shaped rather than doc-shaped"
precedent every other package in this codebase already sets for its
own locked counterpart (see e.g. :mod:`uaqe.exporter`'s and
:mod:`uaqe.compression`'s own ``__init__`` docstrings). ``enabled_reports``
therefore now names *this* package's ``report_type()`` values rather
than the doc's subsystem names; :data:`DEFAULT_ENABLED_REPORTS` is the
practical default for that realized schema. ``output_format`` is
carried unmodified for schema fidelity with the locked
``reports.json``, but is otherwise informational here: unlike the
doc's original one-renderer-many-subsystems design, this package's
renderers are already one-per-format, so no renderer needs a second
"which format do I emit" switch of its own.

Every entry in ``enabled_reports`` that names a ``report_type()`` no
renderer supplied to :meth:`ReportPlanner.build_plan` actually
registers is recorded on the returned :class:`ReportPlan` as an
``unrecognized_report_types`` entry (and warned on) rather than
raising — mirroring the "a report must always be producible, never
raise" defensive posture :func:`~uaqe.report.report_result.
build_report_result` already establishes for this package. Likewise, a
supplied renderer whose type is simply not enabled this run is not an
error; it is recorded as ``skipped_report_types``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.interfaces.i_report_renderer import IReportRenderer

#: ``reports.json``'s locked schema version (``06_Config_Spec.md`` §7).
DEFAULT_SCHEMA_VERSION: str = "1.0"

#: Practical default for ``ReportConfig.enabled_reports`` given this
#: package's realized, format-shaped ``report_type()`` values — see the
#: module docstring's deviation note. Every first-party single-run
#: renderer except ``\"pdf\"`` is enabled by default (PDF rendering is
#: comparatively expensive and most deployments only need it on
#: request), matching the "conservative, opt-in for the heavier format"
#: precedent already set elsewhere in this codebase (e.g.
#: ``Exporter``'s packaging step being independently skippable).
DEFAULT_ENABLED_REPORTS: tuple = ("markdown", "html", "json", "csv", "summary")

#: ``reports.json``'s locked default ``output_format``.
DEFAULT_OUTPUT_FORMAT: str = "markdown"

#: ``reports.json``'s locked default ``include_raw_metrics_appendix``.
DEFAULT_INCLUDE_RAW_METRICS_APPENDIX: bool = True


@dataclass(frozen=True)
class ReportConfig:
    """Configuration governing which reports :class:`ReportPlanner`
    selects, mirroring ``reports.json`` (``06_Config_Spec.md`` §7)
    field-for-field.

    Attributes:
        schema_version: The config schema version this instance was
            constructed against; carried through for parity with the
            locked JSON schema, not otherwise interpreted by this
            module.
        enabled_reports: The ``report_type()`` values that should run
            this call. A renderer supplied to :meth:`ReportPlanner.
            build_plan` whose ``report_type()`` is not in this list is
            skipped; an entry here with no matching supplied renderer
            is recorded as unrecognized rather than raising.
        output_format: Carried unmodified from the locked schema; see
            the module docstring's deviation note for why this
            package's renderers do not each need to consult it.
        include_raw_metrics_appendix: Whether renderers should append a
            raw ``StageResult.payload``-derived appendix after their
            human-readable summary. This module does not itself act on
            the flag (no renderer supplied to :meth:`ReportPlanner.
            build_plan` currently reads it); it is threaded through
            unchanged so a future renderer can opt into it without a
            second config surface.
    """

    schema_version: str = DEFAULT_SCHEMA_VERSION
    enabled_reports: List[str] = field(
        default_factory=lambda: list(DEFAULT_ENABLED_REPORTS)
    )
    output_format: str = DEFAULT_OUTPUT_FORMAT
    include_raw_metrics_appendix: bool = DEFAULT_INCLUDE_RAW_METRICS_APPENDIX

    def to_dict(self) -> Dict[str, Any]:
        """Return this config as a plain, JSON-serializable ``dict``."""
        return {
            "schema_version": self.schema_version,
            "enabled_reports": list(self.enabled_reports),
            "output_format": self.output_format,
            "include_raw_metrics_appendix": self.include_raw_metrics_appendix,
        }


@dataclass(frozen=True)
class ReportPlan:
    """Which renderers a single :class:`~uaqe.report.report_generator.
    ReportGenerator` call should invoke, and why.

    Attributes:
        selected_renderers: The renderers to actually invoke this call,
            in the same relative order they were supplied to
            :meth:`ReportPlanner.build_plan` (i.e. ``enabled_reports``
            controls *membership*, not ordering).
        selected_report_types: ``report_type()`` of every entry in
            ``selected_renderers``, for convenient logging/inspection
            without re-deriving it.
        skipped_report_types: ``report_type()`` values of renderers
            that were supplied but not enabled this run.
        unrecognized_report_types: Entries of ``ReportConfig.
            enabled_reports`` that matched no supplied renderer's
            ``report_type()``.
        duplicate_report_types: ``report_type()`` values for which more
            than one supplied renderer registered; only the first
            encountered is kept in ``selected_renderers``/considered
            for selection, the rest are dropped.
        rationale: A short, human-readable summary of this plan.
    """

    selected_renderers: List[IReportRenderer] = field(default_factory=list)
    selected_report_types: List[str] = field(default_factory=list)
    skipped_report_types: List[str] = field(default_factory=list)
    unrecognized_report_types: List[str] = field(default_factory=list)
    duplicate_report_types: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return this plan's metadata as a plain, JSON-serializable
        ``dict`` (``selected_renderers`` themselves are not
        serializable and are represented only by
        ``selected_report_types``).
        """
        return {
            "selected_report_types": list(self.selected_report_types),
            "skipped_report_types": list(self.skipped_report_types),
            "unrecognized_report_types": list(self.unrecognized_report_types),
            "duplicate_report_types": list(self.duplicate_report_types),
            "rationale": self.rationale,
        }


class ReportPlanner:
    """Resolves a :class:`ReportPlan` from a set of available renderers
    and a :class:`ReportConfig`.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            planner operates silently.
    """

    def __init__(self, logger: Optional[ILogger] = None) -> None:
        """Initialize a ``ReportPlanner``.

        Args:
            logger: Optional structured logging sink.
        """
        self.logger: Optional[ILogger] = logger

    def build_plan(
        self,
        available_renderers: List[IReportRenderer],
        config: Optional[ReportConfig] = None,
    ) -> ReportPlan:
        """Resolve which of ``available_renderers`` should run this call.

        Args:
            available_renderers: Every ``IReportRenderer`` the calling
                ``ReportGenerator`` was constructed with, in supplied
                order.
            config: The run's ``ReportConfig``; a fresh, default-valued
                ``ReportConfig`` is used if omitted (per the same
                "optional config, sensible default" precedent
                ``BenchmarkConfig``/``Benchmarker`` already set).

        Returns:
            The resolved :class:`ReportPlan`. Never raises: an empty
            ``available_renderers`` list or an ``enabled_reports`` list
            matching nothing simply yields an empty, still-valid plan.
        """
        resolved_config = config or ReportConfig()
        enabled = set(resolved_config.enabled_reports)

        selected: List[IReportRenderer] = []
        selected_types: List[str] = []
        skipped_types: List[str] = []
        duplicate_types: List[str] = []
        seen_types: set = set()

        for renderer in available_renderers:
            report_type = renderer.report_type()
            if report_type in seen_types:
                duplicate_types.append(report_type)
                continue
            seen_types.add(report_type)

            if report_type in enabled:
                selected.append(renderer)
                selected_types.append(report_type)
            else:
                skipped_types.append(report_type)

        unrecognized_types = sorted(enabled - seen_types)

        rationale = (
            f"Selected {len(selected_types)}/{len(available_renderers)} "
            f"supplied renderer(s) ({', '.join(selected_types) or 'none'}) "
            f"per ReportConfig.enabled_reports."
        )
        if skipped_types:
            rationale += f" Skipped (not enabled): {', '.join(skipped_types)}."
        if unrecognized_types:
            rationale += (
                f" No supplied renderer matched: {', '.join(unrecognized_types)}."
            )
        if duplicate_types:
            rationale += (
                f" Ignored duplicate registration(s) for: "
                f"{', '.join(duplicate_types)}."
            )

        if self.logger is not None:
            self.logger.info(
                "Report plan built.",
                selected_report_types=selected_types,
                skipped_report_types=skipped_types,
                unrecognized_report_types=unrecognized_types,
                duplicate_report_types=duplicate_types,
            )

        return ReportPlan(
            selected_renderers=selected,
            selected_report_types=selected_types,
            skipped_report_types=skipped_types,
            unrecognized_report_types=unrecognized_types,
            duplicate_report_types=duplicate_types,
            rationale=rationale,
        )
