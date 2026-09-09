"""The terminal pipeline stage — renders and writes a completed run's
full report bundle.

``ReportGenerator`` is the locked ``uaqe.domain.reporting.
report_generator.ReportGenerator`` (``03_API_Specification.md`` §13.1,
``09_Architecture_Lock.md`` §3/§9): registered under context key
``\"report_generator\"``, the last stage in the canonical 19-stage
pipeline order (``09_Architecture_Lock.md`` §12 —
``... -> deployment_readiness_scorer -> optimization_advisor ->
report_generator``), returning ``List[ReportDocument]`` as its
``StageResult.payload`` (``09_Architecture_Lock.md`` §9's locked return
type table).

Per the same "location deviation only" precedent
:mod:`uaqe.report.report_document` already documents for
``ReportDocument`` itself, this module lives at
``uaqe.report.report_generator`` (this codebase's established flat
top-level-package layout) rather than nested under
``uaqe.domain.reporting`` — the class name, constructor signature, and
``execute() -> StageResult`` contract are otherwise unchanged from the
locked specification.

Per ``09_Architecture_Lock.md`` §13 rule 4, ``ReportGenerator`` is one
of only two components (the other being ``Exporter``) permitted to
write to ``reports/``/``outputs/``. The doc's original design routes
that write through an injected ``FilesystemRepository``
(``uaqe.infrastructure.filesystem.filesystem_repository``); this
package instead writes directly via the standard library, the same
deviation :mod:`uaqe.exporter.artifact_manifest` and
:mod:`uaqe.exporter.package_builder` already make from that same
original design (see those modules' own ``open(...)``/``os`` usage) —
``uaqe.domain``/this package's own top-level layer MUST NOT import
``uaqe.infrastructure`` (``09_Architecture_Lock.md`` §8 rule 1), so a
concrete ``FilesystemRepository`` cannot be constructed or type-hinted
here regardless; :meth:`ReportGenerator._write` reproduces only the
minimal "create parent directories, then write text or bytes" behavior
``FilesystemRepository``/``FileManager`` already provide, scoped to
this one call site.

Rendering (via each renderer supplied at construction, selected by
:class:`~uaqe.report.report_planner.ReportPlanner`) and writing are
both individually best-effort: one renderer raising, or one document
failing to write, is recorded as a warning and does not prevent the
remaining renderers/documents in the same call from completing —
mirroring the "a report must always be producible from a partial run,
it never raises" posture
:func:`~uaqe.report.report_result.build_report_result` already
establishes for the rest of this package. ``StageResult.success`` is
``True`` only if every selected renderer both rendered and (when
writing is enabled) wrote without error.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

from uaqe.common.exceptions import UAQEError
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.interfaces.i_report_renderer import IReportRenderer
from uaqe.common.result_types import StageResult
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.domain.pipeline_stage import PipelineStage
from uaqe.reports.report_document import ReportDocument
from uaqe.reports.report_planner import ReportConfig, ReportPlanner

#: Default base directory this stage writes rendered documents under,
#: matching ``config.json``'s ``reports_path`` default
#: (``06_Config_Spec.md`` §1).
DEFAULT_OUTPUT_DIR: str = "reports"


class ReportGenerator(PipelineStage):
    """Renders every enabled report format for a completed run and
    writes each to ``reports/<run_id>/...``.

    Attributes:
        _renderers: Every ``IReportRenderer`` this stage was
            constructed with (first-party or plugin, per
            ``PluginRegistry.report_renderers``); which of these
            actually run a given call is decided by ``_planner``, not
            hardcoded here.
        _logger: Structured logging sink.
        _planner: Decides which of ``_renderers`` run this call, given
            ``_config``.
        _config: This run's ``ReportConfig`` (``enabled_reports``,
            etc.); a default-valued ``ReportConfig`` is used if none is
            supplied.
        _output_dir: Base directory rendered documents are written
            under; each document's own ``file_path`` (already
            ``<run_id>/<name>.<ext>``, per every renderer in this
            package) is joined onto it.
        _write_to_disk: Whether ``execute()`` writes each rendered
            document to disk at all. ``True`` by default; set
            ``False`` for callers (e.g. a REST entry point streaming
            documents directly to a response body) that only need the
            in-memory ``ReportDocument`` list and do not want
            ``reports/`` touched.
    """

    def __init__(
        self,
        renderers: List[IReportRenderer],
        logger: ILogger,
        planner: Optional[ReportPlanner] = None,
        config: Optional[ReportConfig] = None,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        write_to_disk: bool = True,
    ) -> None:
        """Initialize a ``ReportGenerator``.

        Args:
            renderers: Every registered ``IReportRenderer`` (first-party
                or plugin) available to select from.
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
            planner: The renderer-selection collaborator; a fresh
                :class:`~uaqe.report.report_planner.ReportPlanner` (with
                ``logger`` threaded through) is constructed if omitted.
            config: The run's ``ReportConfig``; a fresh, default-valued
                ``ReportConfig`` is used if omitted.
            output_dir: Base directory rendered documents are written
                under.
            write_to_disk: Whether ``execute()`` writes documents to
                disk. See the attribute docstring above.
        """
        self._renderers: List[IReportRenderer] = list(renderers)
        self._logger: ILogger = logger
        self._planner: ReportPlanner = planner or ReportPlanner(logger=logger)
        self._config: ReportConfig = config or ReportConfig()
        self._output_dir: str = output_dir
        self._write_to_disk: bool = write_to_disk

    def execute(self, context: PipelineContext) -> StageResult:
        """Render (and, unless disabled, write) this run's report bundle.

        Args:
            context: The current run's pipeline context, carrying
                every stage's ``StageResult`` produced so far.

        Returns:
            A ``StageResult`` whose ``payload`` is the
            ``List[ReportDocument]`` produced this call (per
            ``09_Architecture_Lock.md`` §9), and whose ``success`` is
            ``True`` only if every selected renderer both rendered and
            (when ``write_to_disk`` is enabled) wrote without error.
        """
        start = time.monotonic()
        warnings: List[str] = []

        plan = self._planner.build_plan(self._renderers, self._config)
        if plan.skipped_report_types:
            warnings.append(
                f"Skipped (not enabled): {', '.join(plan.skipped_report_types)}."
            )
        if plan.unrecognized_report_types:
            warnings.append(
                "ReportConfig.enabled_reports named report type(s) with no "
                f"matching renderer: {', '.join(plan.unrecognized_report_types)}."
            )
        if plan.duplicate_report_types:
            warnings.append(
                "Ignored duplicate renderer registration(s) for: "
                f"{', '.join(plan.duplicate_report_types)}."
            )

        documents: List[ReportDocument] = []
        all_succeeded = True

        for renderer in plan.selected_renderers:
            report_type = renderer.report_type()
            try:
                document = renderer.render(context)
            except UAQEError as exc:
                all_succeeded = False
                warnings.append(f"[{report_type}] rendering failed: {exc.message}")
                if self._logger is not None:
                    self._logger.error(
                        "Report renderer failed.",
                        report_type=report_type,
                        code=exc.code,
                    )
                continue

            warnings.extend(f"[{report_type}] {w}" for w in document.warnings)
            documents.append(document)

            if not self._write_to_disk:
                continue

            try:
                written_path = self._write(document)
            except UAQEError as exc:
                all_succeeded = False
                warnings.append(
                    f"[{report_type}] failed to write {document.file_path!r}: "
                    f"{exc.message}"
                )
                if self._logger is not None:
                    self._logger.error(
                        "Report write failed.",
                        report_type=report_type,
                        file_path=document.file_path,
                        code=exc.code,
                    )
                continue

            if self._logger is not None:
                self._logger.info(
                    "Report document written.",
                    report_type=report_type,
                    written_path=written_path,
                )

        duration_ms = (time.monotonic() - start) * 1000.0

        if self._logger is not None:
            self._logger.info(
                "Report generation complete.",
                document_count=len(documents),
                success=all_succeeded,
                duration_ms=duration_ms,
            )

        return StageResult(
            stage_name=self.name(),
            success=all_succeeded,
            payload=documents,
            warnings=warnings,
            duration_ms=duration_ms,
        )

    def _write(self, document: ReportDocument) -> str:
        """Write ``document`` to ``<output_dir>/<document.file_path>``.

        Creates parent directories as needed, reproducing the minimal
        subset of ``FilesystemRepository``/``FileManager`` behavior
        this one call site needs — see the module docstring for why
        that concrete infrastructure class cannot be imported here.

        Args:
            document: The rendered document to write. Binary documents
                (``document.is_binary()``) are written from
                ``binary_content``; every other document is written
                from ``content`` as UTF-8 text.

        Returns:
            The absolute filesystem path written to.

        Raises:
            UAQEError: (as ``ConfigurationError``) if the write fails,
                e.g. permission denied or disk full.
        """
        # Imported lazily to avoid importing uaqe.common.exceptions'
        # entire surface at module scope purely for this one
        # write-failure path.
        from uaqe.common.exceptions import ConfigurationError

        target = os.path.join(self._output_dir, document.file_path)
        try:
            parent = os.path.dirname(target)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if document.is_binary():
                with open(target, "wb") as handle:
                    handle.write(document.binary_content or b"")
            else:
                with open(target, "w", encoding="utf-8") as handle:
                    handle.write(document.content)
        except OSError as exc:
            raise ConfigurationError(
                f"Cannot write report document to {target!r}.",
                code="FILESYSTEM_WRITE_FAILED",
                remediation_hint="Verify the destination path is writable and has free space.",
            ) from exc
        return os.path.abspath(target)
