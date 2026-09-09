"""Executes an ordered list of ``PipelineStage`` instances against a
``PipelineContext``, producing a ``RunResult``.

Per ``09_Architecture_Lock.md`` §13 rule 5, ``resume()`` is idempotent
per already-populated context entries: a stage never re-executes if
``context.has(stage.name())`` is ``True``. The reaction to a stage
failure is governed by ``PipelineErrorPolicy`` (``ABORT``,
``SKIP_STAGE``, or ``BEST_EFFORT``).

Locked contract: ``03_API_Specification.md`` §2.4.
"""

import time
from typing import List

from uaqe.common.exceptions import ConfigurationError, UAQEError
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.result_types import RunResult, StageResult
from uaqe.common.types import PipelineErrorPolicy
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.domain.pipeline_stage import PipelineStage


class PipelineOrchestrator:
    """Runs a fixed list of stages in order, honoring the configured
    error policy and resume semantics.

    Attributes:
        _stages: The ordered stages to execute, as materialized by
            ``PipelineBuilder.build()``.
        _context: The run's ``PipelineContext``.
        _logger: The logger every stage-boundary event is reported to.
        _error_policy: The policy applied when a stage raises a
            ``UAQEError``.
    """

    def __init__(
        self,
        stages: List[PipelineStage],
        context: PipelineContext,
        logger: ILogger,
        error_policy: PipelineErrorPolicy,
    ) -> None:
        """Initialize the orchestrator for a single run.

        Args:
            stages: The ordered stages to execute.
            context: The run's pipeline context.
            logger: The logger to report stage-boundary events to.
            error_policy: The policy applied when a stage fails.
        """
        self._stages: List[PipelineStage] = stages
        self._context: PipelineContext = context
        self._logger: ILogger = logger
        self._error_policy: PipelineErrorPolicy = error_policy

    def run(self) -> RunResult:
        """Execute every stage in order from a fresh context.

        Returns:
            The ``RunResult`` summarizing this run.
        """
        return self._execute(resuming=False)

    def resume(self, run_id: str) -> RunResult:
        """Resume a previously started run, skipping any stage already
        present in ``_context``.

        Args:
            run_id: The run identifier being resumed. Must match the
                identifier the orchestrator's ``_context`` was created
                with.

        Returns:
            The ``RunResult`` summarizing the resumed run.

        Raises:
            ConfigurationError: If ``run_id`` does not match this
                orchestrator's ``_context``.
        """
        if run_id != self._context._run_id:  # noqa: SLF001 - same-package access
            raise ConfigurationError(
                f"Cannot resume run {run_id!r}: this orchestrator's context "
                f"belongs to run {self._context._run_id!r}.",  # noqa: SLF001
                code="RUN_ID_MISMATCH",
                remediation_hint=(
                    "Construct PipelineOrchestrator with the PipelineContext "
                    "for the run being resumed."
                ),
            )
        return self._execute(resuming=True)

    def _execute(self, resuming: bool) -> RunResult:
        """Run every stage in ``_stages``, honoring resume/error-policy
        semantics.

        Args:
            resuming: Whether already-populated stages should be
                skipped (per ``09_Architecture_Lock.md`` §13 rule 5).

        Returns:
            The ``RunResult`` summarizing the run.
        """
        errors: List[UAQEError] = []
        overall_success = True

        for stage in self._stages:
            stage_name = stage.name()

            if resuming and self._context.has(stage_name):
                self._logger.info(
                    "Skipping already-completed stage on resume.",
                    stage=stage_name,
                )
                continue

            start = time.monotonic()
            try:
                result = stage.execute(self._context)
            except UAQEError as exc:
                import traceback
                import sys
                print("STAGE EXECUTION ERROR DETECTED:", file=sys.stderr)
                traceback.print_exc()
                duration_ms = (time.monotonic() - start) * 1000.0
                self._logger.error(
                    "Stage raised a UAQEError.",
                    stage=stage_name,
                    code=exc.code,
                    remediation_hint=exc.remediation_hint,
                )
                errors.append(exc)
                failed_result = StageResult(
                    stage_name=stage_name,
                    success=False,
                    payload=None,
                    warnings=[],
                    duration_ms=duration_ms,
                )
                self._context.append(stage_name, failed_result)
                overall_success = False

                if self._error_policy is PipelineErrorPolicy.ABORT:
                    break
                elif self._error_policy is PipelineErrorPolicy.SKIP_STAGE:
                    continue
                elif self._error_policy is PipelineErrorPolicy.BEST_EFFORT:
                    continue
                else:
                    break
            else:
                self._context.append(stage_name, result)
                if not result.success:
                    overall_success = False

        return RunResult(
            run_id=self._context._run_id,  # noqa: SLF001 - same-package access
            success=overall_success,
            stage_results=dict(self._context._stage_results),  # noqa: SLF001
            errors=errors,
            output_paths=[],
        )
