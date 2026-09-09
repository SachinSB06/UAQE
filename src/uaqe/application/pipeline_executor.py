"""``PipelineExecutor`` — runs an already-assembled stage sequence
against a :class:`~uaqe.application.workflow_context.WorkflowContext`.

This is the thin application-layer seam between
:class:`~uaqe.application.workflow_controller.WorkflowController` and
:class:`~uaqe.domain.pipeline_orchestrator.PipelineOrchestrator`: it
owns *when* an orchestrator is constructed and invoked (fresh
``run()`` vs. ``resume()`` on an existing context) and *how* the
run's :class:`~uaqe.application.workflow_context.WorkflowContext`
lifecycle (``PENDING -> RUNNING -> COMPLETED``/``FAILED``) is kept in
sync with the domain-level ``RunResult`` it gets back, so that neither
``WorkflowController`` nor ``PipelineOrchestrator`` itself needs to
know about the other's concerns.
"""

from __future__ import annotations

from uaqe.application.application_errors import WorkflowExecutionError
from uaqe.application.workflow_context import WorkflowContext
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.result_types import RunResult
from uaqe.common.types import PipelineErrorPolicy
from uaqe.domain.pipeline_orchestrator import PipelineOrchestrator


class PipelineExecutor:
    """Runs (or resumes) one workflow's stage sequence.

    Attributes:
        _logger: Structured logging sink.
        _error_policy: The ``PipelineErrorPolicy`` applied to every run
            this executor drives, sourced from the run's
            ``ExecutionConfig.on_error``
            (``09_Architecture_Lock.md`` §5).
    """

    def __init__(self, logger: ILogger, error_policy: PipelineErrorPolicy) -> None:
        """Initialize the executor for a single run.

        Args:
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
            error_policy: The policy applied when a stage raises a
                ``UAQEError``, forwarded unchanged to
                ``PipelineOrchestrator``.
        """
        self._logger: ILogger = logger
        self._error_policy: PipelineErrorPolicy = error_policy

    def run(self, workflow_context: WorkflowContext) -> RunResult:
        """Execute ``workflow_context``'s stage sequence from a fresh
        start.

        Args:
            workflow_context: The run to execute.
                :attr:`~uaqe.application.workflow_context.
                WorkflowContext.pipeline_context` must not already have
                stage results appended; use :meth:`resume` for a
                partially-completed context instead.

        Returns:
            The domain-level ``RunResult`` produced by
            ``PipelineOrchestrator.run()``.

        Raises:
            WorkflowExecutionError: If ``workflow_context`` has no
                stages to run (an empty sequence was supplied by
                :class:`~uaqe.application.workflow_builder.
                WorkflowBuilder`).
        """
        stages = workflow_context.stages
        if not stages:
            raise WorkflowExecutionError(
                f"Cannot run workflow {workflow_context.run_id!r}: no "
                "stages were assembled for it.",
                code="WORKFLOW_EMPTY_STAGE_SEQUENCE",
                remediation_hint=(
                    "Verify WorkflowBuilder.build() populated "
                    "workflow_context.stages before calling "
                    "PipelineExecutor.run()."
                ),
            )

        workflow_context.mark_running()
        orchestrator = PipelineOrchestrator(
            stages,
            workflow_context.pipeline_context,
            self._logger,
            self._error_policy,
        )
        self._logger.info(
            "Pipeline execution starting.",
            run_id=workflow_context.run_id,
            stage_count=len(stages),
        )
        result = orchestrator.run()
        workflow_context.mark_completed(result.success)
        self._logger.info(
            "Pipeline execution finished.",
            run_id=workflow_context.run_id,
            success=result.success,
            duration_ms=workflow_context.elapsed_ms(),
        )
        return result

    def resume(self, workflow_context: WorkflowContext) -> RunResult:
        """Resume a partially-completed ``workflow_context``.

        Per ``09_Architecture_Lock.md`` §13 rule 5,
        ``PipelineOrchestrator.resume()`` is idempotent per
        already-populated ``PipelineContext`` entries — any stage
        already recorded in
        ``workflow_context.pipeline_context`` is skipped rather than
        re-executed.

        Args:
            workflow_context: The run to resume. Its ``run_id`` must
                match the ``run_id`` its
                ``pipeline_context`` was originally constructed with
                (enforced by ``PipelineOrchestrator.resume()`` itself).

        Returns:
            The domain-level ``RunResult`` produced by
            ``PipelineOrchestrator.resume()``.

        Raises:
            WorkflowExecutionError: If ``workflow_context`` has no
                stages assembled to resume.
        """
        stages = workflow_context.stages
        if not stages:
            raise WorkflowExecutionError(
                f"Cannot resume workflow {workflow_context.run_id!r}: no "
                "stages were assembled for it.",
                code="WORKFLOW_EMPTY_STAGE_SEQUENCE",
                remediation_hint=(
                    "Verify WorkflowBuilder.build() populated "
                    "workflow_context.stages before calling "
                    "PipelineExecutor.resume()."
                ),
            )

        workflow_context.mark_running()
        orchestrator = PipelineOrchestrator(
            stages,
            workflow_context.pipeline_context,
            self._logger,
            self._error_policy,
        )
        self._logger.info(
            "Pipeline execution resuming.",
            run_id=workflow_context.run_id,
            stage_count=len(stages),
        )
        result = orchestrator.resume(workflow_context.run_id)
        workflow_context.mark_completed(result.success)
        self._logger.info(
            "Pipeline execution finished (resumed).",
            run_id=workflow_context.run_id,
            success=result.success,
            duration_ms=workflow_context.elapsed_ms(),
        )
        return result
