"""``WorkflowResult`` — the application-layer outcome of one
:class:`~uaqe.application.workflow_controller.WorkflowController` call
(``execute()`` or ``resume()``).

Where the domain-level ``RunResult`` (``uaqe.common.result_types``) is
``PipelineOrchestrator``'s own return value — a flat record of stage
results, errors, and output paths for a single ``run()``/``resume()``
call — ``WorkflowResult`` is the outer, application-owned envelope that
pairs that domain result with the run-level bookkeeping only
:class:`~uaqe.application.workflow_context.WorkflowContext` carries
(``run_id`` minted at the application layer, the run's
:class:`~uaqe.application.workflow_context.WorkflowStatus`, and total
wall-clock duration). This is the value
:class:`~uaqe.application.workflow_controller.WorkflowController`
actually hands back to an ``uaqe.interface`` caller, so that caller
never has to reach into ``uaqe.domain`` types directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from uaqe.application.workflow_context import WorkflowContext, WorkflowStatus
from uaqe.common.result_types import RunResult


@dataclass(frozen=True)
class WorkflowResult:
    """The outcome of one complete (or partially complete) workflow run,
    as returned by :class:`~uaqe.application.workflow_controller.
    WorkflowController`.

    Attributes:
        run_id: The unique identifier of the run, matching the
            originating :class:`~uaqe.application.workflow_context.
            WorkflowContext.run_id`.
        status: The run's terminal (or, if interrupted mid-run, current)
            :class:`~uaqe.application.workflow_context.WorkflowStatus`.
        success: Whether every stage in the run completed successfully
            under the run's ``PipelineErrorPolicy`` — mirrors
            ``RunResult.success``.
        run_result: The domain-level ``RunResult`` produced by
            ``PipelineOrchestrator.run()``/``resume()``, carrying every
            ``StageResult`` and ``UAQEError`` from the run.
        duration_ms: Total wall-clock time spent on this run so far, in
            milliseconds, as measured by
            :meth:`~uaqe.application.workflow_context.WorkflowContext.
            elapsed_ms`.
    """

    run_id: str
    status: WorkflowStatus
    success: bool
    run_result: RunResult
    duration_ms: float

    @classmethod
    def from_context(
        cls, context: WorkflowContext, run_result: RunResult
    ) -> "WorkflowResult":
        """Build a ``WorkflowResult`` from a run's ``WorkflowContext``
        and the ``RunResult`` its ``PipelineExecutor`` call produced.

        Args:
            context: The run's ``WorkflowContext``, already transitioned
                to a terminal status by
                :class:`~uaqe.application.pipeline_executor.
                PipelineExecutor` (``mark_completed()`` called).
            run_result: The ``RunResult`` returned by that same
                ``PipelineExecutor.run()``/``resume()`` call.

        Returns:
            The combined application-layer ``WorkflowResult``.
        """
        return cls(
            run_id=context.run_id,
            status=context.status,
            success=run_result.success,
            run_result=run_result,
            duration_ms=context.elapsed_ms(),
        )
