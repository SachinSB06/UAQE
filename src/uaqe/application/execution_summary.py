"""``ExecutionSummary`` — a flattened, human/log-friendly digest of a
:class:`~uaqe.application.workflow_result.WorkflowResult`.

``WorkflowResult`` (and the domain-level ``RunResult`` it wraps) is
optimized for programmatic consumption — every ``StageResult`` keyed by
stage name, every raw ``UAQEError``. ``ExecutionSummary`` is the
flattened counterpart an ``uaqe.interface`` entry point (CLI output,
REST response body, batch run log line) renders directly: stage counts
instead of a dict to iterate, error codes instead of exception objects,
one ready-to-print ``describe()`` string. It never re-derives anything
``PipelineOrchestrator``/``PipelineExecutor`` already computed — every
field below is a straight projection of :class:`~uaqe.application.
workflow_result.WorkflowResult` and the ``RunResult``/``StageResult``
values nested inside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from uaqe.application.workflow_context import WorkflowStatus
from uaqe.application.workflow_result import WorkflowResult


@dataclass(frozen=True)
class ExecutionSummary:
    """A flattened summary of one workflow run, suitable for direct
    display or logging.

    Attributes:
        run_id: The unique identifier of the run.
        status: The run's terminal (or current)
            :class:`~uaqe.application.workflow_context.WorkflowStatus`.
        success: Whether every stage completed successfully under the
            run's ``PipelineErrorPolicy``.
        total_stage_count: Total number of stages that were executed
            (or skipped-on-resume) during this run.
        succeeded_stage_names: Names of every stage whose
            ``StageResult.success`` was ``True``, in the order
            ``RunResult.stage_results`` iterates them.
        failed_stage_names: Names of every stage whose
            ``StageResult.success`` was ``False``.
        error_codes: The ``UAQEError.code`` of every error raised during
            the run, in the order they were raised.
        output_paths: Filesystem paths written to ``outputs/`` or
            ``reports/`` during the run, per ``RunResult.output_paths``.
        duration_ms: Total wall-clock time spent on this run, in
            milliseconds.
    """

    run_id: str
    status: WorkflowStatus
    success: bool
    total_stage_count: int
    succeeded_stage_names: List[str] = field(default_factory=list)
    failed_stage_names: List[str] = field(default_factory=list)
    error_codes: List[str] = field(default_factory=list)
    output_paths: List[str] = field(default_factory=list)
    duration_ms: float = 0.0

    @classmethod
    def from_result(cls, result: WorkflowResult) -> "ExecutionSummary":
        """Build an ``ExecutionSummary`` from a completed
        ``WorkflowResult``.

        Args:
            result: The ``WorkflowResult`` returned by
                :class:`~uaqe.application.workflow_controller.
                WorkflowController.execute`/``resume``.

        Returns:
            The flattened summary.
        """
        run_result = result.run_result
        succeeded = [
            name
            for name, stage_result in run_result.stage_results.items()
            if stage_result.success
        ]
        failed = [
            name
            for name, stage_result in run_result.stage_results.items()
            if not stage_result.success
        ]
        return cls(
            run_id=result.run_id,
            status=result.status,
            success=result.success,
            total_stage_count=len(run_result.stage_results),
            succeeded_stage_names=succeeded,
            failed_stage_names=failed,
            error_codes=[error.code for error in run_result.errors],
            output_paths=list(run_result.output_paths),
            duration_ms=result.duration_ms,
        )

    def describe(self) -> str:
        """Render a single human-readable summary line.

        Returns:
            A one-line description suitable for CLI output or a log
            line, e.g. ``"run 'run-abc123': COMPLETED (12/12 stages
            succeeded, 842.3ms)"``.
        """
        return (
            f"run {self.run_id!r}: {self.status.value} "
            f"({len(self.succeeded_stage_names)}/{self.total_stage_count} "
            f"stages succeeded, {self.duration_ms:.1f}ms)"
        )
