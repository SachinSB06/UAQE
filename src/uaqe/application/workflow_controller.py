"""``WorkflowController`` — the single application-layer entry point an
``uaqe.interface`` caller (CLI, REST, batch) drives to start, resume,
and inspect workflow runs.

This is the locked ``uaqe.application.WorkflowController``
(``09_Architecture_Lock.md`` §3) itself; every other class in this
package exists to let this one stay a thin coordinator. It composes,
in order:

1. :class:`~uaqe.application.workflow_builder.WorkflowBuilder` —
   validates a :class:`~uaqe.application.workflow_config.WorkflowConfig`
   and assembles a runnable
   :class:`~uaqe.application.workflow_context.WorkflowContext`.
2. :class:`~uaqe.application.pipeline_executor.PipelineExecutor` — runs
   or resumes that context against a fresh
   :class:`~uaqe.domain.pipeline_orchestrator.PipelineOrchestrator`.
3. An in-process ``run_id -> WorkflowContext`` registry, absorbing the
   doc-locked ``SessionManager``'s bookkeeping role
   (``03_API_Specification.md`` §14.4) directly into this class rather
   than a separate file — the same consolidation
   :class:`~uaqe.application.workflow_context.WorkflowContext`'s module
   docstring already documents.

Per the locked layering (``09_Architecture_Lock.md`` §8:
``application -> domain, common``), this module never imports
``uaqe.infrastructure`` or ``uaqe.interface``; every collaborator below
is constructor-injected, ultimately resolved by
``uaqe.interface.composition_root.CompositionRoot.build_workflow_controller``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from uaqe.application.application_errors import SessionNotFoundError
from uaqe.application.pipeline_executor import PipelineExecutor
from uaqe.application.workflow_builder import WorkflowBuilder
from uaqe.application.workflow_config import WorkflowConfig
from uaqe.application.workflow_context import WorkflowContext
from uaqe.application.workflow_result import WorkflowResult
from uaqe.common.interfaces.i_logger import ILogger


class WorkflowController:
    """Starts, resumes, and tracks workflow runs.

    Attributes:
        _builder: Validates a ``WorkflowConfig`` and assembles this
            run's ``WorkflowContext``.
        _executor: Runs/resumes an assembled ``WorkflowContext``.
        _logger: Structured logging sink.
        _sessions: Every ``WorkflowContext`` started by this controller
            instance, keyed by ``run_id``, for the lifetime of this
            process — the registry that replaces the doc-locked
            ``SessionManager``'s ``create_session``/``get_session``/
            ``close_session`` trio (``03_API_Specification.md`` §14.4).
    """

    def __init__(
        self,
        builder: WorkflowBuilder,
        executor: PipelineExecutor,
        logger: ILogger,
    ) -> None:
        """Initialize the controller.

        Args:
            builder: The ``WorkflowBuilder`` used to assemble every run
                started via :meth:`execute`.
            executor: The ``PipelineExecutor`` used to run/resume every
                run this controller drives.
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
        """
        self._builder: WorkflowBuilder = builder
        self._executor: PipelineExecutor = executor
        self._logger: ILogger = logger
        self._sessions: Dict[str, WorkflowContext] = {}

    def execute(
        self, config: WorkflowConfig, run_id: Optional[str] = None
    ) -> WorkflowResult:
        """Start and run a brand-new workflow from ``config``.

        Args:
            config: The run's ``WorkflowConfig``, as supplied by an
                ``uaqe.interface`` entry point.
            run_id: An explicit run identifier to use instead of a
                freshly minted one; forwarded unchanged to
                ``WorkflowBuilder.build``.

        Returns:
            The completed run's ``WorkflowResult``.

        Raises:
            WorkflowConfigurationError: If ``config`` fails validation.
            WorkflowBuildError: If no stages could be assembled for
                ``config``.
            StageResolutionError: If a required stage collaborator was
                not supplied to the injected ``StageFactory``.
        """
        context = self._builder.build(config, run_id=run_id)
        self._sessions[context.run_id] = context

        self._logger.info(
            "Workflow execution requested.",
            run_id=context.run_id,
        )
        run_result = self._executor.run(context)
        return WorkflowResult.from_context(context, run_result)

    def resume(self, run_id: str) -> WorkflowResult:
        """Resume a previously started run identified by ``run_id``.

        Args:
            run_id: The identifier of a run previously started via
                :meth:`execute` on this same controller instance.

        Returns:
            The resumed run's ``WorkflowResult``.

        Raises:
            SessionNotFoundError: If ``run_id`` is not present in this
                controller's session registry.
            WorkflowExecutionError: If the located ``WorkflowContext``
                has no stages assembled to resume (should not occur for
                any context produced by :meth:`execute`).
        """
        context = self.get_context(run_id)

        self._logger.info(
            "Workflow resume requested.",
            run_id=context.run_id,
        )
        run_result = self._executor.resume(context)
        return WorkflowResult.from_context(context, run_result)

    def get_context(self, run_id: str) -> WorkflowContext:
        """Look up a previously started run's ``WorkflowContext``.

        Args:
            run_id: The identifier of a run previously started via
                :meth:`execute` on this same controller instance.

        Returns:
            The matching ``WorkflowContext``.

        Raises:
            SessionNotFoundError: If ``run_id`` is not present in this
                controller's session registry.
        """
        try:
            return self._sessions[run_id]
        except KeyError:
            raise SessionNotFoundError(run_id) from None

    def list_active_run_ids(self) -> List[str]:
        """List every tracked run that has not yet reached a terminal
        status.

        Returns:
            The ``run_id`` of every session whose ``WorkflowContext.
            is_terminal()`` is ``False``, in registration order.
        """
        return [
            run_id
            for run_id, context in self._sessions.items()
            if not context.is_terminal()
        ]
