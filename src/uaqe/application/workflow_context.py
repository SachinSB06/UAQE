"""``WorkflowContext`` — the per-run application-layer envelope around
a single :class:`~uaqe.domain.pipeline_context.PipelineContext`.

Where ``uaqe.domain.pipeline_context.PipelineContext`` is the
append-only, stage-result-only channel ``PipelineStage`` instances read
and write (``09_Architecture_Lock.md`` §13 rule 1), ``WorkflowContext``
is the outer, application-owned record of *one run as a whole*: which
:class:`~uaqe.application.workflow_config.WorkflowConfig` started it,
its ``run_id``, its lifecycle :class:`WorkflowStatus`, and the wall-clock
timestamps ``PipelineOrchestrator`` itself does not track. It replaces
the doc-locked ``SessionManager``'s per-run bookkeeping role
(``03_API_Specification.md`` §14.4: ``create_session``/``get_session``/
``close_session``) with a single value-plus-behavior class that
:class:`~uaqe.application.workflow_controller.WorkflowController` keeps
a registry of, rather than a separate ``session_manager.py`` file —
the same one-class-absorbs-a-doc-locked-service consolidation already
applied to ``ModelIngestionService``/``HardwareSelectionService`` in
:mod:`uaqe.application.stage_factory`.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import List, Optional

from uaqe.application.workflow_config import WorkflowConfig
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.domain.pipeline_stage import PipelineStage


class WorkflowStatus(Enum):
    """The lifecycle state of one workflow run.

    Mirrors the states ``PipelineOrchestrator.run()``/``resume()``
    implicitly move a run through; ``WorkflowContext`` is what makes
    that state explicit and queryable between (or during) orchestrator
    invocations, since ``PipelineContext`` itself carries no status
    field.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class WorkflowContext:
    """One run's application-layer session state.

    Attributes:
        run_id: The unique identifier of this run, threaded unchanged
            through every log line for the run's lifetime
            (``11_Implementation_Rules.md`` §11).
        config: The :class:`~uaqe.application.workflow_config.
            WorkflowConfig` this run was started from. Immutable for
            the run's lifetime, matching the frozen-dataclass contract
            every config value object in this codebase holds.
        pipeline_context: The domain-level
            :class:`~uaqe.domain.pipeline_context.PipelineContext`
            every :class:`~uaqe.domain.pipeline_stage.PipelineStage`
            in this run reads from and is appended to by
            :class:`~uaqe.domain.pipeline_orchestrator.
            PipelineOrchestrator`.
        stages: This run's ordered, constructed stage sequence, as
            assembled by :class:`~uaqe.application.workflow_builder.
            WorkflowBuilder` via :class:`~uaqe.application.
            stage_factory.StageFactory`. Empty until
            ``WorkflowBuilder.build()`` populates it; read by
            :class:`~uaqe.application.pipeline_executor.
            PipelineExecutor` to construct this run's
            ``PipelineOrchestrator``.
        status: This run's current :class:`WorkflowStatus`.
        started_at_monotonic: The ``time.monotonic()`` value recorded
            when this context was constructed, used to compute
            wall-clock run duration independent of system clock
            adjustments — the same primitive every stage already uses
            for its own ``duration_ms`` (e.g.
            ``uaqe.optimizer.optimizer.Optimizer.execute``).
        completed_at_monotonic: The ``time.monotonic()`` value recorded
            when :meth:`mark_completed` or :meth:`mark_failed` was
            called; ``None`` while :attr:`status` is ``PENDING`` or
            ``RUNNING``.
    """

    def __init__(
        self,
        config: WorkflowConfig,
        run_id: Optional[str] = None,
    ) -> None:
        """Initialize a fresh ``WorkflowContext`` for a new run.

        Args:
            config: The validated :class:`~uaqe.application.
                workflow_config.WorkflowConfig` this run was started
                from. Callers are expected to have already called
                :meth:`~uaqe.application.workflow_config.WorkflowConfig.
                validate` (normally done by
                :class:`~uaqe.application.workflow_builder.
                WorkflowBuilder` before this context is constructed).
            run_id: An explicit run identifier, e.g. supplied by an
                ``uaqe.interface`` entry point that already generated
                one under ``config.json``'s
                ``default_run_id_prefix``. A fresh UUID4-suffixed id
                (prefixed with :attr:`~uaqe.application.workflow_config.
                WorkflowConfig.run_id_prefix` if set) is minted if
                omitted.
        """
        self.run_id: str = run_id or self._mint_run_id(config)
        self.config: WorkflowConfig = config
        self.pipeline_context: PipelineContext = PipelineContext(self.run_id)
        self.stages: List[PipelineStage] = []
        self.status: WorkflowStatus = WorkflowStatus.PENDING
        self.started_at_monotonic: float = time.monotonic()
        self.completed_at_monotonic: Optional[float] = None

    @staticmethod
    def _mint_run_id(config: WorkflowConfig) -> str:
        """Mint a fresh run identifier when none was supplied.

        Args:
            config: The run's ``WorkflowConfig``, consulted for an
                optional :attr:`~uaqe.application.workflow_config.
                WorkflowConfig.run_id_prefix` override.

        Returns:
            A run identifier unique enough for the lifetime of a single
            process's ``WorkflowController`` session registry.
        """
        prefix = config.run_id_prefix or "run"
        return f"{prefix}-{uuid.uuid4().hex[:12]}"

    def mark_running(self) -> None:
        """Transition this context to :attr:`WorkflowStatus.RUNNING`.

        Called by :class:`~uaqe.application.pipeline_executor.
        PipelineExecutor` immediately before it invokes
        ``PipelineOrchestrator.run()``/``resume()``.
        """
        self.status = WorkflowStatus.RUNNING

    def mark_completed(self, success: bool) -> None:
        """Transition this context to its terminal status and record
        the completion timestamp.

        Args:
            success: Whether the run completed with every stage
                succeeding under the run's error policy — mirrors
                ``RunResult.success``.
        """
        self.status = WorkflowStatus.COMPLETED if success else WorkflowStatus.FAILED
        self.completed_at_monotonic = time.monotonic()

    def elapsed_ms(self) -> float:
        """Return wall-clock milliseconds elapsed since this context
        was constructed.

        Returns:
            The elapsed time up to :attr:`completed_at_monotonic` if
            the run has finished, else up to the current time for a
            still-running run.
        """
        end = self.completed_at_monotonic or time.monotonic()
        return (end - self.started_at_monotonic) * 1000.0

    def is_terminal(self) -> bool:
        """Report whether this run has reached a terminal status.

        Returns:
            ``True`` if :attr:`status` is ``COMPLETED`` or ``FAILED``.
        """
        return self.status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED)
