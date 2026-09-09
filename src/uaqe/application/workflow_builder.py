"""``WorkflowBuilder`` — validates a
:class:`~uaqe.application.workflow_config.WorkflowConfig` and assembles
the runnable :class:`~uaqe.application.workflow_context.WorkflowContext`
:class:`~uaqe.application.workflow_controller.WorkflowController` hands
to :class:`~uaqe.application.pipeline_executor.PipelineExecutor`.

This is the application-layer counterpart to the doc-locked
``PipelineBuilder`` (``09_Architecture_Lock.md`` §3,
``uaqe.domain.pipeline``): where a locked ``PipelineBuilder`` would
materialize the stage order directly, this package already delegates
stage *construction* to :class:`~uaqe.application.stage_factory.
StageFactory` (see that module's docstring for why). ``WorkflowBuilder``
is therefore the thin layer above ``StageFactory`` that does the two
things a stage factory alone cannot: validating the incoming
``WorkflowConfig`` before any stage is touched, and turning the
resulting stage sequence into a fully-formed ``WorkflowContext`` ready
to run — the same "assembly vs. construction" split
:class:`~uaqe.application.application_errors.WorkflowBuildError`'s
docstring already draws against
:class:`~uaqe.application.application_errors.StageResolutionError`.
"""

from __future__ import annotations

from typing import Optional

from uaqe.application.application_errors import WorkflowBuildError
from uaqe.application.stage_factory import StageFactory
from uaqe.application.workflow_config import WorkflowConfig
from uaqe.application.workflow_context import WorkflowContext
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.result_types import StageResult


class WorkflowBuilder:
    """Assembles one run's ``WorkflowContext`` from its ``WorkflowConfig``.

    Attributes:
        _stage_factory: Resolves and constructs this run's ordered
            stage sequence.
        _logger: Structured logging sink.
    """

    def __init__(self, stage_factory: StageFactory, logger: ILogger) -> None:
        """Initialize the builder.

        Args:
            stage_factory: The ``StageFactory`` used to construct every
                run's stage sequence. Injected rather than constructed
                here, per the constructor-injection rule every class in
                this package follows.
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
        """
        self._stage_factory: StageFactory = stage_factory
        self._logger: ILogger = logger

    def build(
        self, config: WorkflowConfig, run_id: Optional[str] = None
    ) -> WorkflowContext:
        """Validate ``config`` and assemble a fresh, runnable
        ``WorkflowContext`` for it.

        Args:
            config: The run's ``WorkflowConfig``, as supplied by an
                ``uaqe.interface`` entry point.
            run_id: An explicit run identifier to use instead of a
                freshly minted one; forwarded unchanged to
                ``WorkflowContext.__init__``.

        Returns:
            A ``WorkflowContext`` whose ``stages`` is populated with
            this run's complete, ordered stage sequence and whose
            ``status`` is still ``PENDING`` — ready to be handed to
            :class:`~uaqe.application.pipeline_executor.
            PipelineExecutor.run`.

        Raises:
            WorkflowConfigurationError: If ``config.validate()`` fails
                (missing/empty required fields, malformed
                ``config_overrides``).
            WorkflowBuildError: If ``StageFactory.build_stage_sequence``
                resolves an empty stage sequence for ``config`` — a run
                with no stages can never produce a meaningful
                ``RunResult`` and is treated as an assembly failure
                rather than being silently handed to
                ``PipelineExecutor`` (which would itself raise a
                ``WorkflowExecutionError`` for the same condition, one
                layer later and with less assembly-time context).
            StageResolutionError: Propagated unchanged from
                ``StageFactory.build_stage_sequence`` if any individual
                stage cannot be constructed.
        """
        config.validate()

        context = WorkflowContext(config, run_id=run_id)
        context.pipeline_context.append(
            "workflow_config",
            StageResult(stage_name="workflow_config", success=True, payload=config),
        )
        stages = self._stage_factory.build_stage_sequence(config)
        if not stages:
            raise WorkflowBuildError(
                f"Cannot build workflow {context.run_id!r}: "
                "StageFactory.build_stage_sequence returned an empty "
                "stage sequence.",
                code="WORKFLOW_EMPTY_STAGE_SEQUENCE",
                remediation_hint=(
                    "Verify StageFactoryDependencies was wired with at "
                    "least one implemented stage's collaborators before "
                    "calling WorkflowBuilder.build()."
                ),
            )
        context.stages = stages

        self._logger.info(
            "Workflow assembled.",
            run_id=context.run_id,
            stage_count=len(stages),
        )
        return context
