"""Application-layer exception hierarchy for the Universal AI
Quantization Engine.

Per ``07_Coding_Standards.md`` §7, every exception raised in
``uaqe.application`` (as in ``uaqe.domain``) MUST be a subclass of
``UAQEError`` (``uaqe.common.exceptions``). ``09_Architecture_Lock.md``
§3 locks the *complete* set of twelve first-level ``UAQEError``
subclasses (``ModelLoadError`` .. ``PluginLoadError``); it does not
lock — and therefore does not forbid — more specific subclasses of an
*already-locked* subclass, the same latitude
``11_Implementation_Rules.md`` §4.9's "additional private/protected
helper methods MAY be added ... provided they do not change any locked
public signature" grants at the method level. Every exception below is
therefore declared as a narrower subclass of the single locked
exception whose ``code``/semantics it already matches
(``ConfigurationError`` — "configuration cannot be loaded, is missing
required fields, or otherwise fails validation" — is also exactly what
``PipelineOrchestrator.resume()`` already raises today for a
``RUN_ID_MISMATCH``, see ``uaqe.domain.pipeline_orchestrator``), rather
than by inventing a thirteenth top-level ``UAQEError`` branch.

This module has no dependency on any other ``uaqe`` package beyond
``uaqe.common``, per the layering rules in ``09_Architecture_Lock.md``
§8 (``application -> domain, common``).
"""

from typing import Optional

from uaqe.common.exceptions import ConfigurationError


class WorkflowConfigurationError(ConfigurationError):
    """Raised when a :class:`~uaqe.application.workflow_config.
    WorkflowConfig` fails validation before a run is even attempted.

    Typically raised by
    :meth:`~uaqe.application.workflow_config.WorkflowConfig.validate`
    for a missing ``model_path``, an empty ``hardware_profile_id``, or
    a malformed ``config_overrides`` entry — i.e. any failure that
    occurs *before* :class:`~uaqe.application.stage_factory.
    StageFactory` or :class:`~uaqe.application.pipeline_executor.
    PipelineExecutor` are ever invoked.
    """


class StageResolutionError(ConfigurationError):
    """Raised when :class:`~uaqe.application.stage_factory.
    StageFactory` cannot resolve or construct a requested pipeline
    stage.

    Typically raised when a stage name is not present in this
    codebase's implemented-stage registry (see
    :data:`~uaqe.application.stage_factory.IMPLEMENTED_STAGE_ORDER`),
    or when a collaborator a stage's constructor requires (an
    ``IFrameworkAdapter``, ``IExporterBackend``, ``IReportRenderer``,
    or strategy mapping) was not supplied to the factory.
    """


class WorkflowBuildError(ConfigurationError):
    """Raised when :class:`~uaqe.application.workflow_builder.
    WorkflowBuilder` cannot assemble a runnable stage sequence for a
    :class:`~uaqe.application.workflow_config.WorkflowConfig`.

    Distinct from :class:`StageResolutionError`: this is raised for
    assembly-level problems (e.g. an empty resolved stage sequence, or
    a stage sequence that violates the locked ordering in
    ``09_Architecture_Lock.md`` §12) rather than a single stage's own
    construction failure, which propagates as-is from
    :class:`StageResolutionError`.
    """


class WorkflowExecutionError(ConfigurationError):
    """Raised by :class:`~uaqe.application.workflow_controller.
    WorkflowController` or :class:`~uaqe.application.
    pipeline_executor.PipelineExecutor` for a failure in *running* a
    workflow that is not itself a single stage's ``UAQEError`` (which
    is instead captured into ``RunResult.errors`` per
    ``09_Architecture_Lock.md`` §13 and never re-raised).

    Typically raised when :meth:`~uaqe.application.
    workflow_controller.WorkflowController.resume` is called with a
    ``run_id`` unknown to this controller's session registry, or when
    :meth:`~uaqe.application.pipeline_executor.PipelineExecutor.run`
    is invoked with an empty stage sequence.
    """


class SessionNotFoundError(WorkflowExecutionError):
    """Raised when a :class:`~uaqe.application.workflow_context.
    WorkflowContext` is requested by ``run_id`` from
    :class:`~uaqe.application.workflow_controller.WorkflowController`
    and no such run is active.

    Narrower than :class:`WorkflowExecutionError` so that callers (a
    CLI, REST, or batch entry point) may distinguish "no such run" from
    every other workflow-execution failure without inspecting ``code``
    strings; both remain a :class:`~uaqe.common.exceptions.
    ConfigurationError` for any caller that only checks the broader
    type.
    """

    def __init__(
        self,
        run_id: str,
        *,
        remediation_hint: Optional[str] = (
            "Call WorkflowController.execute(config) to start a new "
            "run, or verify the run_id passed to resume()/get_context()."
        ),
    ) -> None:
        """Initialize a ``SessionNotFoundError`` for a missing ``run_id``.

        Args:
            run_id: The run identifier that could not be found.
            remediation_hint: Suggested corrective action; overridable
                by callers with more specific context.
        """
        super().__init__(
            f"No active workflow session found for run_id {run_id!r}.",
            code="WORKFLOW_SESSION_NOT_FOUND",
            remediation_hint=remediation_hint,
        )
        self.run_id: str = run_id
