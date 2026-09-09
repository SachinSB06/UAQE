"""Application layer for the Universal AI Quantization Engine.

Per ``09_Architecture_Lock.md`` §2/§3, this package is ``uaqe.application``
and its locked public class inventory is exactly ``RunRequest``,
``ModelIngestionService``, ``HardwareSelectionService``,
``SessionManager``, ``WorkflowController``. As documented in
:mod:`uaqe.application.workflow_config`, :mod:`uaqe.application.
workflow_context`, and :mod:`uaqe.application.stage_factory`'s module
docstrings, this codebase snapshot implements that same *role* under
different names — ``WorkflowConfig`` for ``RunRequest``,
``StageFactory`` folding in ``ModelIngestionService``/
``HardwareSelectionService``'s seam-binding responsibilities, and
``WorkflowController`` itself absorbing ``SessionManager``'s registry
role directly rather than through a separate file — the same
diverging-names-same-role precedent ``uaqe.optimizer``'s own
``__init__.py`` already documents for itself. ``WorkflowController`` is
this package's one locked name and is unchanged.

This package's ten files, front to back:

1. :mod:`~uaqe.application.workflow_config` — ``WorkflowConfig``, the
   single input value object a caller supplies to start a run.
2. :mod:`~uaqe.application.workflow_context` — ``WorkflowContext`` /
   ``WorkflowStatus``, one run's application-layer session state.
3. :mod:`~uaqe.application.stage_factory` — ``StageFactory`` /
   ``StageFactoryDependencies``, resolves and constructs one run's
   ordered stage sequence.
4. :mod:`~uaqe.application.workflow_builder` — ``WorkflowBuilder``,
   validates a ``WorkflowConfig`` and assembles a runnable
   ``WorkflowContext``.
5. :mod:`~uaqe.application.pipeline_executor` — ``PipelineExecutor``,
   runs/resumes an assembled ``WorkflowContext`` against a
   ``PipelineOrchestrator``.
6. :mod:`~uaqe.application.workflow_result` — ``WorkflowResult``, the
   application-layer outcome of one run.
7. :mod:`~uaqe.application.execution_summary` — ``ExecutionSummary``, a
   flattened, display-ready digest of a ``WorkflowResult``.
8. :mod:`~uaqe.application.workflow_controller` — ``WorkflowController``,
   the single entry point an ``uaqe.interface`` caller drives.
9. :mod:`~uaqe.application.application_errors` — this package's
   ``UAQEError`` subclass hierarchy.

Per the locked layering (``09_Architecture_Lock.md`` §8:
``application -> domain, common``), nothing in this package imports
``uaqe.infrastructure`` or ``uaqe.interface``; every interface-typed
collaborator any class below needs is constructor-injected, ultimately
resolved by
``uaqe.interface.composition_root.CompositionRoot.build_workflow_controller``.
"""

from __future__ import annotations

from uaqe.application.application_errors import (
    SessionNotFoundError,
    StageResolutionError,
    WorkflowBuildError,
    WorkflowConfigurationError,
    WorkflowExecutionError,
)
from uaqe.application.execution_summary import ExecutionSummary
from uaqe.application.pipeline_executor import PipelineExecutor
from uaqe.application.stage_factory import (
    IMPLEMENTED_STAGE_ORDER,
    StageFactory,
    StageFactoryDependencies,
)
from uaqe.application.workflow_builder import WorkflowBuilder
from uaqe.application.workflow_config import WorkflowConfig
from uaqe.application.workflow_context import WorkflowContext, WorkflowStatus
from uaqe.application.workflow_controller import WorkflowController
from uaqe.application.workflow_result import WorkflowResult

__all__ = [
    "WorkflowConfig",
    "WorkflowContext",
    "WorkflowStatus",
    "StageFactory",
    "StageFactoryDependencies",
    "IMPLEMENTED_STAGE_ORDER",
    "WorkflowBuilder",
    "PipelineExecutor",
    "WorkflowResult",
    "ExecutionSummary",
    "WorkflowController",
    "WorkflowConfigurationError",
    "StageResolutionError",
    "WorkflowBuildError",
    "WorkflowExecutionError",
    "SessionNotFoundError",
]
