"""Loads a raw source-framework model file into the framework-independent
``IMR``, delegating to the correct ``IFrameworkAdapter``.

Per ``01_Project_Architecture.md`` §6, ``ModelLoader``'s responsibility
is strictly to load the raw file and delegate to the correct
``FrameworkAdapter`` to produce an ``IMR`` — it must never perform
analysis or validation logic (that is ``ModelAnalyzer``'s and
``ModelValidator``'s responsibility, respectively).

Locked contract: ``03_API_Specification.md`` §3.1.
Locked payload mapping: ``09_Architecture_Lock.md`` §9
(``model_loader`` -> ``IMR``).
"""

from __future__ import annotations

import time
from typing import List, Optional

from uaqe.common.exceptions import ModelLoadError
from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_framework_adapter import IFrameworkAdapter
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.result_types import StageResult
from uaqe.model_loader.framework_detector import FrameworkDetector
from uaqe.model_loader.loader_factory import LoaderFactory
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.domain.pipeline_stage import PipelineStage


class ModelLoader(PipelineStage):
    """Delegates to the correct ``IFrameworkAdapter`` to produce an ``IMR``.

    Per ``10_Module_Development_Guide.md`` §3's Future Extensions note,
    adding a new input format never requires a change here: a new
    adapter is registered with the injected ``adapters`` list (via
    ``CompositionRoot`` or ``PluginRegistry``), and this class continues
    to iterate them unchanged, delegating extension recognition to
    :class:`~uaqe.domain.model.framework_detector.FrameworkDetector` and
    adapter resolution to
    :class:`~uaqe.domain.model.loader_factory.LoaderFactory`.

    Attributes:
        _adapters: Every ``IFrameworkAdapter`` available to this run,
            constructor-injected by ``CompositionRoot``. Never a
            concrete adapter imported directly — per
            ``10_Module_Development_Guide.md`` §3 Dependencies, this
            package may not import ``uaqe.infrastructure``.
        _logger: The logger this stage reports load start/end and
            failures to.
        _detector: Resolves a source file's canonical framework name
            from its extension.
        _factory: Resolves the ``IFrameworkAdapter`` instance that
            supports a given extension.
        _pending_path: The source model path for the run currently
            being ingested, set via :meth:`bind_source_path` before
            :meth:`execute` is invoked. See the note on that method for
            why this indirection exists.
    """

    def __init__(self, adapters: List[IFrameworkAdapter], logger: ILogger) -> None:
        """Initialize the loader with its available adapters.

        Args:
            adapters: The ``IFrameworkAdapter`` instances this loader
                may delegate to, in resolution-priority order.
            logger: The logger to report stage events to.
        """
        self._adapters: List[IFrameworkAdapter] = adapters
        self._logger: ILogger = logger
        self._detector: FrameworkDetector = FrameworkDetector()
        self._factory: LoaderFactory = LoaderFactory(adapters)
        self._pending_path: Optional[str] = None

    def bind_source_path(self, path: str) -> None:
        """Bind the source model path this run's :meth:`execute` will load.

        ``PipelineStage.execute(context: PipelineContext) -> StageResult``
        is locked verbatim (``09_Architecture_Lock.md`` §4) and, unlike
        every downstream stage, ``model_loader`` has no prior stage
        result in ``context`` to read its input from — its input is the
        raw ``RunRequest.model_path`` (``04_Data_Flow.md`` §2), supplied
        by ``ModelIngestionService`` before the pipeline's stage loop
        begins. This method is the documented seam
        ``ModelIngestionService`` uses to hand that path to this stage
        without altering the locked ``execute()`` signature; it is
        internal API (``09_Architecture_Lock.md`` §11) and not part of
        ``03_API_Specification.md``.

        Args:
            path: Filesystem path to the source model file for the
                current run.
        """
        self._pending_path = path

    def execute(self, context: PipelineContext) -> StageResult:
        """Load the bound source model file and produce its ``IMR``.

        Args:
            context: The run's pipeline context. ``model_loader`` reads
                no prior stage result (it is the first stage in the
                locked pipeline order, ``09_Architecture_Lock.md`` §12);
                its input is the path previously supplied via
                :meth:`bind_source_path`.

        Returns:
            A ``StageResult`` whose ``payload`` is the loaded ``IMR``.

        Raises:
            ModelLoadError: If no source path has been bound, the
                extension cannot be recognized, no adapter supports it,
                or the resolved adapter fails to load the file. Per
                ``04_Data_Flow.md`` §2, a ``ModelLoadError`` here always
                forces ``ExecutionConfig.on_error`` to behave as
                ``ABORT`` regardless of the configured policy, since no
                valid ``IMR`` exists to hand to any later stage.
        """
        started_at = time.monotonic()
        if self._pending_path is None:
            raise ModelLoadError(
                "ModelLoader.execute() called with no source path bound.",
                code="MODEL_PATH_NOT_BOUND",
                remediation_hint=(
                    "Call bind_source_path(path) — normally done by "
                    "ModelIngestionService — before running the pipeline."
                ),
            )
        path = self._pending_path

        self._logger.info("model_loader: starting load", stage="model_loader", path=path)
        try:
            framework = self.detect_framework(path)
            adapter = self._factory.resolve(self._detector.extension_of(path))
            imr: IMR = adapter.load(path)
        except ModelLoadError as exc:
            log_context = exc.to_log_context()
            log_context["stage"] = self.name()
            self._logger.error(
                "model_loader: load failed",
                path=path,
                **log_context,
            )
            raise

        duration_ms = (time.monotonic() - started_at) * 1000.0
        self._logger.info(
            "model_loader: load succeeded",
            stage="model_loader",
            path=path,
            framework=framework,
            op_count=imr.metadata.op_count,
            duration_ms=duration_ms,
        )
        return StageResult(
            stage_name=self.name(),
            success=True,
            payload=imr,
            warnings=[],
            duration_ms=duration_ms,
        )

    def detect_framework(self, path: str) -> str:
        """Resolve the canonical source-framework name for ``path``.

        Args:
            path: Filesystem path to the source model file.

        Returns:
            The canonical framework name (e.g. ``"pytorch"``,
            ``"onnx"``), as resolved by
            :class:`~uaqe.domain.model.framework_detector.FrameworkDetector`.

        Raises:
            ModelLoadError: If ``path``'s extension cannot be
                recognized as any supported source framework.
        """
        return self._detector.detect(path)
