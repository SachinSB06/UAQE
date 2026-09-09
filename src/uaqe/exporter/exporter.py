"""Deployment-artifact export — the locked ``uaqe.domain.export.exporter``
contract (``03_API_Specification.md`` §9.1), decomposed into this
package.

``Exporter`` is the sole orchestrating ``PipelineStage`` for this
package (registered under context key ``"exporter"``), the pipeline's
terminal IMR-consuming stage (``09_Architecture_Lock.md`` §12:
``... -> memory_optimizer -> exporter -> evaluation -> ...``). Per
``04_Data_Flow.md`` §7 and the same precedent every model-transforming
stage in this codebase already sets
(:class:`~uaqe.compression.compression_planner.CompressionPlanner`,
:class:`~uaqe.optimizer.memory_optimizer.MemoryOptimizer`), it reads the
*current* ``IMR`` from the latest stage in the model-transforming
chain rather than a fixed context key: ``"memory_optimizer"`` when
present, falling back through ``"optimizer"``, ``"compression_planner"``,
``"quantization_planner"``, and finally ``"model_loader"`` for a run
where a later stage was disabled or has not yet executed.

Pipeline within this stage's single ``execute()`` call:

1. :class:`~uaqe.exporter.export_planner.ExportPlanner` decides which
   ``ExportFormat`` to target for the run's resolved ``HardwareProfile``.
2. :class:`~uaqe.exporter.export_validator.ExportValidator` re-checks
   the final ``IMR`` against that target immediately before writing
   anything (see that module's docstring for exactly what it checks
   and why, given everything upstream already validated).
3. :meth:`Exporter.select_backend` resolves the concrete
   ``IExporterBackend`` registered for the planned ``ExportFormat`` —
   this method's locked signature (``select_backend(profile) ->
   IExporterBackend``) is preserved exactly; format resolution happens
   internally via step 1's planner so the public contract stays
   profile-in, backend-out.
4. The selected backend's ``export()`` writes the artifact file(s) and
   returns the initial ``DeploymentArtifact``.
5. :class:`~uaqe.exporter.artifact_manifest.ArtifactManifestBuilder`
   checksums those files into a ``manifest.json``.
6. :class:`~uaqe.exporter.package_builder.PackageBuilder` bundles the
   artifact files and manifest into one deployment archive.

Backends (each an ``IExporterBackend``, resolved by
:meth:`Exporter.select_backend` — never imported directly outside of
composition-root wiring):

- :class:`~uaqe.exporter.binary_exporter.BinaryExporter` — ``BIN``.
- :class:`~uaqe.exporter.hex_exporter.HexExporter` — ``HEX``.
- :class:`~uaqe.exporter.mem_exporter.MemExporter` — ``MEM``.
- :class:`~uaqe.exporter.tflite_exporter.TFLiteExporter` — ``TFLITE``.
- :class:`~uaqe.exporter.onnx_exporter.OnnxExporter` — ``ONNX``.

``ExportFormat.H_HEADER`` (``03_API_Specification.md`` §1.2) has no
first-party backend in this package yet; :meth:`Exporter.select_backend`
raises ``ExportError`` for a profile whose plan resolves to it, the
same failure mode as any other format with no registered backend, so
adding an ``H_HEADER`` backend later requires no change to
``Exporter`` itself — only one more entry in the ``backends`` list
supplied at construction.

Scope note on ``uaqe.common.interfaces.i_exporter_backend``'s
``TYPE_CHECKING``-only forward references: that module (pre-dating this
package's decomposition) points ``DeploymentArtifact`` at
``uaqe.domain.export.exporter`` and ``HardwareProfile`` at
``uaqe.domain.hardware.hardware_manager``. Neither module exists in
this codebase's actual layout — ``HardwareProfile`` lives at
``uaqe.domain.hardware_manager`` (see that module's own docstring for
why), and ``DeploymentArtifact`` is defined right here, in
``uaqe.exporter.exporter``, following the same "decomposed package
holds the richer domain dataclass" precedent
``uaqe.compression.compression_planner.CompressionPlan`` and
``uaqe.optimizer.optimization_result.OptimizationResult`` already set.
Because those references are ``TYPE_CHECKING``-guarded, nothing at
runtime actually imports through them, so this does not affect any
concrete backend's behavior — only a static type checker resolving the
abstract method's annotations would notice the stale path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, List, Optional

from uaqe.common.exceptions import ExportError
from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_exporter_backend import IExporterBackend
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.result_types import StageResult
from uaqe.common.types import ExportFormat
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.domain.pipeline_stage import PipelineStage
from uaqe.runtime.runtime_selector import RuntimeSelector
from uaqe.exporter.exporter_factory import ExporterFactory

if TYPE_CHECKING:
    # Import cycle note: these collaborators each import
    # ``DeploymentArtifact`` from this module at module scope, so they
    # cannot be imported back here at runtime (see __init__'s lazy
    # imports below); TYPE_CHECKING keeps their names resolvable for
    # static type checkers without creating that cycle at import time.
    from uaqe.exporter.artifact_manifest import ArtifactManifestBuilder
    from uaqe.exporter.export_planner import ExportPlanner
    from uaqe.exporter.export_validator import ExportValidator
    from uaqe.exporter.package_builder import PackageBuilder

#: The context key this stage reads the final ``IMR`` from when
#: present, per :class:`~uaqe.optimizer.memory_optimizer.MemoryOptimizer`'s
#: own registered stage name — see the module docstring.
_MEMORY_OPTIMIZER_STAGE_NAME = "memory_optimizer"

#: Fallback context keys, in resolution order, when
#: ``"memory_optimizer"`` has not (yet) run — mirroring
#: ``MemoryOptimizer._resolve_current_imr``'s own fallback chain, with
#: ``"memory_optimizer"`` itself now the primary key rather than one of
#: the fallbacks.
_FALLBACK_STAGE_NAMES = (
    "optimizer",
    "compression_planner",
    "quantization_planner",
    "model_loader",
)

#: The ``HardwareProfile``-producing stage this stage exports against.
_HARDWARE_STAGE_NAME = "hardware_manager"

#: The subdirectory (under a run's export output directory) that
#: manifests are written into.
_MANIFEST_SUBDIR = "manifests"


@dataclass(frozen=True)
class DeploymentArtifact:
    """The complete, produced output of one export.

    Superset of the locked minimal shape (``file_paths``,
    ``export_format``, ``target_profile_id``, ``size_bytes`` —
    ``03_API_Specification.md`` §9.1) with the additional fields this
    package's manifest/packaging steps need, following the same
    precedent ``uaqe.compression.compression_planner.CompressionPlan``
    already set for its own locked counterpart.

    Attributes:
        file_paths: Every artifact file path a backend wrote, in the
            order the backend produced them. Populated by the backend
            itself; never includes ``manifest_path`` or
            ``package_path``.
        export_format: The ``ExportFormat`` produced.
        target_profile_id: The ``HardwareProfile.profile_id`` this
            artifact was produced for.
        size_bytes: The total size, in bytes, of ``file_paths`` (as
            returned by the backend — for a multi-file backend this is
            the sum across all of them).
        manifest_path: The path of the written ``manifest.json``, filled
            in by :class:`Exporter` after the backend returns; ``""``
            until then.
        package_path: The path of the assembled deployment package,
            filled in by :class:`Exporter` if packaging is enabled;
            ``None`` if packaging was skipped or has not yet run.
        warnings: Non-fatal warnings accumulated while producing this
            artifact.
    """

    file_paths: List[str] = field(default_factory=list)
    export_format: ExportFormat = ExportFormat.BIN
    target_profile_id: str = ""
    size_bytes: int = 0
    manifest_path: str = ""
    package_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class Exporter(PipelineStage):
    """Selects the target's export backend and produces its
    :class:`DeploymentArtifact`.

    Attributes:
        _backends: Every available ``IExporterBackend``, keyed by the
            ``ExportFormat`` it declares via its ``EXPORT_FORMAT`` class
            attribute (the convention every first-party backend in this
            package follows).
        _logger: Structured logging sink.
        _planner: Decides which ``ExportFormat`` to target.
        _validator: Re-validates the final ``IMR`` against the target
            immediately before export.
        _manifest_builder: Checksums the written artifact files into a
            manifest.
        _package_builder: Bundles the artifact files and manifest into
            one deployment archive; ``None`` to skip packaging
            entirely and leave loose files plus a manifest.
        _output_dir: Base directory this run's manifests (and, via
            ``_package_builder``, packages) are written under.
    """

    def __init__(
        self,
        backends: List[IExporterBackend],
        logger: ILogger,
        planner: Optional["ExportPlanner"] = None,  # noqa: F821
        validator: Optional["ExportValidator"] = None,  # noqa: F821
        manifest_builder: Optional["ArtifactManifestBuilder"] = None,  # noqa: F821
        package_builder: Optional["PackageBuilder"] = None,  # noqa: F821
        output_dir: str = "outputs/exports",
        requested_runtime: Optional[str] = None,
        model_path: Optional[str] = None,
        calibration_dataset_path: Optional[str] = None,
    ) -> None:
        """Initialize the ``Exporter``.

        Args:
            backends: Every registered ``IExporterBackend`` (first-party
                or plugin). Each must expose an ``EXPORT_FORMAT`` class
                attribute; which entries are actually needed depends on
                the resolved ``HardwareProfile`` and is validated lazily
                in :meth:`select_backend`, not here.
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
            planner: The format-selection collaborator; a fresh
                :class:`~uaqe.exporter.export_planner.ExportPlanner` is
                constructed if omitted.
            validator: The pre-export validation collaborator; a fresh
                :class:`~uaqe.exporter.export_validator.ExportValidator`
                is constructed if omitted.
            manifest_builder: The checksum-manifest collaborator; a
                fresh :class:`~uaqe.exporter.artifact_manifest.
                ArtifactManifestBuilder` is constructed if omitted.
            package_builder: The final-packaging collaborator; a fresh
                :class:`~uaqe.exporter.package_builder.PackageBuilder`
                is constructed if omitted. Pass ``package_builder=None``
                is not distinguishable from "use the default" — to skip
                packaging entirely, construct with a builder whose
                ``build`` is a no-op, or simply ignore
                ``DeploymentArtifact.package_path`` downstream.
            output_dir: Base directory this run's manifests and
                packages are written under.
            requested_runtime: Target runtime requested via override.
            model_path: Path to the original source model.
        """
        # Imported lazily, inside __init__, to avoid a module-level
        # import cycle: export_report.py and package_builder.py both
        # import DeploymentArtifact from *this* module, so this module
        # cannot import them back at module scope.
        from uaqe.exporter.artifact_manifest import ArtifactManifestBuilder
        from uaqe.exporter.export_planner import ExportPlanner
        from uaqe.exporter.export_validator import ExportValidator
        from uaqe.exporter.package_builder import PackageBuilder

        self._backends: Dict[ExportFormat, IExporterBackend] = {
            backend.EXPORT_FORMAT: backend for backend in backends  # type: ignore[attr-defined]
        }
        self._logger = logger
        self._planner = planner or ExportPlanner()
        self._validator = validator or ExportValidator()
        self._manifest_builder = manifest_builder or ArtifactManifestBuilder()
        self._package_builder = package_builder or PackageBuilder()
        self._output_dir = output_dir
        self._requested_runtime = requested_runtime
        self._model_path = model_path
        self._calibration_dataset_path = calibration_dataset_path
        self._runtime_selector = RuntimeSelector()
        self._exporter_factory = ExporterFactory(backends)

    def execute(self, context: PipelineContext) -> StageResult:
        """Export this run's final ``IMR`` to its target's deployment
        artifact.

        Reads the current ``IMR`` per the resolution order documented
        in the module docstring, and the target ``HardwareProfile``
        from ``"hardware_manager"``.

        Args:
            context: The current run's pipeline context.

        Returns:
            A ``StageResult`` whose ``payload`` is a tuple of
            ``(DeploymentArtifact, IMR)`` — the finalized artifact
            (with ``manifest_path``/``package_path`` populated) and the
            same, unmodified final ``IMR`` it was produced from.

        Raises:
            ExportError: If validation fails, no backend is registered
                for the planned format, or writing any output file
                fails.
        """
        start = time.monotonic()
        imr = self._resolve_current_imr(context)
        profile = context.get(_HARDWARE_STAGE_NAME).payload

        # 1. RuntimeSelector selects runtime
        runtime = self._runtime_selector.select_runtime(profile, self._requested_runtime)
        
        # 2. ExporterFactory returns the concrete backend for the runtime
        backend = self._exporter_factory.get_exporter(runtime)

        # 3. Temporarily bind the source model path and calibration dataset path to the backend if supported
        if hasattr(backend, "bind_source_model_path"):
            backend.bind_source_model_path(self._model_path)

        cal_path = self._calibration_dataset_path
        if not cal_path and context.has("workflow_config"):
            w_config = context.get("workflow_config").payload
            cal_path = getattr(w_config, "calibration_dataset_path", None)

        if hasattr(backend, "bind_calibration_dataset_path"):
            backend.bind_calibration_dataset_path(cal_path)
            
        if hasattr(backend, "_output_dir"):
            backend._output_dir = self._output_dir

        plan = self._planner.build_plan(profile, requested_format=backend.EXPORT_FORMAT)
        self._validator.validate(imr, profile, plan.selected_format)

        artifact = backend.export(imr, profile)

        manifest = self._manifest_builder.build(imr, profile, artifact)
        manifest_dir = f"{self._output_dir}/{_MANIFEST_SUBDIR}/{profile.profile_id}"
        manifest_path = self._manifest_builder.write(manifest, manifest_dir)
        artifact = replace(artifact, manifest_path=manifest_path)

        package_path = self._package_builder.build(
            artifact, manifest_path, profile, self._output_dir
        )
        artifact = replace(artifact, package_path=package_path)

        duration_ms = (time.monotonic() - start) * 1000.0
        self._logger.info(
            "Export completed.",
            stage_name=self.name(),
            target_profile_id=profile.profile_id,
            export_format=plan.selected_format.value,
            file_paths=artifact.file_paths,
            package_path=artifact.package_path,
            duration_ms=duration_ms,
        )
        return StageResult(
            stage_name=self.name(),
            success=True,
            payload=(artifact, imr),
            duration_ms=duration_ms,
        )

    def select_backend(self, profile: HardwareProfile) -> IExporterBackend:
        """Resolve the ``IExporterBackend`` registered for ``profile``.

        Args:
            profile: The resolved deployment target.

        Returns:
            The ``IExporterBackend`` registered for the ``ExportFormat``
            :class:`~uaqe.exporter.export_planner.ExportPlanner` selects
            for ``profile``.

        Raises:
            ExportError: If no backend is registered for that format.
        """
        plan = self._planner.build_plan(profile)
        return self._select_backend_for_format(plan.selected_format)

    def _select_backend_for_format(self, export_format: ExportFormat) -> IExporterBackend:
        """Look up the registered backend for ``export_format``.

        Args:
            export_format: The format to resolve.

        Returns:
            The registered ``IExporterBackend``.

        Raises:
            ExportError: If ``export_format`` has no registered backend.
        """
        backend = self._backends.get(export_format)
        if backend is None:
            raise ExportError(
                f"No registered IExporterBackend for "
                f"ExportFormat.{export_format.name}. Registered formats: "
                f"{[fmt.value for fmt in self._backends]}.",
                code="EXPORT_MISSING_BACKEND",
            )
        return backend

    def _resolve_current_imr(self, context: PipelineContext) -> IMR:
        """Resolve the final ``IMR`` from ``context``.

        Args:
            context: The current run's pipeline context.

        Returns:
            The ``IMR`` produced by ``"memory_optimizer"`` if that stage
            has run, else the first of :data:`_FALLBACK_STAGE_NAMES`
            found in ``context``.

        Raises:
            KeyError: If none of ``"memory_optimizer"`` or
                :data:`_FALLBACK_STAGE_NAMES` has been recorded in
                ``context``.
        """
        if context.has(_MEMORY_OPTIMIZER_STAGE_NAME):
            payload = context.get(_MEMORY_OPTIMIZER_STAGE_NAME).payload
            return payload[1] if isinstance(payload, tuple) else payload

        for stage_name in _FALLBACK_STAGE_NAMES:
            if context.has(stage_name):
                payload = context.get(stage_name).payload
                return payload[1] if isinstance(payload, tuple) else payload

        raise KeyError(
            "No IMR-producing stage recorded in context; expected one of "
            f"{(_MEMORY_OPTIMIZER_STAGE_NAME,) + _FALLBACK_STAGE_NAMES!r}."
        )
