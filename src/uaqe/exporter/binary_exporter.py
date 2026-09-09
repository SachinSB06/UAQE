"""Raw binary (``ExportFormat.BIN``) exporter backend, and the shared
parameter-serialization/file-write helpers every other first-party
backend in this package builds on.

``BinaryExporter`` is the reference backend
(:mod:`uaqe.exporter`'s own ``__init__`` docstring: ``\"BIN``: one flat
file of raw, concatenated parameter bytes\"``): it owns
:func:`serialize_parameters`, the single place an ``IMR``'s learned
parameters are flattened into one deterministic byte buffer, and
:func:`write_artifact_file`, the single place any backend in this
package actually touches the filesystem. Every other backend
(:class:`~uaqe.exporter.hex_exporter.HexExporter`,
:class:`~uaqe.exporter.mem_exporter.MemExporter`) imports both
functions from here rather than re-deriving them, so an Intel HEX file
and a ``.mem`` file for the same ``IMR`` are always encodings of
*exactly* the same underlying bytes a ``.bin`` export of that same
``IMR`` would produce — there is exactly one "what are this model's
serialized parameter bytes" answer in this package.
:class:`~uaqe.exporter.export_validator.ExportValidator` also imports
:func:`serialize_parameters` directly, for the same reason: its
final-size check must measure the same bytes a real export would
produce, not a second, possibly-drifting estimate.

:func:`serialize_parameters`'s ordering is deterministic — every
consumer (the checksums in
:class:`~uaqe.exporter.artifact_manifest.ArtifactManifestBuilder`, the
size check in ``ExportValidator``, repeat exports of an unchanged
``IMR``) depends on identical input producing identical bytes:

1. Layers are visited in :meth:`~uaqe.common.imr.IMR.topological_order`
   rather than ``imr.layers``' own list order, since that order is
   explicitly documented as not guaranteed
   (:class:`~uaqe.common.imr.IMR`'s own docstring).
2. Within one layer, parameters are visited in sorted-by-name order
   (``IMRLayer.parameters`` is a plain ``dict``; iteration order is
   insertion order, which is not itself a property this module wants
   to depend on).
3. Each parameter contributes exactly its ``IMRTensor.data`` bytes,
   concatenated with no padding, alignment, or separator — the
   \"no images, no vector graphics, no embedded fonts\" minimalism
   precedent :mod:`uaqe.report.pdf_report` sets for its own
   dependency-free format applies here too: no header, no shape/dtype
   metadata, no length prefix. A ``.bin``/``.hex``/``.mem`` artifact is
   consumed by a flash-programming toolchain or ``$readmemh``
   directive that expects a bare data stream; shape/dtype metadata (if
   a target's runtime needs it at all) is a ``HardwareProfile``/build-
   system concern outside this buffer, not encoded inside it.
"""

from __future__ import annotations

import os
from typing import List, Sequence

from uaqe.common.exceptions import ExportError
from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_exporter_backend import IExporterBackend
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.types import ExportFormat
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.exporter.exporter import DeploymentArtifact

#: Default base directory every first-party backend in this package
#: writes artifact files under, matching ``config.json``'s
#: ``outputs_path`` default (``06_Config_Spec.md`` §1) joined with this
#: package's own ``exports`` subdirectory (mirroring
#: ``Exporter``'s own ``output_dir`` default of
#: ``\"outputs/exports\"``).
DEFAULT_OUTPUT_DIR: str = "outputs/exports"


def serialize_parameters(imr: IMR) -> bytes:
    """Flatten every learned parameter in ``imr`` into one deterministic
    byte buffer.

    See the module docstring for the exact, deterministic ordering
    this function guarantees and why no header/metadata is included.

    Args:
        imr: The final, fully-optimized model to serialize.

    Returns:
        The concatenated raw bytes of every parameter tensor across
        every layer, in topological-layer then sorted-parameter-name
        order. ``b\"\"`` for an ``IMR`` with no layers or no parameters.
    """
    buffer = bytearray()
    for layer in imr.topological_order():
        for parameter_name in sorted(layer.parameters):
            buffer += layer.parameters[parameter_name].data
    return bytes(buffer)


def write_artifact_file(path: str, data: bytes) -> None:
    """Write ``data`` to ``path``, creating parent directories as needed.

    The sole file-write call site shared by every first-party backend
    in this package (see the module docstring for why); no backend
    calls ``open()`` a second, independent way.

    Args:
        path: The absolute or relative filesystem path to write to.
        data: The raw bytes to write.

    Raises:
        ExportError: If the write fails (e.g. permission denied, disk
            full, or an intermediate path segment already exists as a
            file).
    """
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(data)
    except OSError as exc:
        raise ExportError(
            f"Cannot write export artifact to {path!r}: {exc}.",
            code="EXPORT_WRITE_FAILED",
            remediation_hint="Verify the destination path is writable and has free space.",
        ) from exc


class BinaryExporter(IExporterBackend):
    """``ExportFormat.BIN`` backend: one flat file of
    :func:`serialize_parameters`'s raw, concatenated parameter bytes,
    with no additional encoding.

    Attributes:
        EXPORT_FORMAT: This backend's ``ExportFormat``.
    """

    EXPORT_FORMAT = ExportFormat.BIN

    def __init__(
        self,
        logger: ILogger,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        supported_target_profile_ids: Sequence[str] = (),
    ) -> None:
        """Initialize the ``BinaryExporter``.

        Args:
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
            output_dir: Base directory artifact files are written
                under; the final path is
                ``<output_dir>/<target.profile_id>/model.bin``.
            supported_target_profile_ids: The ``HardwareProfile.
                profile_id`` values this backend instance declares
                support for, returned by :meth:`supported_targets`.
                Empty by default: since a raw binary dump has no
                target-specific encoding, an empty sequence here is
                read as \"no explicit restriction\" by callers that
                consult :meth:`supported_targets` (mirroring
                ``HexExporter``'s and ``MemExporter``'s own identical
                default), not as \"supports nothing\" —
                :class:`~uaqe.exporter.exporter.Exporter` itself
                resolves backends by ``ExportFormat`` via
                ``EXPORT_FORMAT``, not by consulting this list.
        """
        self._logger = logger
        self._output_dir = output_dir
        self._supported_target_profile_ids: List[str] = list(
            supported_target_profile_ids
        )

    def export(self, imr: IMR, target: HardwareProfile) -> DeploymentArtifact:
        """Serialize ``imr`` to one raw ``.bin`` file for ``target``.

        Args:
            imr: The final, fully-optimized model to export.
            target: The resolved hardware profile to export for.

        Returns:
            A ``DeploymentArtifact`` naming the single written file.

        Raises:
            ExportError: If the file cannot be written.
        """
        blob = serialize_parameters(imr)
        path = os.path.join(self._output_dir, target.profile_id, "model.bin")
        write_artifact_file(path, blob)
        self._logger.info(
            "Exported BIN artifact.",
            target_profile_id=target.profile_id,
            path=path,
            size_bytes=len(blob),
        )
        return DeploymentArtifact(
            file_paths=[path],
            export_format=self.EXPORT_FORMAT,
            target_profile_id=target.profile_id,
            size_bytes=len(blob),
        )

    def supported_targets(self) -> List[str]:
        """Return the ``HardwareProfile.profile_id`` values this backend
        supports.
        """
        return list(self._supported_target_profile_ids)
