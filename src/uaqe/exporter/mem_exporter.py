"""FPGA memory-initialization (``ExportFormat.MEM``) exporter backend.

Encodes the same flat parameter buffer
:mod:`~uaqe.exporter.binary_exporter` produces as a plain-hex
``$readmemh``-compatible ``.mem`` file: one whitespace-separated hex
word per line, no addresses, matching the file Verilog/VHDL block-RAM
initialization directives expect (the format
``HardwareProfile.fpga_resources``-bearing profiles' ``runtime`` values
like ``"bare-metal-hdl"`` are built around).
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
from uaqe.exporter.binary_exporter import (
    DEFAULT_OUTPUT_DIR,
    serialize_parameters,
    write_artifact_file,
)
from uaqe.exporter.exporter import DeploymentArtifact

#: Default BRAM word width, in bytes, used to group the flat parameter
#: buffer into ``$readmemh`` words when a profile's
#: ``fpga_resources`` does not otherwise imply one. One byte per word
#: is the safest, most portable default: it never requires padding and
#: matches every ``bram_kb``-only profile that does not (yet) surface
#: an explicit data-bus width.
_DEFAULT_WORD_WIDTH_BYTES = 1


def _to_readmemh(data: bytes, word_width_bytes: int) -> str:
    """Encode ``data`` as a ``$readmemh``-compatible hex-word text body.

    Args:
        data: The raw bytes to encode.
        word_width_bytes: The number of bytes grouped into each emitted
            hex word (big-endian within a word). ``data`` is zero-padded
            on its final word if its length is not an exact multiple of
            ``word_width_bytes``.

    Returns:
        One upper-case hex word per line, newline-terminated. Empty
        string for empty ``data``.

    Raises:
        ExportError: If ``word_width_bytes`` is not positive.
    """
    if word_width_bytes <= 0:
        raise ExportError(
            f"word_width_bytes must be positive, got {word_width_bytes}.",
            code="EXPORT_INVALID_WORD_WIDTH",
        )

    padding = (-len(data)) % word_width_bytes
    padded = data + b"\x00" * padding

    lines: List[str] = []
    for offset in range(0, len(padded), word_width_bytes):
        word = padded[offset : offset + word_width_bytes]
        lines.append(word.hex().upper())
    return "\n".join(lines) + ("\n" if lines else "")


class MemExporter(IExporterBackend):
    """``ExportFormat.MEM`` backend: a ``$readmemh``-compatible ``.mem``
    text encoding of the same raw parameter bytes
    :class:`~uaqe.exporter.binary_exporter.BinaryExporter` produces.

    Attributes:
        EXPORT_FORMAT: This backend's ``ExportFormat``.
    """

    EXPORT_FORMAT = ExportFormat.MEM

    def __init__(
        self,
        logger: ILogger,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        word_width_bytes: int = _DEFAULT_WORD_WIDTH_BYTES,
        supported_target_profile_ids: Sequence[str] = (),
    ) -> None:
        """Initialize the ``MemExporter``.

        Args:
            logger: Structured logging sink.
            output_dir: Base directory artifact files are written under.
            word_width_bytes: The hex-word width used by
                :func:`_to_readmemh`.
            supported_target_profile_ids: See
                :class:`~uaqe.exporter.binary_exporter.BinaryExporter`'s
                constructor docstring for how this is wired.
        """
        self._logger = logger
        self._output_dir = output_dir
        self._word_width_bytes = word_width_bytes
        self._supported_target_profile_ids: List[str] = list(
            supported_target_profile_ids
        )

    def export(self, imr: IMR, target: HardwareProfile) -> DeploymentArtifact:
        """Serialize ``imr`` to one ``$readmemh``-style ``.mem`` file for
        ``target``.

        Args:
            imr: The final, fully-optimized model to export.
            target: The resolved hardware profile to export for.

        Returns:
            A ``DeploymentArtifact`` naming the single written file.

        Raises:
            ExportError: If the file cannot be written.
        """
        blob = serialize_parameters(imr)
        text = _to_readmemh(blob, self._word_width_bytes)
        path = os.path.join(self._output_dir, target.profile_id, "model.mem")
        write_artifact_file(path, text.encode("ascii"))
        self._logger.info(
            "Exported MEM artifact.",
            target_profile_id=target.profile_id,
            path=path,
            source_bytes=len(blob),
            word_width_bytes=self._word_width_bytes,
        )
        return DeploymentArtifact(
            file_paths=[path],
            export_format=self.EXPORT_FORMAT,
            target_profile_id=target.profile_id,
            size_bytes=len(text.encode("ascii")),
        )

    def supported_targets(self) -> List[str]:
        """Return the ``HardwareProfile.profile_id`` values this backend
        supports.
        """
        return list(self._supported_target_profile_ids)
