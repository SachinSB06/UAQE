"""Intel HEX (``ExportFormat.HEX``) exporter backend.

Encodes the same flat parameter buffer
:mod:`~uaqe.exporter.binary_exporter` produces as a standard Intel HEX
text file — the record format most MCU flash-programming toolchains
(e.g. ``avrdude``, vendor flash loaders referenced by
``05_Hardware_Profile_Spec.md`` §4's ``runtime`` values) accept
directly, unlike a raw ``.bin``.
"""

from __future__ import annotations

import os
from typing import List, Sequence

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

#: Bytes of payload encoded per Intel HEX data record (``:10...``),
#: the conventional record length used by most Intel HEX consumers.
_BYTES_PER_RECORD = 16

#: Intel HEX record type codes used by :func:`_to_intel_hex`.
_RECORD_TYPE_DATA = 0x00
_RECORD_TYPE_EOF = 0x01
_RECORD_TYPE_EXTENDED_LINEAR_ADDRESS = 0x04


def _checksum(record_bytes: bytes) -> int:
    """Compute an Intel HEX record's two's-complement checksum byte."""
    return (-sum(record_bytes)) & 0xFF


def _format_record(record_type: int, address: int, payload: bytes) -> str:
    """Format one Intel HEX record line (without the trailing newline)."""
    header = bytes([len(payload), (address >> 8) & 0xFF, address & 0xFF, record_type])
    body = header + payload
    return f":{body.hex()}{_checksum(body):02X}".upper()


def _to_intel_hex(data: bytes) -> str:
    """Encode ``data`` as a complete Intel HEX file body.

    Emits an Extended Linear Address record (type ``04``) whenever the
    running byte offset crosses a 64 KiB boundary, so buffers larger
    than 65,536 bytes still address correctly rather than silently
    wrapping (a common Intel HEX authoring bug).

    Args:
        data: The raw bytes to encode, typically
            :func:`~uaqe.exporter.binary_exporter.serialize_parameters`'s
            output.

    Returns:
        The full Intel HEX file contents, including the terminating
        End Of File record, newline-terminated.
    """
    lines: List[str] = []
    current_upper_address = -1
    for offset in range(0, len(data), _BYTES_PER_RECORD):
        chunk = data[offset : offset + _BYTES_PER_RECORD]
        upper_address = (offset >> 16) & 0xFFFF
        if upper_address != current_upper_address:
            lines.append(
                _format_record(
                    _RECORD_TYPE_EXTENDED_LINEAR_ADDRESS,
                    0,
                    upper_address.to_bytes(2, "big"),
                )
            )
            current_upper_address = upper_address
        lines.append(_format_record(_RECORD_TYPE_DATA, offset & 0xFFFF, chunk))
    lines.append(_format_record(_RECORD_TYPE_EOF, 0, b""))
    return "\n".join(lines) + "\n"


class HexExporter(IExporterBackend):
    """``ExportFormat.HEX`` backend: an Intel HEX text encoding of the
    same raw parameter bytes :class:`~uaqe.exporter.binary_exporter.
    BinaryExporter` produces.

    Attributes:
        EXPORT_FORMAT: This backend's ``ExportFormat``.
    """

    EXPORT_FORMAT = ExportFormat.HEX

    def __init__(
        self,
        logger: ILogger,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        supported_target_profile_ids: Sequence[str] = (),
    ) -> None:
        """Initialize the ``HexExporter``.

        Args:
            logger: Structured logging sink.
            output_dir: Base directory artifact files are written under.
            supported_target_profile_ids: See
                :class:`~uaqe.exporter.binary_exporter.BinaryExporter`'s
                constructor docstring for how this is wired.
        """
        self._logger = logger
        self._output_dir = output_dir
        self._supported_target_profile_ids: List[str] = list(
            supported_target_profile_ids
        )

    def export(self, imr: IMR, target: HardwareProfile) -> DeploymentArtifact:
        """Serialize ``imr`` to one Intel HEX (``.hex``) file for ``target``.

        Args:
            imr: The final, fully-optimized model to export.
            target: The resolved hardware profile to export for.

        Returns:
            A ``DeploymentArtifact`` naming the single written file.

        Raises:
            ExportError: If the file cannot be written.
        """
        blob = serialize_parameters(imr)
        text = _to_intel_hex(blob)
        path = os.path.join(self._output_dir, target.profile_id, "model.hex")
        write_artifact_file(path, text.encode("ascii"))
        self._logger.info(
            "Exported HEX artifact.",
            target_profile_id=target.profile_id,
            path=path,
            source_bytes=len(blob),
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
