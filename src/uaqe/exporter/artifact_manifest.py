"""Checksum manifest for a completed export.

``ArtifactManifestBuilder`` runs immediately after a backend's
``export()`` call, reading back every file it wrote
(``DeploymentArtifact.file_paths``) to record each one's size and
SHA-256 checksum alongside a snapshot of the source ``IMR``'s
provenance. The written ``manifest.json`` is what
:mod:`~uaqe.exporter.package_builder` bundles alongside the artifact
files themselves, and what a deployment/flashing script can use to
verify a downloaded artifact was not corrupted or swapped before
writing it to a device.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from uaqe.common.exceptions import ExportError
from uaqe.common.imr import IMR
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.exporter.exporter import DeploymentArtifact

#: The manifest schema version written to every
#: :class:`ArtifactManifest`, bumped whenever :meth:`ArtifactManifest.to_dict`'s
#: shape changes in a way a reader needs to branch on.
MANIFEST_SCHEMA_VERSION = "1.0"

#: The hashing algorithm used for every :class:`ManifestFileEntry`.
CHECKSUM_ALGORITHM = "sha256"

#: The chunk size used while streaming a file through the checksum
#: hasher, chosen to bound peak memory use for large artifacts without
#: adding meaningful per-call overhead.
_HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ManifestFileEntry:
    """One artifact file's manifest record.

    Attributes:
        relative_path: The file's path relative to the manifest's own
            directory, so the manifest remains valid if the whole
            export directory is moved or copied as a unit.
        size_bytes: The file's size on disk, in bytes.
        checksum: The file's checksum, hex-encoded, computed with
            :data:`CHECKSUM_ALGORITHM`.
    """

    relative_path: str
    size_bytes: int
    checksum: str


@dataclass(frozen=True)
class ArtifactManifest:
    """A complete, self-describing record of one export's output files.

    Attributes:
        schema_version: See :data:`MANIFEST_SCHEMA_VERSION`.
        target_profile_id: The ``HardwareProfile.profile_id`` this
            export was produced for.
        export_format: The ``ExportFormat.value`` this export used.
        source_framework: The originating framework of the exported
            model, carried through from ``IMRMetadata``.
        source_format: The originating file format/extension of the
            exported model, carried through from ``IMRMetadata``.
        op_count: The number of layers in the exported model.
        total_parameters: The total learnable parameter element count
            in the exported model.
        generated_at: UTC ISO-8601 timestamp of when this manifest was
            built.
        checksum_algorithm: See :data:`CHECKSUM_ALGORITHM`.
        files: One :class:`ManifestFileEntry` per file in
            ``DeploymentArtifact.file_paths``.
    """

    target_profile_id: str
    export_format: str
    source_framework: str
    source_format: str
    op_count: int
    total_parameters: int
    generated_at: str
    files: List[ManifestFileEntry] = field(default_factory=list)
    schema_version: str = MANIFEST_SCHEMA_VERSION
    checksum_algorithm: str = CHECKSUM_ALGORITHM

    def to_dict(self) -> Dict[str, Any]:
        """Return this manifest as a plain, JSON-serializable ``dict``."""
        return {
            "schema_version": self.schema_version,
            "target_profile_id": self.target_profile_id,
            "export_format": self.export_format,
            "source_framework": self.source_framework,
            "source_format": self.source_format,
            "op_count": self.op_count,
            "total_parameters": self.total_parameters,
            "generated_at": self.generated_at,
            "checksum_algorithm": self.checksum_algorithm,
            "files": [
                {
                    "relative_path": entry.relative_path,
                    "size_bytes": entry.size_bytes,
                    "checksum": entry.checksum,
                }
                for entry in self.files
            ],
        }

    def to_json(self) -> str:
        """Render this manifest as indented, deterministic JSON text."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n"


def _sha256_of_file(path: str) -> str:
    """Compute the hex-encoded SHA-256 checksum of the file at ``path``.

    Args:
        path: The file to checksum.

    Returns:
        The hex-encoded digest.

    Raises:
        ExportError: If ``path`` cannot be opened or read.
    """
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
                hasher.update(chunk)
    except OSError as exc:
        raise ExportError(
            f"Failed to checksum export artifact at {path!r}: {exc}",
            code="EXPORT_MANIFEST_READ_FAILED",
        ) from exc
    return hasher.hexdigest()


class ArtifactManifestBuilder:
    """Builds and persists an :class:`ArtifactManifest` for a completed
    export.
    """

    def build(
        self, imr: IMR, target: HardwareProfile, artifact: DeploymentArtifact
    ) -> ArtifactManifest:
        """Build the manifest for ``artifact``'s already-written files.

        Args:
            imr: The exported model, for provenance metadata.
            target: The resolved deployment target ``artifact`` was
                produced for.
            artifact: The ``DeploymentArtifact`` returned by the backend
                that ran; every path in ``artifact.file_paths`` must
                already exist on disk.

        Returns:
            The built :class:`ArtifactManifest`, not yet written to disk
            (see :meth:`write`).

        Raises:
            ExportError: If any file in ``artifact.file_paths`` cannot be
                read.
        """
        manifest_dir = os.path.dirname(artifact.file_paths[0]) if artifact.file_paths else ""
        entries = [
            ManifestFileEntry(
                relative_path=os.path.relpath(path, start=manifest_dir or "."),
                size_bytes=os.path.getsize(path),
                checksum=_sha256_of_file(path),
            )
            for path in artifact.file_paths
        ]
        return ArtifactManifest(
            target_profile_id=target.profile_id,
            export_format=artifact.export_format.value,
            source_framework=imr.metadata.source_framework,
            source_format=imr.metadata.source_format,
            op_count=imr.metadata.op_count,
            total_parameters=imr.metadata.total_parameters,
            generated_at=datetime.now(timezone.utc).isoformat(),
            files=entries,
        )

    def write(self, manifest: ArtifactManifest, directory: str) -> str:
        """Write ``manifest`` as ``manifest.json`` under ``directory``.

        Args:
            manifest: The manifest to persist.
            directory: The directory to write ``manifest.json`` into
                (created if missing).

        Returns:
            The full path of the written manifest file.

        Raises:
            ExportError: If the manifest cannot be written.
        """
        path = os.path.join(directory, "manifest.json")
        try:
            os.makedirs(directory, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(manifest.to_json())
        except OSError as exc:
            raise ExportError(
                f"Failed to write export manifest to {path!r}: {exc}",
                code="EXPORT_MANIFEST_WRITE_FAILED",
            ) from exc
        return path
