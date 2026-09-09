"""Final deployment-package assembly.

``PackageBuilder`` is the last step ``Exporter`` runs: it bundles every
file a backend wrote (``DeploymentArtifact.file_paths``) together with
the ``manifest.json``
:mod:`~uaqe.exporter.artifact_manifest` produced into one archive, so a
person deploying to a board has exactly one file to copy/download
rather than needing to know which loose files belong together.
"""

from __future__ import annotations

import os
import zipfile
from typing import Iterable

from uaqe.common.exceptions import ExportError
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.exporter.exporter import DeploymentArtifact

#: The subdirectory (under a run's export output directory) that
#: assembled packages are written into, keeping them clearly separated
#: from the loose per-format artifact files they bundle.
PACKAGE_SUBDIR = "packages"


class PackageBuilder:
    """Bundles a completed export's files into one archive."""

    def build(
        self,
        artifact: DeploymentArtifact,
        manifest_path: str,
        target: HardwareProfile,
        output_dir: str,
    ) -> str:
        """Assemble one zip archive containing ``artifact``'s files plus
        the manifest.

        Args:
            artifact: The completed ``DeploymentArtifact`` whose
                ``file_paths`` to bundle.
            manifest_path: The path of the ``manifest.json``
                :meth:`~uaqe.exporter.artifact_manifest.
                ArtifactManifestBuilder.write` produced for this export.
            target: The resolved deployment target this package is for.
            output_dir: The base export output directory this run is
                writing under; the package is written to
                ``{output_dir}/packages/{profile_id}_{format}.zip``.

        Returns:
            The path of the written zip archive.

        Raises:
            ExportError: If the archive cannot be written, or if any
                file in ``artifact.file_paths`` no longer exists.
        """
        package_dir = os.path.join(output_dir, PACKAGE_SUBDIR)
        package_name = (
            f"{target.profile_id}_{artifact.export_format.value.lower()}.zip"
        )
        package_path = os.path.join(package_dir, package_name)

        try:
            os.makedirs(package_dir, exist_ok=True)
            with zipfile.ZipFile(package_path, mode="w") as archive:
                self._add_files(archive, artifact.file_paths)
                archive.write(manifest_path, arcname=os.path.basename(manifest_path))
        except OSError as exc:
            raise ExportError(
                f"Failed to assemble deployment package at "
                f"{package_path!r}: {exc}",
                code="EXPORT_PACKAGE_WRITE_FAILED",
            ) from exc

        return package_path

    def _add_files(self, archive: zipfile.ZipFile, file_paths: Iterable[str]) -> None:
        """Add each path in ``file_paths`` to ``archive`` under its own
        basename.

        Args:
            archive: The open ``ZipFile`` to write into.
            file_paths: The artifact files to add.

        Raises:
            ExportError: If any path does not exist.
        """
        for path in file_paths:
            if not os.path.isfile(path):
                raise ExportError(
                    f"Expected export artifact file not found: {path!r}.",
                    code="EXPORT_PACKAGE_MISSING_FILE",
                )
            archive.write(path, arcname=os.path.basename(path))
