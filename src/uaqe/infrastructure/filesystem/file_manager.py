"""Byte/text file I/O against paths resolved by
:class:`~uaqe.infrastructure.filesystem.path_manager.PathManager`.

Per ``11_Implementation_Rules.md`` §5.3, all reads and writes under a
configured root (``workdir/``, ``outputs/``, ``reports/``, ``logs/``)
must go through path-containment checks before any file operation
occurs. ``FileManager`` performs the actual I/O; it never constructs
or validates a path itself — that is delegated entirely to an
injected ``PathManager``, keeping each class to a single
responsibility (``11_Implementation_Rules.md`` §2, §19.1).

``FileManager`` is an internal collaborator of
:mod:`uaqe.infrastructure.filesystem.filesystem_repository`, not a
name in the locked class inventory of ``09_Architecture_Lock.md`` §8
(which names only ``FilesystemRepository`` for this area) — it is not
constructed or used outside this package.
"""

from __future__ import annotations

from uaqe.common.exceptions import ConfigurationError
from uaqe.infrastructure.filesystem.path_manager import PathManager


class FileManager:
    """Reads and writes file content at paths validated by a ``PathManager``.

    Attributes:
        path_manager: The collaborator this manager delegates all path
            resolution and containment checks to. Never constructed
            internally (``11_Implementation_Rules.md`` §3.1 —
            constructor injection only).
    """

    def __init__(self, path_manager: PathManager) -> None:
        """Initialize a ``FileManager``.

        Args:
            path_manager: The ``PathManager`` used to resolve every
                ``relative_path`` argument against its configured root
                and reject any path that would escape it.
        """
        self.path_manager: PathManager = path_manager

    def write_bytes(self, relative_path: str, data: bytes) -> str:
        """Write ``data`` to ``relative_path`` under the configured root.

        Args:
            relative_path: Path, relative to the ``PathManager``'s
                root, to write to. Parent directories are created as
                needed.
            data: The raw bytes to write.

        Returns:
            The absolute filesystem path written to, as a string.

        Raises:
            ConfigurationError: If ``relative_path`` would resolve
                outside the configured root, or if the write fails
                (e.g. permission denied, disk full).
        """
        target = self.path_manager.resolve(relative_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        except OSError as exc:
            raise ConfigurationError(
                f"Cannot write bytes to {relative_path!r}.",
                code="FILESYSTEM_WRITE_FAILED",
                remediation_hint="Verify the destination path is writable and has free space.",
            ) from exc
        return str(target)

    def write_text(self, relative_path: str, text: str) -> str:
        """Write ``text`` to ``relative_path`` under the configured root.

        Args:
            relative_path: Path, relative to the ``PathManager``'s
                root, to write to. Parent directories are created as
                needed.
            text: The text content to write, encoded as UTF-8.

        Returns:
            The absolute filesystem path written to, as a string.

        Raises:
            ConfigurationError: If ``relative_path`` would resolve
                outside the configured root, or if the write fails.
        """
        target = self.path_manager.resolve(relative_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(
                f"Cannot write text to {relative_path!r}.",
                code="FILESYSTEM_WRITE_FAILED",
                remediation_hint="Verify the destination path is writable and has free space.",
            ) from exc
        return str(target)

    def read_bytes(self, relative_path: str) -> bytes:
        """Read and return the raw bytes at ``relative_path``.

        Args:
            relative_path: Path, relative to the ``PathManager``'s
                root, to read.

        Returns:
            The file's raw byte content.

        Raises:
            FileNotFoundError: If no file exists at ``relative_path``.
                Matches the locked contract in
                ``03_API_Specification.md`` §17.3 exactly, unlike this
                manager's write methods, which raise
                ``ConfigurationError`` for every other failure mode.
            ConfigurationError: If ``relative_path`` would resolve
                outside the configured root, or if the read fails for
                a reason other than the file being absent (e.g.
                permission denied).
        """
        target = self.path_manager.resolve(relative_path)
        if not target.exists():
            raise FileNotFoundError(
                f"No file at {relative_path!r} under {self.path_manager.base_path}."
            )
        try:
            return target.read_bytes()
        except OSError as exc:
            raise ConfigurationError(
                f"Cannot read bytes from {relative_path!r}.",
                code="FILESYSTEM_READ_FAILED",
                remediation_hint="Verify the source path is readable.",
            ) from exc

    def read_text(self, relative_path: str) -> str:
        """Read and return the UTF-8 text content at ``relative_path``.

        Args:
            relative_path: Path, relative to the ``PathManager``'s
                root, to read.

        Returns:
            The file's decoded text content.

        Raises:
            FileNotFoundError: If no file exists at ``relative_path``.
            ConfigurationError: If ``relative_path`` would resolve
                outside the configured root, or if the read fails for
                a reason other than the file being absent (e.g.
                permission denied, invalid encoding).
        """
        target = self.path_manager.resolve(relative_path)
        if not target.exists():
            raise FileNotFoundError(
                f"No file at {relative_path!r} under {self.path_manager.base_path}."
            )
        try:
            return target.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(
                f"Cannot read text from {relative_path!r}.",
                code="FILESYSTEM_READ_FAILED",
                remediation_hint="Verify the source path is readable and UTF-8 encoded.",
            ) from exc
