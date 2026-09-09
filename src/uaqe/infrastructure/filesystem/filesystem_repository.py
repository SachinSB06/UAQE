"""Sole component permitted to perform raw path construction under a
configured root such as ``workdir/``, ``outputs/``, ``reports/``, or
``logs/``.

Per ``11_Implementation_Rules.md`` §5.3 and ``04_Data_Flow.md``, no
other class calls ``os.path``/``pathlib`` directly against these
roots; ``FilesystemRepository`` is injected only into ``Exporter``,
``ReportGenerator``, and ``PipelineOrchestrator`` (for checkpointing).
Every path this class resolves is validated to stay contained inside
its configured root, which is the primary defense against path
traversal referenced in ``12_Project_Test_Plan.md`` §13.

Locked contract: ``03_API_Specification.md`` §17.3.
"""

from __future__ import annotations

from pathlib import Path

from uaqe.common.exceptions import ConfigurationError


class FilesystemRepository:
    """Path-contained file I/O for a single configured root directory.

    Attributes:
        base_path: The absolute, resolved root directory every
            ``relative_path`` argument is resolved against. No path
            this repository produces or accesses may resolve outside
            of it.
    """

    def __init__(self, base_path: str) -> None:
        """Initialize a ``FilesystemRepository`` rooted at ``base_path``.

        Args:
            base_path: The root directory (e.g. ``config.json``'s
                ``outputs_path``) all relative paths are resolved
                against. Created if it does not already exist.

        Raises:
            ConfigurationError: If ``base_path`` cannot be created or
                resolved (e.g. a parent segment exists as a file, or
                permissions deny directory creation).
        """
        try:
            resolved = Path(base_path).resolve()
            resolved.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigurationError(
                f"Cannot create or resolve filesystem repository root {base_path!r}.",
                code="FILESYSTEM_ROOT_UNAVAILABLE",
                remediation_hint="Verify the configured path is writable and not a file.",
            ) from exc
        self.base_path: Path = resolved

    def write_bytes(self, relative_path: str, data: bytes) -> str:
        """Write ``data`` to ``relative_path`` under ``base_path``.

        Args:
            relative_path: Path, relative to ``base_path``, to write
                to. Parent directories are created as needed.
            data: The raw bytes to write.

        Returns:
            The absolute filesystem path written to, as a string.

        Raises:
            ConfigurationError: If ``relative_path`` would resolve
                outside ``base_path``, or if the write fails (e.g.
                permission denied, disk full).
        """
        target = self._resolve(relative_path)
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
        """Write ``text`` to ``relative_path`` under ``base_path``.

        Args:
            relative_path: Path, relative to ``base_path``, to write
                to. Parent directories are created as needed.
            text: The text content to write, encoded as UTF-8.

        Returns:
            The absolute filesystem path written to, as a string.

        Raises:
            ConfigurationError: If ``relative_path`` would resolve
                outside ``base_path``, or if the write fails.
        """
        target = self._resolve(relative_path)
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
            relative_path: Path, relative to ``base_path``, to read.

        Returns:
            The file's raw byte content.

        Raises:
            FileNotFoundError: If no file exists at ``relative_path``.
                Matches the locked contract in
                ``03_API_Specification.md`` §17.3 exactly, unlike this
                repository's other methods, which raise
                ``ConfigurationError`` for every other failure mode.
            ConfigurationError: If ``relative_path`` would resolve
                outside ``base_path``, or if the read fails for a
                reason other than the file being absent (e.g.
                permission denied).
        """
        target = self._resolve(relative_path)
        if not target.exists():
            raise FileNotFoundError(f"No file at {relative_path!r} under {self.base_path}.")
        try:
            return target.read_bytes()
        except OSError as exc:
            raise ConfigurationError(
                f"Cannot read bytes from {relative_path!r}.",
                code="FILESYSTEM_READ_FAILED",
                remediation_hint="Verify the source path is readable.",
            ) from exc

    def _resolve(self, relative_path: str) -> Path:
        """Resolve ``relative_path`` against ``base_path``, enforcing containment.

        Args:
            relative_path: The caller-supplied relative path.

        Returns:
            The resolved absolute ``Path``, guaranteed to lie inside
            ``base_path``.

        Raises:
            ConfigurationError: If the resolved path would lie outside
                ``base_path`` (path traversal attempt).
        """
        candidate = (self.base_path / relative_path).resolve()
        if not candidate.is_relative_to(self.base_path):
            raise ConfigurationError(
                f"Path {relative_path!r} resolves outside the repository root "
                f"{self.base_path}.",
                code="PATH_TRAVERSAL_REJECTED",
                remediation_hint="Use a relative_path that stays within the configured root.",
            )
        return candidate
