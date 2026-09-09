"""Path resolution and containment enforcement for a single configured
root directory.

Per ``11_Implementation_Rules.md`` §5.3, ``FilesystemRepository`` is the
sole component permitted to perform raw path construction under a
configured root such as ``workdir/``, ``outputs/``, ``reports/``, or
``logs/``; every path it resolves must be validated to stay contained
inside that root, which is the primary defense against path traversal
referenced in ``12_Project_Test_Plan.md`` §13 (Security Testing —
Path Traversal).

``PathManager`` is that containment logic extracted into its own
single-responsibility collaborator: it owns nothing but "given this
root, is this relative path safe, and what absolute path does it map
to." It is an internal collaborator of
:mod:`uaqe.infrastructure.filesystem.filesystem_repository`
and :mod:`uaqe.infrastructure.filesystem.file_manager`, not a name in
the locked class inventory of ``09_Architecture_Lock.md`` §8 (which
names only ``FilesystemRepository`` for this area) — it is not
constructed or used outside this package.
"""

from __future__ import annotations

from pathlib import Path

from uaqe.common.exceptions import ConfigurationError


class PathManager:
    """Resolves relative paths against a single root, rejecting escapes.

    ``PathManager`` never reads or writes file content; it only
    computes and validates filesystem paths. Actual I/O is the
    responsibility of :class:`~uaqe.infrastructure.filesystem.file_manager.FileManager`
    (``11_Implementation_Rules.md`` §2 — single responsibility per
    class).

    Attributes:
        base_path: The absolute, resolved root directory every
            ``relative_path`` argument is resolved against. No path
            this manager produces may resolve outside of it.
    """

    def __init__(self, base_path: str) -> None:
        """Initialize a ``PathManager`` rooted at ``base_path``.

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
                f"Cannot create or resolve filesystem root {base_path!r}.",
                code="FILESYSTEM_ROOT_UNAVAILABLE",
                remediation_hint="Verify the configured path is writable and not a file.",
            ) from exc
        self.base_path: Path = resolved

    def resolve(self, relative_path: str) -> Path:
        """Resolve ``relative_path`` against ``base_path``, enforcing containment.

        Args:
            relative_path: The caller-supplied path, relative to
                ``base_path``.

        Returns:
            The resolved absolute ``Path``, guaranteed to lie inside
            ``base_path``.

        Raises:
            ConfigurationError: If the resolved path would lie outside
                ``base_path`` (path traversal attempt), e.g. via a
                ``relative_path`` containing ``../`` segments that
                escape the root.
        """
        candidate = (self.base_path / relative_path).resolve()
        if not candidate.is_relative_to(self.base_path):
            raise ConfigurationError(
                f"Path {relative_path!r} resolves outside the filesystem root "
                f"{self.base_path}.",
                code="PATH_TRAVERSAL_REJECTED",
                remediation_hint="Use a relative_path that stays within the configured root.",
            )
        return candidate

    def exists(self, relative_path: str) -> bool:
        """Report whether ``relative_path`` exists under ``base_path``.

        Args:
            relative_path: The caller-supplied path, relative to
                ``base_path``.

        Returns:
            ``True`` if a file or directory exists at the resolved
            location, ``False`` otherwise.

        Raises:
            ConfigurationError: If ``relative_path`` would resolve
                outside ``base_path``.
        """
        return self.resolve(relative_path).exists()
