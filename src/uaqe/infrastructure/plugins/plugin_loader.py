"""``PluginLoader`` — a single-purpose wrapper around
``PluginRegistry.discover()`` for loading plugins from one or more
directories.

.. important::
    **This class is not part of the locked architecture.**
    ``09_Architecture_Lock.md`` §8's module ownership table for
    ``uaqe.infrastructure.plugins`` names exactly ``PluginRegistry``
    (plus the ``builtin/`` subpackage, per ``02_Folder_Structure.md``
    §17). ``PluginRegistry.discover()`` already performs the entire
    load — import, metadata validation, interface-version check, class
    resolution, instantiation, and registration
    (``11_Implementation_Rules.md`` §8.2-§8.5). This file was generated
    on explicit request despite that overlap; it has not been
    reconciled via the RFC process in §14, and should be formalized or
    removed before being wired into a real ``CompositionRoot``.

This class duplicates none of ``PluginRegistry``'s parsing/validation
logic — it holds no plugin state of its own and does not reimplement
``_discover_module``/``_import_module``/``_resolve_plugin_class``. Its
only job is to give "which directories get scanned, and in what order"
its own small, testable surface, separate from the registry data
structure itself — the same read/write separation
``PluginRepository`` already applies to ``PluginRegistry`` on the read
side, applied here on the load side instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from uaqe.common.exceptions import PluginLoadError
from uaqe.infrastructure.plugins.plugin_registry import PluginRegistry


class PluginLoader:
    """Loads plugins from disk into an already-constructed ``PluginRegistry``.

    Attributes:
        registry: The ``PluginRegistry`` every loaded plugin is
            registered into. Owned by whoever constructs this
            ``PluginLoader`` (typically ``CompositionRoot``); this
            class never constructs its own registry.
    """

    def __init__(self, registry: PluginRegistry) -> None:
        """Initialize a ``PluginLoader`` writing into ``registry``.

        Args:
            registry: The ``PluginRegistry`` to load discovered plugins
                into.
        """
        self.registry: PluginRegistry = registry

    def load_from_directory(self, plugin_dir: str) -> None:
        """Discover and register every plugin under ``plugin_dir``.

        Args:
            plugin_dir: The root ``plugins/`` directory to scan — see
                ``PluginRegistry.discover()`` for the exact locked
                subfolder layout this scans.

        Raises:
            PluginLoadError: If ``plugin_dir`` does not exist or is not
                a directory, or if ``PluginRegistry.discover()`` itself
                raises (malformed metadata, unsupported interface
                version, ambiguous plugin class, or a registration
                collision without ``allow_override=True``).
        """
        if not Path(plugin_dir).is_dir():
            raise PluginLoadError(
                f"Plugin directory {plugin_dir!r} does not exist or is not a "
                "directory.",
                code="PLUGIN_DIRECTORY_MISSING",
                remediation_hint=f"Create {plugin_dir} before loading plugins.",
            )
        self.registry.discover(plugin_dir)

    def load_from_directories(self, plugin_dirs: List[str]) -> None:
        """Discover and register every plugin under each directory in ``plugin_dirs``.

        Directories are loaded in the order given; a later directory's
        plugin may fail to register if it collides with one already
        registered from an earlier directory and does not declare
        ``allow_override=True`` (``PluginRegistry._register``).

        Args:
            plugin_dirs: The root ``plugins/`` directories to scan, in
                load order.

        Raises:
            PluginLoadError: If any directory does not exist or is not
                a directory, or if ``PluginRegistry.discover()`` raises
                for any plugin within it.
        """
        for plugin_dir in plugin_dirs:
            self.load_from_directory(plugin_dir)
