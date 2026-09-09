"""``uaqe.infrastructure.plugins`` — plugin discovery, loading, and lookup.

Per ``09_Architecture_Lock.md`` §8 / ``02_Folder_Structure.md`` §17,
this package's locked inventory is exactly ``PluginRegistry`` (plus the
``builtin/`` subpackage). ``PluginLoader`` and ``PluginManager`` are
**not** part of that locked inventory — see each module's own
docstring. They were added on explicit request, following the same
pattern already applied to ``PluginRepository``/``ModelRepository``/
``LoggerFactory`` elsewhere in this codebase: clearly flagged rather
than silently presented as locked, and not wired into anything by
default.
"""

from uaqe.infrastructure.plugins.plugin_registry import PluginRegistry
from uaqe.infrastructure.plugins.plugin_loader import PluginLoader
from uaqe.infrastructure.plugins.plugin_manager import PluginManager

__all__ = [
    # --- locked inventory (09_Architecture_Lock.md §8) ---
    "PluginRegistry",
    # --- not part of the locked inventory; see each module's own
    # docstring for the caveat ---
    "PluginLoader",
    "PluginManager",
]
