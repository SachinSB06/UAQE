"""``PluginManager`` — a single facade over plugin loading and lookup,
composing an owned ``PluginRegistry`` with ``PluginLoader``.

.. important::
    **This class is not part of the locked architecture.**
    ``09_Architecture_Lock.md`` §8's module ownership table for
    ``uaqe.infrastructure.plugins`` names exactly ``PluginRegistry``.
    This file was generated on explicit request despite that gap; it
    has not been reconciled via the RFC process in §14, and — like
    ``PluginLoader`` — should be formalized or removed before being
    wired into a real ``CompositionRoot``.

``PluginManager`` does not duplicate ``PluginRegistry``'s discovery,
validation, or storage logic, nor ``PluginLoader``'s directory-scan
orchestration: it owns one of each and delegates every call. It exists
only to give a caller that wants "load plugins, then look them up"
(e.g. a CLI subcommand listing installed plugins, or a small script
outside the full dependency graph) a single object to construct,
rather than wiring a ``PluginRegistry`` and a ``PluginLoader``
separately. ``CompositionRoot`` itself already builds and wires a
``PluginRegistry`` directly per ``11_Implementation_Rules.md`` §6.5;
this class is not a replacement for that wiring.
"""

from __future__ import annotations

from typing import List, Optional

from uaqe.common.interfaces.i_compression_strategy import ICompressionStrategy
from uaqe.common.interfaces.i_exporter_backend import IExporterBackend
from uaqe.common.interfaces.i_quantization_strategy import IQuantizationStrategy
from uaqe.common.interfaces.i_report_renderer import IReportRenderer
from uaqe.infrastructure.plugins.plugin_loader import PluginLoader
from uaqe.infrastructure.plugins.plugin_registry import PluginRegistry


class PluginManager:
    """Owns a ``PluginRegistry`` and a ``PluginLoader``; exposes load + lookup.

    Attributes:
        registry: The owned ``PluginRegistry`` every plugin is loaded
            into and looked up from.
        loader: The owned ``PluginLoader`` used by :meth:`load`.
    """

    def __init__(self, registry: Optional[PluginRegistry] = None) -> None:
        """Initialize a ``PluginManager``.

        Args:
            registry: An existing ``PluginRegistry`` to manage. If
                omitted, a new, empty ``PluginRegistry`` is constructed
                — matching that class's own locked, no-argument
                constructor (``03_API_Specification.md`` §20.1).
        """
        self.registry: PluginRegistry = registry if registry is not None else PluginRegistry()
        self.loader: PluginLoader = PluginLoader(self.registry)

    def load(self, plugin_dir: str) -> None:
        """Discover and register every plugin under ``plugin_dir``.

        Args:
            plugin_dir: The root ``plugins/`` directory to scan.

        Raises:
            PluginLoadError: See ``PluginLoader.load_from_directory``.
        """
        self.loader.load_from_directory(plugin_dir)

    def load_all(self, plugin_dirs: List[str]) -> None:
        """Discover and register every plugin under each directory in ``plugin_dirs``.

        Args:
            plugin_dirs: The root ``plugins/`` directories to scan, in
                load order.

        Raises:
            PluginLoadError: See ``PluginLoader.load_from_directories``.
        """
        self.loader.load_from_directories(plugin_dirs)

    def get_quantization_strategy(self, name: str) -> IQuantizationStrategy:
        """Resolve a registered ``IQuantizationStrategy`` by name.

        Args:
            name: The strategy's registration name.

        Returns:
            The registered ``IQuantizationStrategy``.

        Raises:
            PluginLoadError: If no strategy is registered under ``name``.
        """
        return self.registry.get_quantization_strategy(name)

    def get_compression_strategy(self, name: str) -> ICompressionStrategy:
        """Resolve a registered ``ICompressionStrategy`` by name.

        Args:
            name: The strategy's registration name.

        Returns:
            The registered ``ICompressionStrategy``.

        Raises:
            PluginLoadError: If no strategy is registered under ``name``.
        """
        return self.registry.compression_strategies[name]

    def get_exporter_backend(self, target: str) -> IExporterBackend:
        """Resolve a registered ``IExporterBackend`` by target profile id.

        Args:
            target: A ``HardwareProfile.profile_id`` the backend
                declared via ``supported_targets()``.

        Returns:
            The registered ``IExporterBackend``.

        Raises:
            PluginLoadError: If no backend is registered under ``target``.
        """
        return self.registry.exporter_backends[target]

    def get_report_renderer(self, report_type: str) -> IReportRenderer:
        """Resolve a registered ``IReportRenderer`` by report type.

        Args:
            report_type: The renderer's registration key, i.e. the
                value its ``report_type()`` returns.

        Returns:
            The registered ``IReportRenderer``.

        Raises:
            PluginLoadError: If no renderer is registered under
                ``report_type``.
        """
        return self.registry.report_renderers[report_type]

    def list_quantization_strategies(self) -> List[str]:
        """Return the registration names of every ``IQuantizationStrategy``."""
        return sorted(self.registry.quantization_strategies.keys())

    def list_compression_strategies(self) -> List[str]:
        """Return the registration names of every ``ICompressionStrategy``."""
        return sorted(self.registry.compression_strategies.keys())

    def list_exporter_targets(self) -> List[str]:
        """Return every target profile id with a registered ``IExporterBackend``."""
        return sorted(self.registry.exporter_backends.keys())

    def list_report_renderers(self) -> List[str]:
        """Return the registration keys of every ``IReportRenderer``."""
        return sorted(self.registry.report_renderers.keys())
