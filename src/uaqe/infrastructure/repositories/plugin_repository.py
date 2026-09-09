"""``PluginRepository`` — a read-only, repository-style facade over
``PluginRegistry``.

.. important::
    **This class is not part of the locked architecture.**
    ``09_Architecture_Lock.md`` §8's module ownership table for
    ``uaqe.infrastructure.repositories`` names exactly
    ``ConfigRepository``, ``HardwareProfileRepository``,
    ``FilesystemRepository`` — no ``PluginRepository``. Nothing in
    ``02_Folder_Structure.md``, ``03_API_Specification.md``, or
    ``11_Implementation_Rules.md`` defines this class, and §14 of the
    lock document requires an RFC before a new name is added to a
    locked package's inventory. This file was generated on explicit
    request despite that gap; it should be treated as a proposal, and
    formalized via the RFC process (or removed) before being wired
    into ``CompositionRoot``.

Rationale for the shape below: ``PluginRegistry``
(``uaqe.infrastructure.plugins.plugin_registry``) is deliberately the
one global, mutable, discovery-and-registration surface in the system
(``01_Project_Architecture.md``, "Dependency Injection";
``11_Implementation_Rules.md`` §6.5) — every ``register_*`` and
``discover`` call mutates shared, process-global state. Consumers that
only need to *look up* an already-registered plugin (e.g. a reporting
stage resolving an ``IReportRenderer`` by name, or a CLI command
listing installed strategies) have no need for, and should not be
handed, mutation access. ``PluginRepository`` wraps an already-
constructed ``PluginRegistry`` and exposes only read paths, mirroring
the read/write split every other ``*Repository`` in this package
already has against its own backing store (``ConfigRepository``
against ``config/``, ``HardwareProfileRepository`` against
``hardware_profiles/``). It registers nothing itself and never calls
``discover``; population of the wrapped ``PluginRegistry`` remains
``CompositionRoot``'s responsibility, unchanged.
"""

from __future__ import annotations

from typing import List

from uaqe.common.exceptions import PluginLoadError
from uaqe.common.interfaces.i_compression_strategy import ICompressionStrategy
from uaqe.common.interfaces.i_exporter_backend import IExporterBackend
from uaqe.common.interfaces.i_quantization_strategy import IQuantizationStrategy
from uaqe.common.interfaces.i_report_renderer import IReportRenderer
from uaqe.infrastructure.plugins.plugin_registry import PluginRegistry


class PluginRepository:
    """Read-only lookup surface over an already-populated ``PluginRegistry``.

    Attributes:
        registry: The ``PluginRegistry`` this repository reads from.
            Owned and populated by ``CompositionRoot``; this class
            never mutates it.
    """

    def __init__(self, registry: PluginRegistry) -> None:
        """Initialize a ``PluginRepository`` wrapping ``registry``.

        Args:
            registry: The already-constructed, already-populated
                ``PluginRegistry`` to read from. This repository does
                not call :meth:`PluginRegistry.discover` itself —
                population happens once at ``CompositionRoot`` startup,
                before this repository is constructed.
        """
        self.registry: PluginRegistry = registry

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

    def list_quantization_strategies(self) -> List[str]:
        """Return the registration names of every ``IQuantizationStrategy``."""
        return sorted(self.registry.quantization_strategies.keys())

    def get_compression_strategy(self, name: str) -> ICompressionStrategy:
        """Resolve a registered ``ICompressionStrategy`` by name.

        Args:
            name: The strategy's registration name.

        Returns:
            The registered ``ICompressionStrategy``.

        Raises:
            PluginLoadError: If no strategy is registered under ``name``.
        """
        try:
            return self.registry.compression_strategies[name]
        except KeyError as exc:
            raise PluginLoadError(
                f"No compression strategy registered under name {name!r}.",
                code="PLUGIN_NOT_FOUND",
                remediation_hint=(
                    "Verify the strategy is registered or discoverable under "
                    "plugins/compression_strategies/."
                ),
            ) from exc

    def list_compression_strategies(self) -> List[str]:
        """Return the registration names of every ``ICompressionStrategy``."""
        return sorted(self.registry.compression_strategies.keys())

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
        try:
            return self.registry.exporter_backends[target]
        except KeyError as exc:
            raise PluginLoadError(
                f"No exporter backend registered for target {target!r}.",
                code="PLUGIN_NOT_FOUND",
                remediation_hint=(
                    "Verify a backend declaring this target is registered or "
                    "discoverable under plugins/exporter_backends/."
                ),
            ) from exc

    def list_exporter_targets(self) -> List[str]:
        """Return every target profile id with a registered ``IExporterBackend``."""
        return sorted(self.registry.exporter_backends.keys())

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
        try:
            return self.registry.report_renderers[report_type]
        except KeyError as exc:
            raise PluginLoadError(
                f"No report renderer registered under report_type {report_type!r}.",
                code="PLUGIN_NOT_FOUND",
                remediation_hint=(
                    "Verify the renderer is registered — report_renderers/ has "
                    "no directory-based discovery, see PluginRegistry's module "
                    "docstring, so it must be registered explicitly at "
                    "CompositionRoot startup."
                ),
            ) from exc

    def list_report_renderers(self) -> List[str]:
        """Return the registration keys of every ``IReportRenderer``."""
        return sorted(self.registry.report_renderers.keys())
