"""The single global registry of quantization/compression/exporter/
report-renderer implementations, both built-in and plugin-supplied.

Per ``01_Project_Architecture.md`` §"Dependency Injection", ``PluginRegistry``
is the one intentional exception to constructor-injection-only wiring:
it is a deliberately global registry, populated once at
``CompositionRoot`` startup and consulted by every factory
(``IFrameworkAdapterFactory``, ``IExporterFactory``,
``ICompressionFactory``, ``IQuantizationFactory``, ``IPluginFactory``)
before falling back to their built-in registries
(``11_Implementation_Rules.md`` §6.5).

Locked contract: ``03_API_Specification.md`` §20.1.

Note on ``report_renderers``: ``02_Folder_Structure.md`` §17 locks the
``plugins/`` drop-in directory as exactly
``{quantization_strategies, compression_strategies, exporter_backends}``
— no ``report_renderers/`` subfolder exists in the locked tree, even
though ``register_report_renderer`` and ``report_renderers`` are part
of this class's locked contract. Consequently, :meth:`discover` only
scans the three locked subfolders; ``IReportRenderer`` implementations
can only be added via a direct :meth:`register_report_renderer` call
(e.g. ``CompositionRoot`` wiring built-in renderers explicitly), never
through directory-based discovery, until a ``report_renderers/``
subfolder is added to the locked folder structure.

Note on ``PluginValidationError``: ``11_Implementation_Rules.md`` §8.5
and §9 reference a ``PluginValidationError`` exception that is not
part of the locked twelve-member ``UAQEError`` hierarchy
(``09_Architecture_Lock.md`` §3); only ``PluginLoadError`` exists in
``uaqe.common.exceptions``. Every failure mode described for
``PluginValidationError`` in that document (metadata malformed,
interface-version incompatible, contract-test failure, key collision)
is therefore raised here as ``PluginLoadError`` instead, since no new
exception class may be invented.

Note on plugin metadata: ``11_Implementation_Rules.md`` §8.4 describes
a ``PluginMetadata`` dataclass but it is not part of any locked module
this class may import (it is absent from ``uaqe.common`` and from the
locked ``uaqe.infrastructure.plugins`` class inventory in
``09_Architecture_Lock.md`` §3). Each discovered module's
``PLUGIN_METADATA`` object is therefore read structurally (via
``getattr``, not by importing a formal type) for the fields that
section documents: ``plugin_id``, ``category``, ``interface_version``,
and ``allow_override``.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Type

from uaqe.common.exceptions import PluginLoadError
from uaqe.common.interfaces.i_compression_strategy import ICompressionStrategy
from uaqe.common.interfaces.i_exporter_backend import IExporterBackend
from uaqe.common.interfaces.i_quantization_strategy import IQuantizationStrategy
from uaqe.common.interfaces.i_report_renderer import IReportRenderer

_SUPPORTED_INTERFACE_VERSIONS = frozenset({"1.0"})

_QUANTIZATION_CATEGORY = "QUANTIZATION"
_COMPRESSION_CATEGORY = "COMPRESSION"
_EXPORTER_CATEGORY = "EXPORTER"

# Locked subfolders of plugins/ (02_Folder_Structure.md §17) mapped to the
# plugin category and interface each one is discovered and validated against.
_DISCOVERABLE_CATEGORIES: Dict[str, tuple[str, Type[object]]] = {
    "quantization_strategies": (_QUANTIZATION_CATEGORY, IQuantizationStrategy),
    "compression_strategies": (_COMPRESSION_CATEGORY, ICompressionStrategy),
    "exporter_backends": (_EXPORTER_CATEGORY, IExporterBackend),
}


class PluginRegistry:
    """Global registry of ``I*`` implementations, built-in and plugin-supplied.

    Attributes:
        quantization_strategies: Registered ``IQuantizationStrategy``
            instances, keyed by ``name()``.
        compression_strategies: Registered ``ICompressionStrategy``
            instances, keyed by ``name()``.
        exporter_backends: Registered ``IExporterBackend`` instances,
            keyed by each ``profile_id`` string returned in
            ``supported_targets()``.
        report_renderers: Registered ``IReportRenderer`` instances,
            keyed by ``report_type()``.
    """

    def __init__(self) -> None:
        """Initialize an empty ``PluginRegistry``.

        Note: the locked contract takes no constructor arguments, so
        (unlike every other infrastructure class) this class cannot
        receive an ``ILogger`` via constructor injection; registration
        and discovery failures are surfaced exclusively via raised
        ``PluginLoadError`` instances rather than log calls.
        """
        self.quantization_strategies: Dict[str, IQuantizationStrategy] = {}
        self.compression_strategies: Dict[str, ICompressionStrategy] = {}
        self.exporter_backends: Dict[str, IExporterBackend] = {}
        self.report_renderers: Dict[str, IReportRenderer] = {}

    def register_quantization_strategy(self, strategy: IQuantizationStrategy) -> None:
        """Register ``strategy`` under its ``name()``.

        Args:
            strategy: The ``IQuantizationStrategy`` instance to register.

        Raises:
            PluginLoadError: If ``strategy.name()`` is already registered.
        """
        self._register(
            self.quantization_strategies, strategy.name(), strategy, allow_override=False
        )

    def register_compression_strategy(self, strategy: ICompressionStrategy) -> None:
        """Register ``strategy`` under its ``name()``.

        Args:
            strategy: The ``ICompressionStrategy`` instance to register.

        Raises:
            PluginLoadError: If ``strategy.name()`` is already registered.
        """
        self._register(self.compression_strategies, strategy.name(), strategy, allow_override=False)

    def register_exporter_backend(self, backend: IExporterBackend) -> None:
        """Register ``backend`` under every profile id in ``supported_targets()``.

        Args:
            backend: The ``IExporterBackend`` instance to register.

        Raises:
            PluginLoadError: If any of ``backend.supported_targets()``
                is already registered.
        """
        for target in backend.supported_targets():
            self._register(self.exporter_backends, target, backend, allow_override=False)

    def register_report_renderer(self, renderer: IReportRenderer) -> None:
        """Register ``renderer`` under its ``report_type()``.

        Args:
            renderer: The ``IReportRenderer`` instance to register.

        Raises:
            PluginLoadError: If ``renderer.report_type()`` is already
                registered.
        """
        self._register(
            self.report_renderers, renderer.report_type(), renderer, allow_override=False
        )

    def discover(self, plugin_dir: str) -> None:
        """Scan ``plugin_dir``'s locked subfolders and register every valid plugin.

        For each of ``quantization_strategies/``, ``compression_strategies/``,
        and ``exporter_backends/`` under ``plugin_dir``
        (``02_Folder_Structure.md`` §17), every ``*.py`` module is
        imported, its ``PLUGIN_METADATA`` and single concrete
        interface-implementing class are validated
        (``11_Implementation_Rules.md`` §8.2-§8.5), and the resulting
        instance is registered. Discovery never executes side-effecting
        setup beyond import; only the plugin's own class is instantiated.

        Args:
            plugin_dir: The root ``plugins/`` directory to scan.

        Raises:
            PluginLoadError: If a module fails to import, is missing
                or has malformed ``PLUGIN_METADATA``, declares an
                unsupported ``interface_version``, does not expose
                exactly one class implementing the expected interface,
                or collides with an already-registered key without
                ``allow_override=True``.
        """
        plugin_root = Path(plugin_dir)
        for subfolder_name, (category, interface_cls) in _DISCOVERABLE_CATEGORIES.items():
            subfolder = plugin_root / subfolder_name
            if not subfolder.is_dir():
                continue
            for module_path in sorted(subfolder.glob("*.py")):
                if module_path.name == "__init__.py":
                    continue
                self._discover_module(module_path, category, interface_cls)

    def get_quantization_strategy(self, name: str) -> IQuantizationStrategy:
        """Resolve a registered ``IQuantizationStrategy`` by name.

        Args:
            name: The strategy's registration name.

        Returns:
            The registered ``IQuantizationStrategy``.

        Raises:
            PluginLoadError: If no strategy is registered under ``name``.
        """
        try:
            return self.quantization_strategies[name]
        except KeyError as exc:
            raise PluginLoadError(
                f"No quantization strategy registered under name {name!r}.",
                code="PLUGIN_NOT_FOUND",
                remediation_hint=(
                    "Verify the strategy is registered or discoverable under "
                    "plugins/quantization_strategies/."
                ),
            ) from exc

    def _discover_module(
        self, module_path: Path, category: str, interface_cls: Type[object]
    ) -> None:
        """Import one plugin module, validate it, and register its plugin class.

        Args:
            module_path: The ``.py`` file to import.
            category: The expected ``PLUGIN_METADATA.category`` value
                for this subfolder.
            interface_cls: The ``I*`` interface the module's plugin
                class must implement.

        Raises:
            PluginLoadError: On any import, metadata, version, class-
                resolution, or registration-collision failure.
        """
        module = self._import_module(module_path)
        metadata = getattr(module, "PLUGIN_METADATA", None)
        if metadata is None:
            raise PluginLoadError(
                f"Plugin module {module_path} has no module-level PLUGIN_METADATA.",
                code="PLUGIN_METADATA_MISSING",
                remediation_hint="Add a PLUGIN_METADATA object to the plugin module.",
            )

        metadata_category = str(getattr(metadata, "category", "")).upper()
        if metadata_category != category:
            raise PluginLoadError(
                f"Plugin module {module_path} declares category "
                f"{metadata_category!r}; expected {category!r}.",
                code="PLUGIN_CATEGORY_MISMATCH",
                remediation_hint="Place this plugin under the correct plugins/ subfolder.",
            )

        interface_version = str(getattr(metadata, "interface_version", ""))
        if interface_version not in _SUPPORTED_INTERFACE_VERSIONS:
            raise PluginLoadError(
                f"Plugin module {module_path} declares unsupported "
                f"interface_version {interface_version!r}.",
                code="PLUGIN_INTERFACE_VERSION_UNSUPPORTED",
                remediation_hint=(
                    f"Supported interface_version values: {sorted(_SUPPORTED_INTERFACE_VERSIONS)}."
                ),
            )

        plugin_cls = self._resolve_plugin_class(module, module_path, interface_cls)
        try:
            instance = plugin_cls()
        except Exception as exc:  # noqa: BLE001 - third-party plugin code is untrusted
            raise PluginLoadError(
                f"Plugin class {plugin_cls.__name__} in {module_path} could not "
                "be instantiated.",
                code="PLUGIN_INSTANTIATION_FAILED",
                remediation_hint="Ensure the plugin class has a no-argument constructor.",
            ) from exc

        allow_override = bool(getattr(metadata, "allow_override", False))
        if category == _QUANTIZATION_CATEGORY:
            assert isinstance(instance, IQuantizationStrategy)
            self._register(self.quantization_strategies, instance.name(), instance, allow_override)
        elif category == _COMPRESSION_CATEGORY:
            assert isinstance(instance, ICompressionStrategy)
            self._register(self.compression_strategies, instance.name(), instance, allow_override)
        else:
            assert isinstance(instance, IExporterBackend)
            for target in instance.supported_targets():
                self._register(self.exporter_backends, target, instance, allow_override)

    def _import_module(self, module_path: Path) -> ModuleType:
        """Import a single plugin module from an arbitrary filesystem path.

        Args:
            module_path: The ``.py`` file to import.

        Returns:
            The imported module.

        Raises:
            PluginLoadError: If the module cannot be imported.
        """
        module_name = f"uaqe_plugin_{module_path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"No module spec could be created for {module_path}.")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - third-party plugin code is untrusted
            raise PluginLoadError(
                f"Plugin module {module_path} failed to import.",
                code="PLUGIN_IMPORT_FAILED",
                remediation_hint="Verify the plugin module has no import-time errors.",
            ) from exc
        return module

    def _resolve_plugin_class(
        self, module: ModuleType, module_path: Path, interface_cls: Type[object]
    ) -> Type[object]:
        """Find the single class in ``module`` implementing ``interface_cls``.

        Args:
            module: The imported plugin module.
            module_path: The module's source path, for error messages.
            interface_cls: The ``I*`` interface the plugin class must
                implement.

        Returns:
            The resolved plugin class.

        Raises:
            PluginLoadError: If zero or more than one qualifying class
                is found.
        """
        candidates: List[Type[object]] = [
            member
            for _, member in inspect.getmembers(module, inspect.isclass)
            if issubclass(member, interface_cls)
            and member is not interface_cls
            and member.__module__ == module.__name__
        ]
        if len(candidates) != 1:
            raise PluginLoadError(
                f"Plugin module {module_path} must expose exactly one class "
                f"implementing {interface_cls.__name__}; found {len(candidates)}.",
                code="PLUGIN_CLASS_AMBIGUOUS",
                remediation_hint=(
                    f"Define exactly one {interface_cls.__name__} subclass per plugin module."
                ),
            )
        return candidates[0]

    def _register(
        self,
        registry: Dict[str, object],
        key: str,
        value: object,
        allow_override: bool,
    ) -> None:
        """Insert ``value`` under ``key`` in ``registry``, enforcing collision rules.

        Args:
            registry: The target registration dict.
            key: The registration key.
            value: The instance being registered.
            allow_override: Whether ``value`` may replace an existing
                registration under ``key``.

        Raises:
            PluginLoadError: If ``key`` is already registered and
                ``allow_override`` is ``False``.
        """
        if key in registry and not allow_override:
            raise PluginLoadError(
                f"Registration key {key!r} is already registered and this "
                "plugin does not set allow_override=True.",
                code="PLUGIN_KEY_COLLISION",
                remediation_hint=(
                    "Choose a unique registration name, or set "
                    "PLUGIN_METADATA.allow_override=True to intentionally replace it."
                ),
            )
        registry[key] = value
