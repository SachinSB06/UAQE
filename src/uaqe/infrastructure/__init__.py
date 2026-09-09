"""``uaqe.infrastructure`` — concrete integrations (file I/O, ML framework
parsing, hardware-native export, config loading, logging, metrics,
plugins) for the Universal AI Quantization Engine.

Per ``02_Folder_Structure.md`` §5, this package's direct sub-packages
include ``uaqe.infrastructure.repositories`` (``ConfigRepository``,
``HardwareProfileRepository``, ``FilesystemRepository``),
``framework_adapters``, ``exporter_backends``, ``logging``, ``metrics``,
and ``plugins``.

Per ``09_Architecture_Lock.md`` §8, ``uaqe.infrastructure`` may depend on
``uaqe.common`` and third-party libraries, but never on
``uaqe.interface``, and never reaches sideways into ``uaqe.domain``
business logic. This top-level ``__init__.py`` intentionally re-exports
nothing, for the same reason given in ``uaqe/__init__.py``: a
re-exporting root would make it easy to accidentally short-circuit
layering by importing an unrelated adapter family through this
namespace. Import from the specific sub-package you need instead, e.g.
``from uaqe.infrastructure.repositories import ConfigRepository``.
"""
