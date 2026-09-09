"""``uaqe.infrastructure.repositories`` — the read-only data-access
adapters of the Universal AI Quantization Engine.

Per ``09_Architecture_Lock.md`` §8 (module ownership table), this
package's locked class inventory is exactly: ``ConfigRepository``,
``HardwareProfileRepository``, ``FilesystemRepository``. No name here
may be renamed, and no additional name may be added, without an RFC
per §14 of that document.

``ConfigRepository`` implements ``uaqe.common.interfaces.IConfigRepository``
(``06_Config_Spec.md``, ``11_Implementation_Rules.md`` §5.1). The other
two repositories are out of scope for this package version — their
source modules have not yet been provided — and are intentionally
omitted from the imports and ``__all__`` below rather than imported
from modules that do not yet exist on disk. Add them here once their
files exist, without reordering or renaming the entry already present.
"""

from uaqe.infrastructure.configuration.config_repository import ConfigRepository

__all__ = [
    # config_repository.py
    "ConfigRepository",
    # hardware_profile_repository.py — not yet provided; omitted, see
    # module docstring
    # filesystem_repository.py — not yet provided; omitted, see module
    # docstring (note: a FilesystemRepository already exists at
    # infrastructure/filesystem/filesystem_repository.py; reconciling
    # its location against this locked package is outside the scope of
    # the Configuration module and is left for a dedicated pass)
]
