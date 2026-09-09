"""``uaqe.infrastructure.repositories`` — the read-only data-access
adapters of the Universal AI Quantization Engine.

Per ``09_Architecture_Lock.md`` §8 (module ownership table), this
package's locked class inventory is exactly: ``ConfigRepository``,
``HardwareProfileRepository``, ``FilesystemRepository``. No name here
may be renamed, and no additional name may be added, without an RFC
per §14 of that document.

``ConfigRepository`` implements ``uaqe.common.interfaces.IConfigRepository``
(``06_Config_Spec.md``, ``11_Implementation_Rules.md`` §5.1).
``HardwareProfileRepository`` implements
``uaqe.common.interfaces.IHardwareProfileRepository``
(``05_Hardware_Profile_Spec.md``, ``11_Implementation_Rules.md`` §5.2).
``FilesystemRepository`` is out of scope for this package version — a
``FilesystemRepository`` already exists at
``infrastructure/filesystem/filesystem_repository.py``; reconciling its
location against this locked package is left for a dedicated pass and
it is intentionally omitted from the imports and ``__all__`` below
rather than imported from a module that does not exist at this
package's path. Add it here once that reconciliation happens, without
reordering or renaming the entries already present.
"""

from uaqe.infrastructure.configuration.config_repository import ConfigRepository
from uaqe.infrastructure.repositories.hardware_profile_repository import (
    HardwareProfileRepository,
)

__all__ = [
    # config_repository.py
    "ConfigRepository",
    # hardware_profile_repository.py
    "HardwareProfileRepository",
    # filesystem_repository.py — not yet provided at this package's
    # path; omitted, see module docstring
]
