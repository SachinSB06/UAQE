"""``uaqe.domain.hardware`` — deployment-target modeling for the
Universal AI Quantization Engine.

Per ``09_Architecture_Lock.md`` §8 (module ownership table), this
package's locked class inventory is exactly: ``HardwareProfile``,
``HardwareManager``.

``HardwareProfile`` (and its nested ``FpgaResourceProfile``) is defined
in :mod:`uaqe.domain.hardware.hardware_manager` and is provided here.
``HardwareManager`` itself is out of scope for this pass — only the
value objects it and ``IHardwareProfileRepository`` depend on are
provided, so that ``HardwareProfileRepository`` has a concrete
``HardwareProfile`` to construct and return. Add ``HardwareManager``
here once its module is provided, without renaming or removing the
entries already present.
"""

from uaqe.domain.hardware.hardware_manager import FpgaResourceProfile, HardwareProfile

__all__ = [
    # hardware_manager.py
    "HardwareProfile",
    "FpgaResourceProfile",
    # HardwareManager — not yet provided; omitted, see module docstring
]
