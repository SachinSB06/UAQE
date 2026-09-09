"""``uaqe.infrastructure.logging`` — the structured logging sink of the
Universal AI Quantization Engine.

Per ``09_Architecture_Lock.md`` §8 (module ownership table), this
package's locked class inventory is exactly: ``StructuredLogger``. No
name here may be renamed, and no additional name may be added, without
an RFC per §14 of that document.

``StructuredLogger`` implements ``uaqe.common.interfaces.ILogger``
(``03_API_Specification.md`` §18.1). ``LogFormatter``
(:mod:`uaqe.infrastructure.logging.log_formatter`) is an internal
collaborator of ``StructuredLogger`` only and is intentionally omitted
from the imports and ``__all__`` below — see its module docstring.
"""

from uaqe.infrastructure.logging.structured_logger import StructuredLogger

__all__ = [
    # structured_logger.py
    "StructuredLogger",
    # log_formatter.py — internal collaborator, not part of the locked
    # inventory; see its module docstring
]
