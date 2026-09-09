"""``uaqe`` — top-level namespace package for the Universal AI
Quantization Engine.

Per ``09_Architecture_Lock.md`` §2, this package's direct sub-packages
are locked as ``uaqe.common``, ``uaqe.domain.*``, ``uaqe.application``,
``uaqe.infrastructure.*``, and ``uaqe.interface.*``. This top-level
``__init__.py`` intentionally re-exports nothing: per §8 of that
document, layering is enforced by having each sub-package import only
from the layers beneath it, and a re-exporting root ``__init__`` would
make it easy to accidentally short-circuit that (e.g. importing
``uaqe.infrastructure`` from within ``uaqe.domain`` via the root
namespace). Import from the specific sub-package you need instead, e.g.
``from uaqe.common import IMR``.
"""
