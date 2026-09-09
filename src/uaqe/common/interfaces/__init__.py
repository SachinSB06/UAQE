"""``uaqe.interface`` — entry-point-facing composition boundary of the
Universal AI Quantization Engine.

Per ``02_Folder_Structure.md`` §6, this package is where every concrete
``uaqe.infrastructure`` implementation is resolved and wired into the
``uaqe.application`` layer's constructor-injected collaborators. It is
the only package permitted to import both ``uaqe.infrastructure`` and
``uaqe.application`` in the same module (``09_Architecture_Lock.md``
§8 layering: ``interface -> application, infrastructure, common``).

Currently exposes only :class:`~uaqe.interface.composition_root.CompositionRoot`.
"""

from __future__ import annotations
