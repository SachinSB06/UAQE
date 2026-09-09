"""``uaqe.infrastructure.filesystem`` — path-contained file I/O for the
Universal AI Quantization Engine.

Per ``09_Architecture_Lock.md`` §8 (module ownership table), this
area's locked class inventory is exactly: ``FilesystemRepository``
(``03_API_Specification.md`` §17.3, ``11_Implementation_Rules.md``
§5.3). No name here may be renamed, and no additional name may be
added to that locked inventory, without an RFC per §14 of that
document.

``PathManager`` (:mod:`uaqe.infrastructure.filesystem.path_manager`)
and ``FileManager`` (:mod:`uaqe.infrastructure.filesystem.file_manager`)
are internal collaborators introduced to keep path-containment
validation and raw byte/text I/O each in their own single-
responsibility class (``11_Implementation_Rules.md`` §2, §19.1),
mirroring how ``LogFormatter`` supports ``StructuredLogger`` in
``uaqe.infrastructure.logging``. They are not themselves part of the
locked inventory above and are not constructed or used outside this
package.
"""

from uaqe.infrastructure.filesystem.file_manager import FileManager
from uaqe.infrastructure.filesystem.path_manager import PathManager

__all__ = [
    # path_manager.py — internal collaborator, not part of the locked
    # inventory; see module docstring
    "PathManager",
    # file_manager.py — internal collaborator, not part of the locked
    # inventory; see module docstring
    "FileManager",
    # filesystem_repository.py — the locked ``FilesystemRepository``
    # class; intentionally not imported/re-exported here yet, pending
    # the same reconciliation pass noted in
    # ``infrastructure/repositories/__init__.py`` (this package's path
    # differs from the ``infrastructure.repositories`` path named in
    # ``02_Folder_Structure.md`` §5)
]
