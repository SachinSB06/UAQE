"""Sole first-party implementation of ``IHardwareProfileRepository``.

Loads ``HardwareProfile`` objects from ``hardware_profiles/**/*.json``
(``05_Hardware_Profile_Spec.md`` §§3-5: the ``fpga/``, ``embedded/``,
and ``raspberrypi/`` subfolders), keyed by ``profile_id``
(``11_Implementation_Rules.md`` §5.2).

Locked contract: ``03_API_Specification.md`` §17.2.
Locked location: ``09_Architecture_Lock.md`` §8, module ownership table
(``uaqe.infrastructure.repositories``: ``ConfigRepository``,
``HardwareProfileRepository``, ``FilesystemRepository``).

Lazy loading (``11_Implementation_Rules.md`` §5.5): no profile file is
read until the first call to ``get()`` or ``list_all()`` — startup time
never scales with the size of the hardware profile database. On that
first call, every ``hardware_profiles/**/*.json`` file is parsed and
cached together in one pass rather than one file at a time, since
``list_all()`` requires the complete set regardless and
``05_Hardware_Profile_Spec.md`` does not define a per-profile-id
directory layout that would let ``get()`` locate a single file without
first enumerating the tree. Once populated, the cache is never
invalidated for the life of the process (``11_Implementation_Rules.md``
§5.4), consistent with ``hardware_profiles/`` being read-only at
runtime (§5.6).

Note on ``profile_cache_enabled`` (``06_Config_Spec.md`` §3): §5.4
describes caching as conditional on ``hardware.json``'s
``profile_cache_enabled`` flag, but the locked constructor
(``03_API_Specification.md`` §17.2) takes only ``profiles_path`` — it
is never handed a ``HardwareConfig`` or raw ``hardware.json`` mapping
to read that flag from. Per the same resolution pattern
``ConfigRepository`` applies to its own documentation gaps (see that
class's module docstring), this is treated as a documentation gap, not
license to add an undocumented constructor parameter: this
implementation always caches, which is a safe superset of "cache when
enabled" given profiles are read-only for the process lifetime either
way.

Every returned ``HardwareProfile`` is immutable
(``@dataclass(frozen=True)``, ``11_Implementation_Rules.md`` §5.2), so
unlike ``ConfigRepository``'s ``*Config`` objects, no defensive copy is
needed on read — the object itself cannot be mutated by a caller.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, FrozenSet, List

from uaqe.common.exceptions import ConfigurationError
from uaqe.common.interfaces.i_hardware_profile_repository import (
    IHardwareProfileRepository,
)
from uaqe.common.types import ExportFormat, HardwareClass, Precision
from uaqe.domain.hardware_manager import FpgaResourceProfile, HardwareProfile

#: Major ``schema_version`` components this build of ``uaqe`` accepts
#: for hardware profile files, per ``05_Hardware_Profile_Spec.md`` §7.3.
SUPPORTED_SCHEMA_MAJOR_VERSIONS: FrozenSet[str] = frozenset({"1"})


class HardwareProfileRepository(IHardwareProfileRepository):
    """Loads and caches every ``HardwareProfile`` from ``hardware_profiles/``.

    Attributes:
        profiles_path: The directory containing the ``fpga/``,
            ``embedded/``, and ``raspberrypi/`` subfolders of
            ``*.json`` hardware profile files.
    """

    def __init__(self, profiles_path: str) -> None:
        """Initialize a ``HardwareProfileRepository`` rooted at ``profiles_path``.

        Args:
            profiles_path: The directory containing every
                ``hardware_profiles/**/*.json`` file this repository
                loads. No file under it is read until the first call
                to :meth:`get` or :meth:`list_all`.
        """
        self.profiles_path: str = profiles_path
        self.cache: Dict[str, HardwareProfile] = {}
        self._load_lock: threading.Lock = threading.Lock()
        self._loaded: bool = False

    def get(self, profile_id: str) -> HardwareProfile:
        """Resolve the ``HardwareProfile`` identified by ``profile_id``.

        Args:
            profile_id: The unique identifier of the hardware profile.

        Returns:
            The resolved ``HardwareProfile``.

        Raises:
            ConfigurationError: If ``profile_id`` cannot be resolved,
                or if any hardware profile file under ``profiles_path``
                is missing a field, malformed, or declares an
                unrecognized ``schema_version`` major component.
        """
        self._ensure_loaded()
        try:
            return self.cache[profile_id]
        except KeyError as exc:
            raise ConfigurationError(
                f"No hardware profile registered for profile_id {profile_id!r}.",
                code="HARDWARE_PROFILE_NOT_FOUND",
                remediation_hint=(
                    f"Add a *.json file declaring profile_id={profile_id!r} "
                    f"under {self.profiles_path}, per "
                    "05_Hardware_Profile_Spec.md §§3-5."
                ),
            ) from exc

    def list_all(self) -> List[HardwareProfile]:
        """Return every ``HardwareProfile`` known to this repository.

        Returns:
            Every parsed ``HardwareProfile``, in no particular order.

        Raises:
            ConfigurationError: If any hardware profile file under
                ``profiles_path`` is missing a field, malformed, or
                declares an unrecognized ``schema_version`` major
                component.
        """
        self._ensure_loaded()
        return list(self.cache.values())

    def _ensure_loaded(self) -> None:
        """Populate :attr:`cache` from every ``*.json`` file, exactly once.

        Raises:
            ConfigurationError: If any hardware profile file under
                ``profiles_path`` is missing a field, malformed, or
                declares an unrecognized ``schema_version`` major
                component.
        """
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            for path in sorted(Path(self.profiles_path).rglob("*.json")):
                profile = self._parse_profile(path)
                self.cache[profile.profile_id] = profile
            self._loaded = True

    def _parse_profile(self, path: Path) -> HardwareProfile:
        """Read, validate, and deserialize one hardware profile file.

        Args:
            path: The ``*.json`` file to parse.

        Returns:
            The deserialized ``HardwareProfile``.

        Raises:
            ConfigurationError: If ``path`` is unreadable, is not valid
                JSON, is missing a required field, has an invalid enum
                value, or declares an unrecognized ``schema_version``
                major component.
        """
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(
                f"Hardware profile file {path.name!r} is missing or unreadable.",
                code="HARDWARE_PROFILE_FILE_MISSING",
                remediation_hint=f"Verify {path} exists and is readable.",
            ) from exc

        try:
            data: Dict[str, Any] = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                f"Hardware profile file {path.name!r} is not valid JSON.",
                code="HARDWARE_PROFILE_FILE_MALFORMED",
                remediation_hint=f"Validate {path.name}'s JSON syntax.",
            ) from exc

        self._validate_schema_version(data, path.name)

        try:
            fpga_resources_data = data.get("fpga_resources")
            fpga_resources = (
                FpgaResourceProfile(
                    logic_cells=int(fpga_resources_data["logic_cells"]),
                    bram_kb=int(fpga_resources_data["bram_kb"]),
                    dsp_slices=int(fpga_resources_data["dsp_slices"]),
                    lut_count=int(fpga_resources_data["lut_count"]),
                )
                if fpga_resources_data is not None
                else None
            )
            return HardwareProfile(
                profile_id=str(data["profile_id"]),
                display_name=str(data["display_name"]),
                hardware_class=HardwareClass(data["hardware_class"]),
                ram_bytes=(
                    int(data["ram_bytes"]) if data.get("ram_bytes") is not None else None
                ),
                flash_bytes=(
                    int(data["flash_bytes"])
                    if data.get("flash_bytes") is not None
                    else None
                ),
                storage_bytes=(
                    int(data["storage_bytes"])
                    if data.get("storage_bytes") is not None
                    else None
                ),
                tensor_memory_bytes=int(data["tensor_memory_bytes"]),
                runtime=str(data["runtime"]),
                default_runtime=str(data["default_runtime"]),
                supported_precisions=[
                    Precision(entry) for entry in data["supported_precisions"]
                ],
                max_model_size_bytes=int(data["max_model_size_bytes"]),
                preferred_export_formats=[
                    ExportFormat(entry) for entry in data["preferred_export_formats"]
                ],
                supported_runtimes=list(data.get("supported_runtimes", [])),
                clock_speed_hz=(
                    int(data["clock_speed_hz"])
                    if data.get("clock_speed_hz") is not None
                    else None
                ),
                fpga_resources=fpga_resources,
                constraints=list(data.get("constraints", [])),
                schema_version=str(data["schema_version"]),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ConfigurationError(
                f"Hardware profile file {path.name!r} is missing a required "
                "field or has an invalid value.",
                code="INVALID_HARDWARE_PROFILE",
                remediation_hint=(
                    f"Validate {path.name} against 05_Hardware_Profile_Spec.md §2."
                ),
            ) from exc

    def _validate_schema_version(self, data: Dict[str, Any], filename: str) -> None:
        """Reject a hardware profile mapping with a missing or unrecognized
        ``schema_version``.

        Args:
            data: The parsed hardware profile mapping to validate.
            filename: The source file name, used for error context.

        Raises:
            ConfigurationError: If ``schema_version`` is absent, is not
                of the form ``"<major>.<minor>"``, or its major version
                component is not in
                :data:`SUPPORTED_SCHEMA_MAJOR_VERSIONS`
                (``05_Hardware_Profile_Spec.md`` §7.3).
        """
        schema_version = data.get("schema_version")
        if not schema_version or "." not in str(schema_version):
            raise ConfigurationError(
                f"Hardware profile file {filename!r} is missing a valid "
                "schema_version.",
                code="UNRECOGNIZED_SCHEMA_VERSION",
                remediation_hint=f"Add a schema_version field to {filename}.",
            )
        major_version = str(schema_version).split(".")[0]
        if major_version not in SUPPORTED_SCHEMA_MAJOR_VERSIONS:
            raise ConfigurationError(
                f"Hardware profile file {filename!r} declares unsupported "
                f"schema_version {schema_version!r}.",
                code="UNRECOGNIZED_SCHEMA_VERSION",
                remediation_hint=(
                    "Supported major versions: "
                    f"{sorted(SUPPORTED_SCHEMA_MAJOR_VERSIONS)}."
                ),
            )
