"""Loads and caches ``HardwareProfile`` instances from the on-disk
``hardware_profiles/**/*.json`` database.

Implements ``IHardwareProfileRepository`` (``03_API_Specification.md``
§1.12) so it is a drop-in substitute for
``uaqe.infrastructure.repositories.HardwareProfileRepository`` anywhere
that interface is required; see the module-scope note in
:mod:`uaqe.hardware.hardware_manager` on how this package's placement
relates to the locked ``09_Architecture_Lock.md`` §8 module ownership
table.

Lazy loading: no profile file is read until the first call to
:meth:`HardwareProfileLoader.get` or
:meth:`HardwareProfileLoader.list_all` — startup time never scales with
the size of the hardware profile database. On that first call, every
``hardware_profiles/**/*.json`` file is parsed and cached together in
one pass, since :meth:`list_all` requires the complete set regardless.
Once populated, the cache is never invalidated for the life of the
process, consistent with ``hardware_profiles/`` being read-only at
runtime — unless :meth:`reload` is explicitly called.
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
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.types import ExportFormat, HardwareClass, Precision

from uaqe.domain.hardware_manager import (
    HardwareProfile,
    FpgaResourceProfile,
)
#: Major ``schema_version`` components this build accepts for hardware
#: profile files (``05_Hardware_Profile_Spec.md`` §7.3).
SUPPORTED_SCHEMA_MAJOR_VERSIONS: FrozenSet[str] = frozenset({"1"})


class HardwareProfileLoader(IHardwareProfileRepository):
    """Loads and caches every ``HardwareProfile`` from ``hardware_profiles/``.

    Attributes:
        profiles_path: The directory containing the ``fpga/``,
            ``embedded/``, and ``raspberrypi/`` subfolders of
            ``*.json`` hardware profile files.
        logger: Optional structured logging sink; if omitted, this
            loader operates silently.
    """

    def __init__(self, profiles_path: str, logger: ILogger | None = None) -> None:
        """Initialize a ``HardwareProfileLoader`` rooted at ``profiles_path``.

        Args:
            profiles_path: The directory containing every
                ``hardware_profiles/**/*.json`` file this loader reads.
                No file under it is read until the first call to
                :meth:`get` or :meth:`list_all`.
            logger: Optional structured logging sink.
        """
        self.profiles_path: str = profiles_path
        self.logger: ILogger | None = logger
        self._cache: Dict[str, HardwareProfile] = {}
        self._load_lock: threading.Lock = threading.Lock()
        self._loaded: bool = False

    def get(self, profile_id: str) -> HardwareProfile:
        """Resolve the ``HardwareProfile`` identified by ``profile_id``.

        Args:
            profile_id: The unique identifier of the hardware profile.

        Returns:
            The resolved ``HardwareProfile``.

        Raises:
            ConfigurationError: If ``profile_id`` cannot be resolved.
        """
        self._ensure_loaded()
        try:
            return self._cache[profile_id]
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
        """Return every ``HardwareProfile`` known to this loader.

        Returns:
            Every parsed ``HardwareProfile``, in no particular order.

        Raises:
            ConfigurationError: If any hardware profile file under
                ``profiles_path`` is missing a field, malformed, or
                declares an unrecognized ``schema_version`` major
                component.
        """
        self._ensure_loaded()
        return list(self._cache.values())

    def list_by_class(self, hardware_class: HardwareClass) -> List[HardwareProfile]:
        """Return every ``HardwareProfile`` of a given ``hardware_class``.

        Args:
            hardware_class: The class to filter by.

        Returns:
            Every parsed ``HardwareProfile`` whose ``hardware_class``
            equals ``hardware_class``, in no particular order.
        """
        return [p for p in self.list_all() if p.hardware_class == hardware_class]

    def reload(self) -> None:
        """Discard the cache and force the next :meth:`get` /
        :meth:`list_all` call to re-read every profile file.

        This is not part of the locked ``IHardwareProfileRepository``
        contract; it exists for callers (tests, long-running services)
        that need to pick up an edited profile file without a process
        restart.
        """
        with self._load_lock:
            self._cache = {}
            self._loaded = False

    def _ensure_loaded(self) -> None:
        """Populate the cache from every ``*.json`` file, exactly once.

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
                self._cache[profile.profile_id] = profile
                if self.logger is not None:
                    self.logger.debug(
                        "Loaded hardware profile.",
                        profile_id=profile.profile_id,
                        source_file=str(path),
                    )
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
        """Reject a hardware profile mapping with a missing or
        unrecognized ``schema_version``.

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
