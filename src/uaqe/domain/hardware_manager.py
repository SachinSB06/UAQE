"""``HardwareProfile`` value object and its nested ``FpgaResourceProfile``.

Per ``05_Hardware_Profile_Spec.md`` §1, ``HardwareProfile`` is the
in-memory representation returned by
``HardwareProfileRepository.get(profile_id)`` / ``list_all()``. Per the
module ownership table in ``09_Architecture_Lock.md`` §8
(``uaqe.domain.hardware``: ``HardwareProfile``, ``HardwareManager``),
it lives here rather than in ``uaqe.common.value_objects`` — see the
"Note on value-object field drift" in ``09_Architecture_Lock.md``
(§ following the module ownership table), which resolves the apparent
conflict between that spec's prose ("belongs in
``uaqe.common.value_objects``") and the locked ownership table: the
ownership table is authoritative for *location*, while
``05_Hardware_Profile_Spec.md`` §1 is authoritative for the *complete
field set* (superseding the abbreviated dataclass in
``03_API_Specification.md`` §5.1).

``uaqe.common.interfaces.i_hardware_profile_repository`` already
depends on this exact module path via a ``TYPE_CHECKING``-only forward
reference, since ``uaqe.common`` has no runtime dependency on
``uaqe.domain`` (``09_Architecture_Lock.md`` §8).

``HardwareManager`` is out of scope for this module — see the package
``__init__.py`` docstring. Only the value objects are provided here,
which is sufficient for ``HardwareProfileRepository``
(``11_Implementation_Rules.md`` §5.2) to construct and return
``HardwareProfile`` instances.

Locked field set: ``05_Hardware_Profile_Spec.md`` §1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from uaqe.common.types import ExportFormat, HardwareClass, Precision


@dataclass(frozen=True)
class FpgaResourceProfile:
    """FPGA fabric resource capacity for a single hardware profile.

    Only populated on a ``HardwareProfile`` whose ``hardware_class`` is
    ``HardwareClass.FPGA``; ``None`` on every other class
    (``05_Hardware_Profile_Spec.md`` §1).

    Attributes:
        logic_cells: Total configurable logic cell count.
        bram_kb: Total block RAM capacity, in kilobytes.
        dsp_slices: Total dedicated DSP slice count.
        lut_count: Total look-up table count.
    """

    logic_cells: int
    bram_kb: int
    dsp_slices: int
    lut_count: int


@dataclass(frozen=True)
class HardwareProfile:
    """A single deployment target's capabilities and constraints.

    Deserialized by ``HardwareProfileRepository`` from one
    ``hardware_profiles/**/*.json`` file per
    ``05_Hardware_Profile_Spec.md`` §2. Consumed by ``HardwareManager``,
    ``LayerCompatibilityChecker``, ``QuantizationAdvisor``,
    ``CompressionAdvisor``, ``OptimizationEngine``, ``MemoryOptimizer``,
    ``Exporter``, ``Benchmarker``, and ``DeploymentReadinessScorer``
    (``05_Hardware_Profile_Spec.md`` §1, "Consumers").

    Fields not applicable to a given ``hardware_class`` are ``None``
    rather than omitted, matching the on-disk schema's "no
    optional-key ambiguity" rule (``05_Hardware_Profile_Spec.md`` §2,
    ``07_Coding_Standards.md``).

    Attributes:
        profile_id: Unique identifier, e.g. ``"artix7"``, ``"esp32"``,
            ``"raspberrypi4"``.
        display_name: Human-readable name, e.g. ``"Xilinx Artix-7"``.
        hardware_class: The broad hardware category of this target.
        ram_bytes: Total RAM, in bytes. ``None`` where not applicable.
        flash_bytes: Total flash storage, in bytes. ``None`` where not
            applicable (e.g. FPGA BRAM, measured separately via
            ``fpga_resources``).
        storage_bytes: SD/eMMC storage, in bytes. Populated only for
            ``HardwareClass.RASPBERRY_PI``; ``None`` otherwise.
        tensor_memory_bytes: Usable working memory for
            activations/arena — the hard ceiling ``MemoryOptimizer``
            plans against, distinct from ``ram_bytes``/``flash_bytes``.
        runtime: The runtime identifier, e.g. ``"tflite-micro"``,
            ``"bare-metal-hdl"``, ``"tflite-runtime"``.
        supported_precisions: The ``Precision`` values this target can
            execute.
        max_model_size_bytes: The maximum total serialized model size
            this target accepts.
        preferred_export_formats: The ``ExportFormat`` values, in
            preference order, this target's exporter backends produce.
        clock_speed_hz: The target's clock speed, in Hz. ``None`` where
            not yet populated (``05_Hardware_Profile_Spec.md`` §4).
        fpga_resources: FPGA fabric resource capacity. ``None`` unless
            ``hardware_class`` is ``HardwareClass.FPGA``.
        constraints: Free-text constraint notes surfaced verbatim in
            ``CompatibilityReport.constraint_violations``
            (``05_Hardware_Profile_Spec.md`` §6 rule 1).
        schema_version: The on-disk schema version this profile was
            parsed under, e.g. ``"1.0"``.
    """

    profile_id: str
    display_name: str
    hardware_class: HardwareClass
    ram_bytes: Optional[int]
    flash_bytes: Optional[int]
    storage_bytes: Optional[int]
    tensor_memory_bytes: int
    runtime: str
    default_runtime: str
    max_model_size_bytes: int
    schema_version: str
    supported_precisions: List[Precision] = field(default_factory=list)
    preferred_export_formats: List[ExportFormat] = field(default_factory=list)
    supported_runtimes: List[str] = field(default_factory=list)
    clock_speed_hz: Optional[int] = None
    fpga_resources: Optional[FpgaResourceProfile] = None
    constraints: List[str] = field(default_factory=list)
