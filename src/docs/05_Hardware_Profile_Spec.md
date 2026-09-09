# 05_Hardware_Profile_Spec.md
## Universal AI Quantization Engine — Hardware Profile Specification

**Depends on:** `01_Project_Architecture.md`, `02_Folder_Structure.md`, `03_API_Specification.md`, `04_Data_Flow.md`
**Status:** DRAFT — pending Architecture Lock
**Scope:** Defines the `HardwareProfile` schema and the JSON database files under `hardware_profiles/` (see `02_Folder_Structure.md` §8), consumed by `IHardwareProfileRepository` / `HardwareProfileRepository`.

---

## 1. `HardwareProfile` Value Object

This is the in-memory representation returned by `HardwareProfileRepository.get(profile_id)`. It belongs in `uaqe.common.value_objects` alongside the config value objects in `03_API_Specification.md` §1.3 (immutable).

```
@immutable class HardwareProfile
  + profile_id: str                    # e.g. "artix7", "esp32", "raspberrypi4"
  + display_name: str                  # e.g. "Xilinx Artix-7"
  + hardware_class: HardwareClass       # FPGA | EMBEDDED | RASPBERRY_PI
  + ram_bytes: int
  + flash_bytes: Optional[int]          # None where not applicable (e.g. FPGA BRAM measured separately)
  + storage_bytes: Optional[int]        # SD/eMMC for Raspberry Pi class; None otherwise
  + tensor_memory_bytes: int            # usable working memory for activations/arena
  + runtime: str                        # e.g. "tflite-micro", "bare-metal-hdl", "tflite-runtime"
  + supported_precisions: List[Precision]
  + max_model_size_bytes: int
  + preferred_export_formats: List[ExportFormat]
  + clock_speed_hz: Optional[int]
  + fpga_resources: Optional[FpgaResourceProfile]   # None unless hardware_class == FPGA
  + constraints: List[str]              # free-text constraint notes surfaced in CompatibilityReport
  + schema_version: str                 # matches uaqe.common.constants.HARDWARE_SCHEMA_VERSION
```

```
@immutable class FpgaResourceProfile
  + logic_cells: int
  + bram_kb: int
  + dsp_slices: int
  + lut_count: int
```

**Consumers:** `HardwareManager`, `LayerCompatibilityChecker`, `QuantizationAdvisor`, `CompressionAdvisor`, `OptimizationEngine`, `MemoryOptimizer`, `Exporter`, `Benchmarker`, `DeploymentReadinessScorer` (all per `04_Data_Flow.md`).

---

## 2. JSON File Schema (on-disk representation)

Every file under `hardware_profiles/**/*.json` follows this exact structure. `HardwareProfileRepository` deserializes each into a `HardwareProfile`.

```json
{
  "schema_version": "1.0",
  "profile_id": "artix7",
  "display_name": "Xilinx Artix-7",
  "hardware_class": "FPGA",
  "ram_bytes": null,
  "flash_bytes": null,
  "storage_bytes": null,
  "tensor_memory_bytes": 2097152,
  "runtime": "bare-metal-hdl",
  "supported_precisions": ["INT8", "INT4"],
  "max_model_size_bytes": 4194304,
  "preferred_export_formats": ["MEM", "HEX", "BIN"],
  "clock_speed_hz": 100000000,
  "fpga_resources": {
    "logic_cells": 101440,
    "bram_kb": 4860,
    "dsp_slices": 240,
    "lut_count": 63400
  },
  "constraints": [
    "No native floating-point units; FP32/FP16 layers must be fully quantized before export.",
    "Recurrent layers (LSTM/GRU) unsupported in v1 HDL backend."
  ]
}
```

Fields not applicable to a given hardware class are explicitly set to `null` (never omitted) so that `HardwareProfileRepository` can deserialize all files with one fixed schema, per Coding Standard "no optional-key ambiguity" (see `07_Coding_Standards.md`).

---

## 3. FPGA Database

**Location:** `hardware_profiles/fpga/*.json`
**Files:** `artix7.json`, `zynq7000.json`, `kintex.json`, `cyclonev.json`

| `profile_id` | `display_name` | `tensor_memory_bytes` | `max_model_size_bytes` | `supported_precisions` | `preferred_export_formats` | `runtime` |
|---|---|---|---|---|---|---|
| `artix7` | Xilinx Artix-7 | 2,097,152 | 4,194,304 | INT8, INT4 | MEM, HEX, BIN | bare-metal-hdl |
| `zynq7000` | Xilinx Zynq-7000 | 8,388,608 | 16,777,216 | INT8, INT4, MIXED | MEM, HEX, BIN | bare-metal-hdl |
| `kintex` | Xilinx Kintex-7 | 16,777,216 | 33,554,432 | INT8, INT4, FP16, MIXED | MEM, HEX, BIN | bare-metal-hdl |
| `cyclonev` | Intel Cyclone V | 4,194,304 | 8,388,608 | INT8, INT4 | MEM, HEX, BIN | bare-metal-hdl |

`fpga_resources` fields (`logic_cells`, `bram_kb`, `dsp_slices`, `lut_count`) are populated per-device from vendor datasheets; exact figures are data-entry detail, not architectural — any FPGA backend in `infrastructure/exporter_backends/fpga/` must read these fields rather than hard-code resource limits (Contract Rule, see §6 below).

---

## 4. Embedded Database

**Location:** `hardware_profiles/embedded/*.json`
**Files:** `esp32.json`, `esp32s3.json`, `stm32f4.json`, `stm32h7.json`, `rp2040.json`, `portenta_h7.json`

| `profile_id` | `display_name` | `ram_bytes` | `flash_bytes` | `tensor_memory_bytes` | `max_model_size_bytes` | `supported_precisions` | `preferred_export_formats` | `runtime` |
|---|---|---|---|---|---|---|---|---|
| `esp32` | ESP32 | 524,288 | 4,194,304 | 262,144 | 1,048,576 | INT8 | TFLITE, BIN, H_HEADER | tflite-micro |
| `esp32s3` | ESP32-S3 | 524,288 | 8,388,608 | 393,216 | 2,097,152 | INT8, INT4 | TFLITE, BIN, H_HEADER | tflite-micro |
| `stm32f4` | STM32F4 | 196,608 | 1,048,576 | 131,072 | 524,288 | INT8 | TFLITE, BIN, H_HEADER | tflite-micro |
| `stm32h7` | STM32H7 | 1,048,576 | 2,097,152 | 524,288 | 1,572,864 | INT8, INT4 | TFLITE, BIN, H_HEADER | tflite-micro |
| `rp2040` | RP2040 | 264,192 | 2,097,152 | 131,072 | 786,432 | INT8 | TFLITE, BIN, H_HEADER | tflite-micro |
| `portenta_h7` | Arduino Portenta H7 | 1,048,576 | 8,388,608 | 786,432 | 3,145,728 | INT8, INT4, FP16 | TFLITE, BIN, H_HEADER | tflite-micro |

`storage_bytes` is `null` for every embedded profile (no removable storage assumed in v1). `clock_speed_hz` and `fpga_resources` are `null` (`clock_speed_hz` may be populated in a later schema revision — see §7).

---

## 5. Raspberry Pi Database

**Location:** `hardware_profiles/raspberrypi/*.json`
**Files:** `raspberrypi4.json`, `raspberrypi5.json`

| `profile_id` | `display_name` | `ram_bytes` | `storage_bytes` | `tensor_memory_bytes` | `max_model_size_bytes` | `supported_precisions` | `preferred_export_formats` | `runtime` |
|---|---|---|---|---|---|---|---|---|
| `raspberrypi4` | Raspberry Pi 4 | 4,294,967,296 | 34,359,738,368 | 2,147,483,648 | 536,870,912 | INT8, FP16, MIXED | TFLITE, ONNX | tflite-runtime |
| `raspberrypi5` | Raspberry Pi 5 | 8,589,934,592 | 68,719,476,736 | 4,294,967,296 | 1,073,741,824 | INT8, FP16, MIXED | TFLITE, ONNX | tflite-runtime |

`ram_bytes` and `storage_bytes` reflect the minimum shipped configuration for each board; `flash_bytes` is `null` (Raspberry Pi boards use `storage_bytes`, not onboard flash, per the field-applicability rule in §2).

---

## 6. Hardware Constraint Enforcement Contract

1. `LayerCompatibilityChecker` reads `HardwareProfile.constraints` as free-text advisories surfaced verbatim in `CompatibilityReport.constraint_violations` when a layer or op-type triggers one; it does not parse constraint strings programmatically — enforcement logic for known constraint categories (unsupported op types, precision mismatches) lives in code, not in the JSON.
2. `QuantizationAdvisor` and `CompressionAdvisor` MUST filter candidate `Precision` / `CompressionType` values against `HardwareProfile.supported_precisions` before producing a recommendation; a config recommending an unsupported precision is an architecture violation, not a runtime one — this MUST be enforced regardless of user `config_overrides`.
3. `Exporter`'s selected `IExporterBackend` MUST validate the final IMR's total serialized size against `HardwareProfile.max_model_size_bytes` before writing any output file; on overflow it raises `ExportError`, never silently truncates.
4. `MemoryOptimizer` MUST treat `HardwareProfile.tensor_memory_bytes` as a hard ceiling for activation-arena planning, distinct from `ram_bytes`/`flash_bytes` (which bound total footprint including code and static buffers).
5. Every FPGA `IExporterBackend` implementation MUST read `HardwareProfile.fpga_resources` and raise `ExportError` rather than proceed if projected resource utilization (logic cells, BRAM, DSP slices) exceeds the profile's reported capacity.

---

## 7. Future Extensibility

1. **New hardware within an existing class:** add one new JSON file to the appropriate `hardware_profiles/<class>/` subfolder; no code changes required as long as an `IExporterBackend` already exists for that class's runtime family, or a new backend is registered via `PluginRegistry`.
2. **New hardware class:** requires (a) a new `HardwareClass` enum value in `uaqe.common.types` (`03_API_Specification.md` §1.2), (b) a new `hardware_profiles/<new_class>/` folder, (c) at minimum one `IExporterBackend` implementation, and (d) an RFC against `09_Architecture_Lock.md` since `HardwareClass` is a locked enum.
3. **New profile fields:** the schema is versioned via `schema_version`. Additive fields (e.g. `clock_speed_hz` population, future `npu_resources` block for boards with dedicated NPUs) increment the minor version and MUST include a default/null-safe value for all existing profile files. `HardwareProfileRepository` MUST reject any profile JSON whose `schema_version` major component it does not recognize, raising `ConfigurationError`.
4. **Per-profile calibration hints:** a future `calibration_hints` object (recommended calibration batch size, recommended dataset size) is anticipated but explicitly out of scope for v1; if added, it must be optional and `null`-defaulted in all existing files to avoid breaking `HardwareProfileRepository.get()`.

---

**End of `05_Hardware_Profile_Spec.md`.**
