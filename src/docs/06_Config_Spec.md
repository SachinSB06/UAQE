# 06_Config_Spec.md
## Universal AI Quantization Engine — Configuration Specification

**Depends on:** `01_Project_Architecture.md`, `02_Folder_Structure.md`, `03_API_Specification.md`, `04_Data_Flow.md`, `05_Hardware_Profile_Spec.md`
**Status:** DRAFT — pending Architecture Lock
**Scope:** Defines every file under `config/` (see `02_Folder_Structure.md` §7), the exact fields in each, and their mapping to the value objects in `03_API_Specification.md` §1.3. All files are loaded exclusively by `ConfigRepository` (implements `IConfigRepository`).

---

## 1. `config.json` — Global Application Config

Top-level, framework-agnostic settings not owned by any single subsystem.

```json
{
  "schema_version": "1.0",
  "app_name": "Universal AI Quantization Engine",
  "default_run_id_prefix": "run",
  "workdir_path": "workdir",
  "outputs_path": "outputs",
  "reports_path": "reports",
  "logs_path": "logs",
  "hardware_profiles_path": "hardware_profiles",
  "plugins_path": "plugins",
  "large_model_threshold_mb": 50,
  "on_error": "ABORT"
}
```

| Field | Type | Consumed by | Notes |
|---|---|---|---|
| `schema_version` | str | `ConfigRepository` | Rejected with `ConfigurationError` if major version unrecognized |
| `app_name` | str | `interface/` entry points | Display/logging only |
| `default_run_id_prefix` | str | `SessionManager` | Used when generating `run_id` if caller does not supply one |
| `workdir_path`, `outputs_path`, `reports_path`, `logs_path`, `hardware_profiles_path`, `plugins_path` | str | `FilesystemRepository`, `HardwareProfileRepository`, `PluginRegistry` | Relative to project root; absolute paths also permitted |
| `large_model_threshold_mb` | int | maps to `ExecutionConfig.large_model_threshold_mb` | Threshold above which `PipelineOrchestrator` may adjust threading strategy (see `01_Project_Architecture.md` §"Threading Strategy") |
| `on_error` | str enum: `ABORT`\|`SKIP_STAGE`\|`BEST_EFFORT` | maps to `ExecutionConfig.on_error` (`PipelineErrorPolicy`) | Default error policy; overridable per-run via `RunRequest.config_overrides` |

---

## 2. `settings.yaml` — Runtime/Environment Settings

Environment-level settings that are typically different between dev/staging/production and are therefore kept in YAML, separate from the versioned `config.json`.

```yaml
schema_version: "1.0"
environment: "development"     # development | staging | production
logging:
  min_level: "INFO"            # DEBUG | INFO | WARNING | ERROR | CRITICAL
  sink: "file"                 # file | stdout | both
  json_lines: true
execution:
  parallelism:
    quantization: false
    compression: false
    benchmark: true
  max_concurrent_runs: 4
telemetry:
  enabled: false
  endpoint: null
```

| Field | Type | Consumed by | Notes |
|---|---|---|---|
| `environment` | str | `CompositionRoot` | Selects which `ILogger` sink defaults and which `IExporterBackend` set (real vs. simulated hardware calls) are wired |
| `logging.min_level` | str | `StructuredLogger` (`min_level` ctor param) | |
| `logging.sink` / `json_lines` | str / bool | `StructuredLogger` (`sink_paths` derivation) | |
| `execution.parallelism.*` | bool | maps to `ExecutionConfig.parallelism: Dict[str, bool]` | Keys must match stage names capable of parallel execution: `quantization`, `compression`, `benchmark` only (see `01_Project_Architecture.md` §"Threading Strategy") |
| `execution.max_concurrent_runs` | int | `SessionManager` | Caps `active_runs` dict size; excess requests queue |
| `telemetry.*` | bool / str | `MetricsCollector` | Out of scope for v1 export; reserved |

---

## 3. `hardware.json` — Hardware Subsystem Defaults

Distinct from the per-device files in `hardware_profiles/` (`05_Hardware_Profile_Spec.md`); this file holds cross-cutting hardware-subsystem defaults, not per-board data.

```json
{
  "schema_version": "1.0",
  "default_profile_id": null,
  "profile_cache_enabled": true,
  "strict_compatibility_mode": true,
  "allow_partial_export_on_warning": false
}
```

| Field | Type | Consumed by | Notes |
|---|---|---|---|
| `default_profile_id` | str or null | `HardwareSelectionService` | Used only if `RunRequest.hardware_profile_id` is omitted; `null` forces explicit selection (`ConfigurationError` if absent) |
| `profile_cache_enabled` | bool | `HardwareProfileRepository` | Controls in-memory `cache: Dict[str, HardwareProfile]` usage |
| `strict_compatibility_mode` | bool | `LayerCompatibilityChecker` | When `true`, any `CompatibilityReport.constraint_violations` entry forces `compatible=False`; when `false`, only `unsupported_layers` forces incompatibility |
| `allow_partial_export_on_warning` | bool | `Exporter` | When `true`, `Exporter` may proceed after non-fatal `CompatibilityReport` warnings; default `false` per fail-safe design principle |

---

## 4. `compression.json` — Compression Subsystem Defaults

Maps directly to `CompressionConfig` (`03_API_Specification.md` §1.3) as the default before `CompressionAdvisor` produces a per-run recommendation.

```json
{
  "schema_version": "1.0",
  "enabled_types": ["PRUNING", "WEIGHT_CLUSTERING"],
  "target_ratio": 0.5,
  "pruning_sparsity": 0.3,
  "advisor": {
    "prefer_lossless_when_hardware_allows": true,
    "min_acceptable_accuracy_delta": -0.02
  }
}
```

| Field | Type | Consumed by | Notes |
|---|---|---|---|
| `enabled_types` | List[str] (`CompressionType` enum names) | `CompressionAdvisor` → `CompressionConfig.enabled_types` | Must be a subset of values supported per-hardware (`HardwareProfile` does not directly gate compression, only precision; compression gating is a `CompressionAdvisor` responsibility) |
| `target_ratio` | float (0–1) | → `CompressionConfig.target_ratio` | Target size reduction ratio |
| `pruning_sparsity` | float (0–1) | → `CompressionConfig.pruning_sparsity` | Only meaningful when `PRUNING` in `enabled_types` |
| `advisor.prefer_lossless_when_hardware_allows` | bool | `CompressionAdvisor` | Prefers `HUFFMAN`/`RLE` over `PRUNING`/`WEIGHT_CLUSTERING` when `max_model_size_bytes` headroom permits |
| `advisor.min_acceptable_accuracy_delta` | float | `CompressionAdvisor` | Guardrail; advisor will not recommend a compression combination projected to exceed this negative accuracy delta |

---

## 5. `quantization.json` — Quantization Subsystem Defaults

Maps directly to `QuantizationConfig` (`03_API_Specification.md` §1.3).

```json
{
  "schema_version": "1.0",
  "default_precision": "INT8",
  "per_layer_overrides": {},
  "calibration_batch_size": 32,
  "sensitivity_threshold": 0.01,
  "advisor": {
    "allow_mixed_precision": true,
    "max_precision_levels_considered": 3
  }
}
```

| Field | Type | Consumed by | Notes |
|---|---|---|---|
| `default_precision` | str (`Precision` enum name) | `QuantizationAdvisor` → `QuantizationConfig.default_precision` | Must be a member of the target `HardwareProfile.supported_precisions`; `QuantizationAdvisor` overrides this default if incompatible, logging a warning |
| `per_layer_overrides` | Dict[str, str] | → `QuantizationConfig.per_layer_overrides` | Empty by default; populated at runtime by `QuantizationEngine` using `SensitivityReport` |
| `calibration_batch_size` | int | → `QuantizationConfig.calibration_batch_size`, consumed by `CalibrationEngine` | |
| `sensitivity_threshold` | float | → `QuantizationConfig.sensitivity_threshold`, consumed by `SensitivityAnalyzer` | Accuracy-delta threshold beyond which a layer is excluded from aggressive quantization |
| `advisor.allow_mixed_precision` | bool | `QuantizationAdvisor` | Gates whether `MIXED` is a candidate `default_precision` |
| `advisor.max_precision_levels_considered` | int | `QuantizationAdvisor` | Bounds the advisor's internal search space |

---

## 6. `benchmark.json` — Benchmark Subsystem Defaults

```json
{
  "schema_version": "1.0",
  "trials": 20,
  "warmup_trials": 3,
  "timeout_seconds_per_trial": 30,
  "use_real_hardware_if_available": true,
  "simulation_fallback_enabled": true
}
```

| Field | Type | Consumed by | Notes |
|---|---|---|---|
| `trials` | int | `Benchmarker.run_benchmark(..., trials)` | Default value for the `trials` parameter when not overridden by `config_overrides` |
| `warmup_trials` | int | `Benchmarker` | Excluded from `BenchmarkResult` aggregate statistics |
| `timeout_seconds_per_trial` | int | `Benchmarker` | Raises `BenchmarkError` on timeout |
| `use_real_hardware_if_available` | bool | `Benchmarker` | If `true` and a physical device connection is detected (mechanism is infrastructure-specific, defined per backend), benchmarks run on-device |
| `simulation_fallback_enabled` | bool | `Benchmarker` | If `true`, falls back to `HardwareProfile`-derived analytical estimates when no device is attached; if `false`, raises `BenchmarkError` instead |

---

## 7. `reports.json` — Reporting Subsystem Defaults

```json
{
  "schema_version": "1.0",
  "enabled_reports": ["accuracy", "benchmark", "compression", "deployment", "summary"],
  "output_format": "markdown",
  "include_raw_metrics_appendix": true
}
```

| Field | Type | Consumed by | Notes |
|---|---|---|---|
| `enabled_reports` | List[str] | `ReportGenerator` | Each string must match an `IReportRenderer.report_type()` value; renderers not listed here are skipped even if registered in `PluginRegistry` |
| `output_format` | str: `markdown`\|`html`\|`json` | Each `IReportRenderer` implementation | v1 ships `markdown` renderers only; `html`/`json` reserved for future renderer plugins |
| `include_raw_metrics_appendix` | bool | Each renderer | Controls whether raw `StageResult.payload` dumps are appended after the human-readable summary |

---

## 8. Config Loading & Override Precedence

`ConfigRepository` resolves each `load_*_config()` call with the following precedence, highest first:

1. `RunRequest.config_overrides` (per-run, in-memory, never persisted)
2. The corresponding `config/*.json` file's `advisor`/subsystem-specific block
3. The corresponding `config/*.json` file's top-level defaults
4. Hard-coded fail-safe defaults inside `ConfigRepository` (used only if a file is missing and `strict_compatibility_mode`-style hard failure is not desired — in v1, a missing required file always raises `ConfigurationError` instead; this tier exists for forward compatibility only)

`config_overrides` keys use dotted-path notation matching the JSON structure above, e.g. `"quantization.default_precision": "INT4"`.

---

## 9. Config Validation Contract

1. Every `config/*.json` file MUST declare `schema_version`; `ConfigRepository` raises `ConfigurationError` if absent or if the major version is unrecognized (same rule as `HardwareProfileRepository`, per `05_Hardware_Profile_Spec.md` §7).
2. Enum-valued fields (`default_precision`, `enabled_types`, `on_error`, etc.) MUST be validated against the corresponding `uaqe.common.types` enum at load time, not deferred to point of use.
3. `settings.yaml` is environment-specific and MAY be gitignored per-deployment; `config/*.json` files are version-controlled and part of the Architecture Lock schema (see `09_Architecture_Lock.md`).
4. No config file may introduce a field not documented in this specification without a corresponding update to this document and, if the field maps to a locked value object, an RFC against `09_Architecture_Lock.md`.

---

**End of `06_Config_Spec.md`.**
