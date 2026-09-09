# 13_Developer_Onboarding.md
## Universal AI Quantization Engine — Developer Onboarding Guide

**Depends on:** `01_Project_Architecture.md` through `12_Project_Test_Plan.md`
**Status:** DRAFT — pending Architecture Lock sign-off review of this document
**Scope:** This document does not redefine, override, or restate any binding contract. Wherever this guide summarizes something (folder ownership, class names, config fields, phase order), the authoritative source is the referenced document; if the two ever disagree, the referenced document wins — `09_Architecture_Lock.md` is canonical for anything it covers.

---

## 1. Welcome

Welcome to the **Universal AI Quantization Engine (UAQE)**. This document is the starting point for anyone about to write code, review a pull request, or reason about this system for the first time — human contributor or AI assistant.

UAQE exists because deploying a trained neural network to constrained hardware is currently a fragmented, manual, framework-specific, and hardware-specific problem. A team targeting an FPGA today needs an entirely different toolchain, mental model, and set of manual trade-off decisions than a team targeting an STM32 microcontroller or a Raspberry Pi. UAQE's vision is a single, explainable, hardware-aware system that takes a trained model in any common exchange format and produces a deployment-ready artifact for any of its supported targets — while explaining every decision it made along the way.

This is not a wrapper around existing conversion utilities. It is a decision-making system, and that distinction shapes every rule in this guide.

---

## 2. Project Overview

**What it is:** A hardware-aware Edge AI deployment platform that ingests trained models and produces deployment-ready artifacts through an orchestrated, explainable pipeline (see `01_Project_Architecture.md` §1).

**Supported input frameworks/formats:** PyTorch (`.pth`, `.pt`), ONNX (`.onnx`), TensorFlow (`.pb`), Keras (`.h5`, `.keras`), TFLite (`.tflite`) — each handled by a dedicated `IFrameworkAdapter` implementation (`02_Folder_Structure.md` §5, `03_API_Specification.md` §15).

**Supported hardware targets** (three `HardwareClass` values, twelve profiles):
- **FPGA:** Artix-7, Zynq-7000, Kintex-7, Cyclone V
- **Embedded:** ESP32, ESP32-S3, STM32F4, STM32H7, RP2040, Portenta H7
- **Raspberry Pi:** Raspberry Pi 4, Raspberry Pi 5

**Supported outputs:** hardware-native artifacts per class — `.mem`/`.hex`/`.bin` for FPGA, `.tflite`/`.bin`/`.h` header files for embedded, `.tflite`/`.onnx` for Raspberry Pi (`05_Hardware_Profile_Spec.md` §3–§5) — plus five generated report types (accuracy, benchmark, compression, deployment, summary — `06_Config_Spec.md` §7).

**Overall workflow:** ingest → validate → analyze → check hardware compatibility → recommend quantization → calibrate → analyze sensitivity → quantize → recommend compression → compress → optimize (multi-objective) → optimize memory layout → export → evaluate → benchmark → score deployment readiness → advise → report. The full twenty-step mapping from workflow stage to owning module lives in `01_Project_Architecture.md` §5 — this guide does not repeat it.

**Expected users:** ML engineers deploying to constrained targets, embedded/FPGA firmware teams without deep ML tooling background, research groups studying quantization/compression trade-offs, and — per the module contracts and this guide — future contributors and AI coding assistants extending the system.

---

## 3. Project Goals

**Primary goals:** correctness (never silently produce a broken or oversized artifact — see the fail-safe rules in `05_Hardware_Profile_Spec.md` §6), explainability (every automated decision — quantization plan, compression plan, readiness score — is reportable, not a black box), and hardware fidelity (respecting the physical ceilings of the target device rather than best-effort approximation).

**Secondary goals:** extensibility without core modification (new hardware, new frameworks, new strategies register as plugins/additive files rather than requiring changes to locked contracts — `09_Architecture_Lock.md` §0.2), and reproducibility (deterministic, versioned config and hardware schemas — `06_Config_Spec.md` §9).

**Long-term vision:** a plugin ecosystem for community-contributed exporter backends and quantization/compression strategies, and the architectural seams for distributed/cloud execution noted in `01_Project_Architecture.md` (stateless domain services, horizontal fan-out via `BatchRunner` — §18).

**Research contribution:** a reusable, hardware-agnostic Internal Model Representation (IMR) and a transparent multi-objective optimization framework (accuracy/latency/memory) that can be studied independent of any single hardware backend.

**Industrial relevance:** a single toolchain replacing per-vendor conversion pipelines for teams shipping models to heterogeneous edge fleets (FPGA + MCU + SBC) from one codebase.

---

## 4. System Architecture Summary

UAQE is a **Clean / Layered Architecture**: Interface → Application → Domain (core) → Infrastructure, with dependencies pointing strictly inward (`01_Project_Architecture.md` §2–§3). The Domain layer contains all ML/hardware decision logic and is framework-agnostic — it never imports `torch`, `tensorflow`, or `onnx`, operating only on the Internal Model Representation (IMR). Infrastructure implements Domain-defined interfaces (Ports and Adapters), and only `interface/composition_root.py` is permitted to wire concrete Infrastructure classes into the system.

The pipeline is a strict, sequential chain of `PipelineStage` subclasses orchestrated by `PipelineOrchestrator`, each stage reading prior results from and writing its own result to a shared `PipelineContext` (`03_API_Specification.md` §2, `04_Data_Flow.md` §1). Stage N+1 always depends on stage N's committed state — stages are never parallelized against each other, though internal work within a stage may be (`01_Project_Architecture.md` §16).

Core modules, in pipeline order: `ModelLoader` → `ModelValidator` → `ModelAnalyzer` → `LayerCompatibilityChecker`/`HardwareManager` → `QuantizationAdvisor` → `CalibrationEngine` → `SensitivityAnalyzer` → `QuantizationEngine` → `CompressionAdvisor` → `CompressionEngine` → `OptimizationEngine`/`MemoryOptimizer` → `Exporter` → `Evaluator` → `Benchmarker` → `DeploymentReadinessScorer` → `OptimizationAdvisor` → `ReportGenerator`.

This is a summary for orientation only. For binding signatures, always consult `03_API_Specification.md`; for binding call/data sequencing, `04_Data_Flow.md`; for what is permanently frozen, `09_Architecture_Lock.md`.

---

## 5. Folder Structure Overview

Quick orientation only — the authoritative tree, with every file name, lives in `02_Folder_Structure.md`.

| Folder | Purpose | Owner |
|---|---|---|
| `src/uaqe/common/` | Framework-agnostic primitives (IMR, enums, value objects, exceptions, interfaces) | Architecture team (shared) |
| `src/uaqe/domain/` | All ML/hardware decision logic, zero ML-framework imports | Domain module leads |
| `src/uaqe/application/` | Orchestration/session lifecycle, no ML or hardware logic | Application layer lead |
| `src/uaqe/infrastructure/` | Concrete integrations: framework parsing, hardware export, config/log I/O | Infrastructure module leads (per adapter family) |
| `src/uaqe/interface/` | External entry points and the sole Composition Root | Interface layer lead |
| `config/` | Versioned JSON/YAML defaults, read-only at runtime | `ConfigRepository` |
| `hardware_profiles/` | Per-board hardware database, read-only at runtime | `HardwareProfileRepository`/`HardwareManager` |
| `datasets/` | User-supplied calibration/evaluation data, read-only | `CalibrationEngine`/`Evaluator` |
| `workdir/` | Ephemeral per-run scratch space | `PipelineOrchestrator` |
| `outputs/` | Final deployment artifacts | `Exporter` |
| `reports/` | Generated report documents | `ReportGenerator` |
| `logs/` | Structured JSON-lines logs | `StructuredLogger` |
| `tests/` | 1:1 mirror of `src/uaqe/`, plus integration/fixtures | Each module's owning developer |
| `plugins/` | External/custom plugin drop-in directory | `PluginRegistry` |

No folder is written to by more than one module at runtime (`02_Folder_Structure.md` §19). Do not read this table as a substitute for the full ownership rules there.

---

## 6. Development Workflow

1. **Architecture** — confirm the change fits within `01`–`09`; if it requires a new class, method, folder, schema field, or enum member not already present, an RFC against `09_Architecture_Lock.md` is required before writing code (unless it is explicitly "open for addition" — see §7 below).
2. **Implementation** — build against the exact contract in `03_API_Specification.md`, following the construction/injection patterns in `11_Implementation_Rules.md` and the module template in `10_Module_Development_Guide.md`.
3. **Testing** — write the mirrored unit test module first or alongside implementation (`07_Coding_Standards.md` §10); add contract tests for any new interface implementation; add integration coverage per `12_Project_Test_Plan.md` where the change touches pipeline flow.
4. **Review** — open a PR on a short-lived feature branch; at least one approval from outside the module's primary owner; two approvals plus an RFC reference for anything touching a locked file (`07_Coding_Standards.md` §14).
5. **Integration** — CI must pass: `black`, `isort`, `flake8`/`ruff`, `mypy --strict`, `pytest` with coverage gates, `import-linter` (`07_Coding_Standards.md` §11, `08_Project_Roadmap.md` Phase 15).
6. **Release** — merged work accumulates toward the phase/version goals in `08_Project_Roadmap.md`; see §25 of this guide for the release process.
7. **Maintenance** — bug fixes and additive changes follow the same review bar; renames or signature changes to anything locked always require an RFC, no exceptions for urgency.

---

## 7. Development Rules

Four documents govern every implementation decision and must be consulted, in this order, before writing a line of code:

1. **`09_Architecture_Lock.md`** — is the thing you're about to touch locked? If yes, stop and check whether an RFC is required (renames, signature changes, schema changes) or whether it's additive (new hardware profile file, new plugin, new backend within an existing class — no RFC needed).
2. **`03_API_Specification.md`** — what is the exact signature, constructor, exception contract for the class you're implementing or calling?
3. **`07_Coding_Standards.md`** — naming, formatting, import direction, logging format, error handling, type hints, docstrings, testing minimums.
4. **`11_Implementation_Rules.md`** — how to actually wire it: constructor injection rules, the pipeline stage template, repository/factory patterns.

`04_Data_Flow.md` is the fifth document to check whenever the change affects what a stage reads from or writes to `PipelineContext`.

---

## 8. Module Ownership

| Module | Purpose | Owner | Dependencies | Status | Future Expansion |
|---|---|---|---|---|---|
| `common` | Framework-agnostic primitives (IMR, enums, value objects, exceptions, interfaces) | Architecture team | None | DRAFT — pending lock | New value objects via RFC only |
| `domain.pipeline` | Stage orchestration primitives (context, stage, builder) | Domain lead | `common` | DRAFT | New stage types via RFC |
| `domain.model` | Loading, validation, analysis | Domain lead | `common` | DRAFT | New analysis metrics |
| `domain.compatibility` | Layer/hardware compatibility checking | Domain lead | `common` | DRAFT | New constraint categories |
| `domain.hardware` | Hardware profile resolution & compatibility orchestration | Domain lead | `common` | DRAFT | New `HardwareClass` via RFC |
| `domain.quantization` | Advisor, calibration, sensitivity, engine | Domain lead | `common` | DRAFT | New `IQuantizationStrategy` plugins |
| `domain.compression` | Advisor, engine | Domain lead | `common` | DRAFT | New `ICompressionStrategy` plugins |
| `domain.optimization` | Multi-objective optimization, memory planning | Domain lead | `common` | DRAFT | New objectives |
| `domain.export` | Export orchestration | Domain lead | `common` | DRAFT | New `IExporterBackend` plugins |
| `domain.evaluation` | Accuracy evaluation | Domain lead | `common` | DRAFT | New metrics |
| `domain.benchmark` | Latency/throughput benchmarking | Domain lead | `common` | DRAFT | New hardware benchmark drivers |
| `domain.advisory` | Readiness scoring, recommendations | Domain lead | `common` | DRAFT | New scoring factors |
| `domain.reporting` | Report content generation | Domain lead | `common` | DRAFT | New `IReportRenderer` plugins |
| `application` | Run lifecycle, session, workflow control | Application lead | `domain`, `common` | DRAFT | — |
| `infrastructure.framework_adapters` | Torch/ONNX/TF/Keras/TFLite parsing | Infrastructure lead(s) | `common` | DRAFT | New adapters (MXNet, CoreML, PaddlePaddle — see §17) |
| `infrastructure.exporter_backends` | Per-board export implementations | Infrastructure lead(s) | `common` | DRAFT | New backends per new board/class |
| `infrastructure.repositories` | Config, hardware profile, filesystem I/O | Infrastructure lead | `common` | DRAFT | — |
| `infrastructure.logging` / `metrics` | Structured logging, metrics collection | Infrastructure lead | `common` | DRAFT | — |
| `infrastructure.plugins` | Plugin discovery/registration | Infrastructure lead | `common` | DRAFT | Community plugin ecosystem |
| `interface` | CLI, REST, batch entry points, Composition Root | Interface lead | All layers | DRAFT | GUI, web dashboard (see §26) |

Status reflects the document set's current state (`DRAFT — pending Architecture Lock`). Once `09_Architecture_Lock.md` is formally signed off, update this column to `LOCKED` / `IN PROGRESS` / `IMPLEMENTED` per module as work proceeds.

---

## 9. Recommended Development Order

This mirrors the phase sequence in `08_Project_Roadmap.md` — consult that document for full objectives/deliverables/expected outputs per phase. Summary order:

1. `common` (enums, IMR, value objects, exceptions, result types, interfaces)
2. Logging & configuration infrastructure (`StructuredLogger`, `ConfigRepository`, `FilesystemRepository`)
3. Hardware profile subsystem (`HardwareProfileRepository`, the 12 hardware JSON files, `HardwareManager` resolution)
4. Framework adapters & model ingestion (`ModelLoader`, all five `IFrameworkAdapter`s)
5. Model validation, analysis & compatibility (`ModelValidator`, `ModelAnalyzer`, `LayerCompatibilityChecker`)
6. Quantization subsystem (`QuantizationAdvisor`, `CalibrationEngine`, `SensitivityAnalyzer`, `QuantizationEngine`)
7. Compression subsystem (`CompressionAdvisor`, `CompressionEngine`)
8. Optimization & memory planning (`OptimizationEngine`, `MemoryOptimizer`)
9. Exporter & backends (`Exporter`, all eleven `IExporterBackend` implementations)
10. Evaluation & benchmark (`Evaluator`, `Benchmarker`)
11. Advisory & reporting (`DeploymentReadinessScorer`, `OptimizationAdvisor`, `ReportGenerator`)
12. Pipeline orchestration & application layer (`PipelineOrchestrator`, `WorkflowController`, `SessionManager`)
13. Interface layer & Composition Root (CLI, REST, batch, wiring)
14. Hardening — full test suite, coverage gates
15. Documentation, samples, v1 release

Visualization/GUI and further integration work is explicitly post-v1 (see §26).

---

## 10. Git Workflow

1. **Clone** the repository and verify the tree matches `02_Folder_Structure.md`.
2. **Branch** from `main` using the naming convention in §11 below.
3. **Develop** against the contract in `03_API_Specification.md`, following §6–§7 of this guide.
4. **Test** locally: `pytest`, `mypy --strict`, `black`/`isort`/`ruff`, `import-linter` — all must pass before pushing.
5. **Commit** using Conventional Commits (§12 below).
6. **Push** the feature branch — never commit directly to `main`.
7. **Pull Request** — describe the change, link the roadmap phase/module, include an RFC reference if the change touches a locked item.
8. **Merge** only after CI is green and the review bar in §6 is met.
9. **Release** — tag once a roadmap phase (or version milestone) is complete and verified (see §25).
10. **Tag** using semantic versioning once release criteria are met.

`main` is always deployable; trunk-based development, no direct commits (`07_Coding_Standards.md` §11).

---

## 11. Branch Naming Convention

Format: `<type>/<module>-<short-description>`, where `<type>` ∈ `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

```
feat/quantization-sensitivity-analyzer
fix/exporter-artix7-overflow
docs/config-spec-update
refactor/pipeline-stage-resolution
test/compression-contract-tests
chore/deps-onnx-adapter-bump
```

One module per branch where practical; cross-module branches require sign-off from both module owners (`07_Coding_Standards.md` §12).

---

## 12. Commit Message Convention

Conventional Commits: `<type>(<scope>): <description>`, where `<scope>` matches a module name from `02_Folder_Structure.md`.

```
feat(quantization): add per-layer sensitivity threshold override
fix(exporter): guard against artix7 BRAM overflow on export
docs(config): document reports.json output_format field
refactor(pipeline): extract stage resolution into PipelineBuilder
test(compression): add contract tests for ICompressionStrategy
chore(deps): bump onnx adapter dependency pin
```

---

## 13. Development Environment

- **Python version:** 3.11 minimum, tested up to 3.12; `pyproject.toml` pins `requires-python = ">=3.11,<3.13"` (`07_Coding_Standards.md` §1).
- **Virtual environment:** use a project-local `venv` or equivalent; never install project dependencies globally.
- **Dependency installation:** via `pyproject.toml`; framework-specific dependencies (`torch`, `tensorflow`, `onnx`, etc.) are scoped to their owning `infrastructure/framework_adapters/` module only — never a dependency of `uaqe.domain` or `uaqe.common` (`07_Coding_Standards.md` §14 rule 4).
- **IDE recommendations:** any editor with first-class `mypy --strict` and `black`/`isort` integration; PyCharm or VS Code with the Python/Pylance and Ruff extensions are both suitable.
- **OS support:** development is OS-agnostic (pure Python); hardware-in-the-loop benchmarking (§15) requires the relevant board/toolchain to be attached or a simulation fallback per `06_Config_Spec.md` §6.

---

## 14. Running the Project

- **Installation:** install the package and its dependency groups per `pyproject.toml` (framework adapter extras installed as needed for the frameworks you're working with).
- **Configuration:** populate `config/*.json` and `settings.yaml` per `06_Config_Spec.md`; at minimum `config.json` and `settings.yaml` must exist with valid `schema_version` fields, or `ConfigRepository` raises `ConfigurationError`.
- **Running:** invoke via the CLI entry point (`interface/cli/cli_entry_point.py`), the REST entry point, or the batch runner, all wired through `CompositionRoot.build_workflow_controller()` (`03_API_Specification.md` §21).
- **Testing:** `pytest` from the repository root; `pytest-cov` enforces the coverage gates in §15 below.
- **Generating reports:** reports are produced automatically as the final pipeline stage (`ReportGenerator`) and written under `reports/<run_id>/` per `enabled_reports` in `reports.json`.
- **Cleaning outputs:** `workdir/<run_id>/` is safe to delete between runs (ephemeral scratch space); `outputs/`, `reports/`, and `logs/` are the durable, per-run artifacts and should be cleaned deliberately, not automatically.

---

## 15. Testing Workflow

Per `07_Coding_Standards.md` §10 and `12_Project_Test_Plan.md`:

- **Unit tests:** 1:1 mirror of every `src/uaqe/` module under `tests/unit/`; CI fails the build if a source file lacks a matching test file.
- **Integration tests:** full pipeline runs against `tests/fixtures/sample_models/`, at least one model per supported input format and one hardware profile per `HardwareClass`, under `tests/integration/pipeline_end_to_end/` and `tests/integration/exporter_backends/`.
- **Performance tests:** benchmark-path validation against `Benchmarker`'s trial/timeout/simulation-fallback behavior (`06_Config_Spec.md` §6).
- **Stress tests:** large-model threshold behavior, streamed-artifact fallback, concurrent run limits (`config.json.large_model_threshold_mb`, `settings.yaml.execution.max_concurrent_runs`).
- **Regression tests:** re-run of prior integration fixtures on every release candidate per `12_Project_Test_Plan.md`.
- **Coverage:** minimum 85% line coverage on `uaqe.domain` and `uaqe.application`; minimum 70% on `uaqe.infrastructure` (lower bar due to hardware-dependent paths requiring mocked I/O).

Every `IFrameworkAdapter`, `IExporterBackend`, `IQuantizationStrategy`, `ICompressionStrategy`, and `IReportRenderer` implementation additionally requires a shared, parametrized contract test proving interchangeability with every other implementation of that interface.

---

## 16. Module Development Checklist

- [ ] **Architecture:** confirmed against `01`, `02`, `09` — correct layer, correct folder, nothing locked being renamed without an RFC.
- [ ] **Interfaces:** implements the exact abstract class/signature in `03_API_Specification.md`; all constructor dependencies are interface-typed and injected, never instantiated internally (`11_Implementation_Rules.md` §3).
- [ ] **Logging:** goes through `ILogger` only; structured JSON-lines fields (`run_id`, `stage_name`, etc.) included; correct severity level used (`07_Coding_Standards.md` §6).
- [ ] **Exceptions:** every failure path raises a `UAQEError` subclass with `code`, `message`, `stage` set; no bare `except Exception`; third-party exceptions re-raised with `raise ... from err`.
- [ ] **Tests:** mirrored unit test module written; contract test added if implementing a shared interface; coverage target met for the owning layer.
- [ ] **Documentation:** module-level docstring stating single responsibility; Google-style docstrings on every public class/method with `Args`/`Returns`/`Raises` where applicable.
- [ ] **Performance:** no unnecessary tensor duplication across stages (`01_Project_Architecture.md` §17); peak-memory budget respected before heavy stages; parallelism only where explicitly sanctioned (`01_Project_Architecture.md` §16).

---

## 17. Adding a New Framework

General process, illustrated with MXNet, CoreML, and PaddlePaddle as examples:

1. Confirm the new framework isn't already covered and that adding it doesn't require a new `HardwareClass` or IMR change (pure format-ingestion addition).
2. Implement a new `I<Framework>Adapter` class implementing `IFrameworkAdapter` under `infrastructure/framework_adapters/`, matching the identical contract shown for existing adapters in `03_API_Specification.md` §15 (`load(path) -> IMR`, `supports(extension) -> bool`).
3. Scope the new third-party dependency (e.g. `mxnet`, `coremltools`, `paddlepaddle`) to that single adapter file's dependency group in `pyproject.toml` — never a dependency of `uaqe.domain`.
4. All framework-specific load failures must be caught and re-raised as `ModelLoadError`; no raw third-party exception may leak past the adapter.
5. Register the adapter with `ModelLoader`'s `adapters` list (via `CompositionRoot`, or `PluginRegistry` if shipped as an external plugin).
6. Add the file extension(s) to `ModelLoader.detect_framework()`'s recognized set.
7. Write the mirrored unit test plus a contract test proving the new adapter satisfies the shared `IFrameworkAdapter` test suite.
8. Add at least one fixture model under `tests/fixtures/sample_models/` for the new format and extend the integration test matrix in `12_Project_Test_Plan.md` §4.2.

This is additive — no RFC is required as long as no locked name/signature changes.

---

## 18. Adding a New Hardware Target

1. **Hardware database:** add a new profile JSON file under the correct `hardware_profiles/<class>/` subfolder, following the exact schema in `05_Hardware_Profile_Spec.md` §2 (all fields present, `null` where not applicable — never omitted).
2. **Compatibility checker:** no code change needed for a target within an existing `HardwareClass`; `LayerCompatibilityChecker` and `QuantizationAdvisor`/`CompressionAdvisor` already read `supported_precisions`/`constraints` generically. A genuinely new `HardwareClass` requires an RFC (`05_Hardware_Profile_Spec.md` §7.2).
3. **Exporter:** implement a new `IExporterBackend` under `infrastructure/exporter_backends/<class>/`, reading resource limits from the `HardwareProfile` rather than hard-coding them (Contract Rule, `05_Hardware_Profile_Spec.md` §3).
4. **Benchmark:** ensure `Benchmarker`'s simulation fallback can derive analytical estimates from the new profile's fields if no physical device is attached.
5. **Testing:** contract test for the new `IExporterBackend`; add the new profile to the hardware-profile deserialization test matrix (`08_Project_Roadmap.md` Phase 4 expected outputs).

---

## 19. Adding a New Quantization Strategy

1. **Strategy class:** implement `IQuantizationStrategy` (`apply(imr, plan) -> IMR`, `name() -> str`), raising `QuantizationError` on failure.
2. **Registration:** register with `PluginRegistry.register_quantization_strategy()` (in-tree) or via plugin discovery under `plugins/quantization_strategies/` (external).
3. **Configuration:** if the strategy needs new tunables, propose an addition to `quantization.json`'s `advisor` block per `06_Config_Spec.md` §5 — additive fields only; anything touching `QuantizationConfig`'s locked shape needs an RFC.
4. **Testing:** shared `IQuantizationStrategy` contract test suite, plus strategy-specific unit tests for its numerical behavior.
5. **Benchmark:** validate the strategy's accuracy/latency trade-off through the standard `Evaluator`/`Benchmarker` pipeline before recommending it for production use in `QuantizationAdvisor`.

---

## 20. Adding a New Compression Algorithm

1. **Registration:** add a new `CompressionType` enum member only via RFC (locked enum, `09_Architecture_Lock.md`); implementations of existing types are additive.
2. **Implementation:** implement `ICompressionStrategy` (`apply(imr, plan) -> IMR`, `name() -> str`), raising `CompressionError` on failure.
3. **Testing:** shared `ICompressionStrategy` contract test suite plus algorithm-specific tests.
4. **Compatibility:** `CompressionAdvisor` must filter candidate types against what the target hardware and current model state can support before recommending — do not bypass this filter in the new implementation.

---

## 21. Adding a New Export Backend

1. **Exporter interface:** implement `IExporterBackend` (`export(imr, target) -> DeploymentArtifact`, `supported_targets() -> List[str]`), matching every existing backend's identical contract (`03_API_Specification.md` §16).
2. **Hardware backend:** validate final serialized size against `HardwareProfile.max_model_size_bytes` before writing any file; for FPGA targets, validate projected resource utilization against `HardwareProfile.fpga_resources` (`05_Hardware_Profile_Spec.md` §6).
3. **Output files:** write only through `FilesystemRepository`; never touch `outputs/` directly.
4. **Validation:** raise `ExportError` on any overflow or unsupported-layer condition — never silently truncate or best-effort export unless `allow_partial_export_on_warning` is explicitly `true` in `hardware.json`.

---

## 22. Common Development Mistakes

- **Architecture violations:** putting ML/hardware decision logic in `application/` or `interface/`; importing a concrete Infrastructure class outside `composition_root.py`.
- **Dependency violations:** importing `torch`/`tensorflow`/`onnx` anywhere under `uaqe.domain` or `uaqe.common`.
- **Circular imports:** cross-module domain calls made by direct concrete reference instead of through `PipelineOrchestrator`/injected interfaces.
- **Logging mistakes:** using bare `print()` or the stdlib `logging` module directly instead of the injected `ILogger`; omitting `run_id`/`stage_name` structured fields.
- **Configuration mistakes:** hard-coding a default instead of reading it from `ConfigRepository`; forgetting `schema_version` in a new config file; treating `config_overrides` precedence incorrectly (`06_Config_Spec.md` §8).
- **Testing mistakes:** shipping a source file with no mirrored test file; skipping the shared contract test suite for a new interface implementation.
- **Performance mistakes:** duplicating large tensors across stages without justification; parallelizing across pipeline stages (architecturally disallowed, not just discouraged — `01_Project_Architecture.md` §16).

---

## 23. Frequently Asked Questions

**Architecture**
1. *Can I add a new top-level folder?* Only via RFC against `09_Architecture_Lock.md` §1.
2. *Can `domain` import from `infrastructure`?* Never — dependency direction is locked (`07_Coding_Standards.md` §5).
3. *Who is allowed to instantiate concrete Infrastructure classes?* Only `CompositionRoot`.
4. *What happens if a document in `01`–`09` disagrees with `09_Architecture_Lock.md`?* The lock document wins.
5. *Is `IMR` allowed as an abbreviation for new code?* Yes, it's an established acronym; do not invent new ones.

**Development**
6. *What Python version should I target?* 3.11 minimum, tested to 3.12.
7. *Can I use relative imports?* No — absolute imports only.
8. *What formatter is authoritative?* `black`, line length 100; disputes are settled by `black`.
9. *Where do framework-specific dependencies go in `pyproject.toml`?* Scoped to their owning adapter file's dependency group only.
10. *Can a `PipelineStage` return `None`?* Never — always a `StageResult`.

**Testing**
11. *Is 100% coverage required?* No — 85% for `domain`/`application`, 70% for `infrastructure`.
12. *What if I add a source file without a test file?* CI fails the build; the 1:1 mirror rule is enforced mechanically.
13. *What is a contract test?* A shared, parametrized test suite run against every implementation of a given interface to guarantee interchangeability.
14. *Do integration tests need real hardware?* No — `simulation_fallback_enabled` in `benchmark.json` allows analytical estimates when no device is attached.
15. *Where do sample models for tests live?* `tests/fixtures/sample_models/`.

**Hardware**
16. *How many hardware profiles exist today?* Twelve, across three classes (FPGA, Embedded, Raspberry Pi).
17. *Can I add a board without touching code?* Yes, if it belongs to an existing `HardwareClass` and a compatible `IExporterBackend` already exists.
18. *What if a profile field doesn't apply to a board?* Set it to `null` explicitly — never omit the key.
19. *Who enforces `max_model_size_bytes`?* The selected `IExporterBackend`, before writing any output file.
20. *What ceiling does `MemoryOptimizer` treat as hard?* `HardwareProfile.tensor_memory_bytes`, distinct from `ram_bytes`/`flash_bytes`.

**Export**
21. *What does `Exporter` do on overflow?* Raises `ExportError` — never silently truncates.
22. *Can partial export happen on a warning?* Only if `hardware.json`'s `allow_partial_export_on_warning` is `true`; default is `false`.
23. *Where do FPGA-specific resource checks live?* In each FPGA `IExporterBackend`, reading `HardwareProfile.fpga_resources`.

**Quantization**
24. *Who filters unsupported precisions?* `QuantizationAdvisor`, against `HardwareProfile.supported_precisions`, regardless of `config_overrides`.
25. *What triggers a layer being excluded from aggressive quantization?* Exceeding `sensitivity_threshold` in `quantization.json`.
26. *Is mixed precision always available?* Only if `advisor.allow_mixed_precision` is `true` in `quantization.json`.

**Compression**
27. *Does `HardwareProfile` gate compression types directly?* No — only precision; compression gating is `CompressionAdvisor`'s responsibility.
28. *What guardrail limits compression aggressiveness?* `advisor.min_acceptable_accuracy_delta` in `compression.json`.

**Performance**
29. *Can pipeline stages run in parallel with each other?* No — architecturally disallowed; only work *within* a stage may parallelize.
30. *What happens for very large models?* IMR snapshots stream to `workdir/<run_id>/artifacts/` instead of staying fully in memory once `large_model_threshold_mb` is exceeded.

---

## 24. Contribution Guidelines

- **How to contribute:** pick a module from §8/§9 that matches the current roadmap phase; confirm no open PR already covers it; follow §6's workflow.
- **How to submit a PR:** short-lived feature branch (§11 naming), Conventional Commits (§12), PR description linking the relevant roadmap phase and, if applicable, an RFC.
- **Review process:** minimum one approval from outside the module's primary owner; two approvals plus an explicit RFC reference for anything touching `09_Architecture_Lock.md`; reviewers check spec conformance, data-flow compliance, and test coverage before style.
- **Documentation requirements:** module-level docstring, full Google-style docstrings on public methods, and an update to this guide's Module Ownership table (§8) if status changes.
- **Testing requirements:** mirrored unit tests, contract tests for new interface implementations, coverage gates per §15.

---

## 25. Release Process

- **Development:** work proceeds phase-by-phase per `08_Project_Roadmap.md`; a phase is not "done" until its tests pass (roadmap principle §0.3).
- **Testing:** full suite (`12_Project_Test_Plan.md`) run against the accumulated codebase before any release candidate is cut.
- **Release Candidate:** cut once a roadmap phase's expected outputs are all met and CI is green on `main`.
- **Stable Release:** promoted from a release candidate after regression and acceptance testing (`12_Project_Test_Plan.md` §9, §14) pass with no open blocking defects.
- **Versioning:** semantic versioning; the `v1.x` line is what `09_Architecture_Lock.md` freezes — breaking changes to locked items require a major version bump and an approved RFC.
- **Git Tags:** applied at the stable release commit on `main`.
- **Release Notes:** summarize completed roadmap phase(s), new hardware/framework/strategy additions, and any RFC-approved changes since the last release.

---

## 26. Future Expansion

Explicitly out of scope for v1 but architecturally anticipated:

- **New frameworks:** MXNet, CoreML, PaddlePaddle (§17 process already supports this additively).
- **New hardware:** additional boards within existing classes are additive; new `HardwareClass` values require an RFC (`05_Hardware_Profile_Spec.md` §7.2).
- **Plugin ecosystem:** community-contributed strategies/backends via `plugins/` and `PluginRegistry.discover()`.
- **Cloud integration:** the stateless-domain-service design (`01_Project_Architecture.md` §18) is the seam for future cloud/distributed execution.
- **Distributed optimization:** `OptimizationEngine`'s embarrassingly-parallel search space is a candidate for distributed execution without redesign.
- **GUI / Web dashboard / REST-facing product surface:** the `interface` layer already isolates entry points from domain logic, making a GUI or web dashboard an additive Interface-layer client rather than a core rewrite.
- **Per-profile calibration hints:** anticipated but explicitly deferred hardware profile field (`05_Hardware_Profile_Spec.md` §7.4).

---

## 27. Developer Resources

- **Documentation:** this document set (`01`–`13`) is the primary reference; keep `09_Architecture_Lock.md` open while implementing anything non-trivial.
- **Coding guidelines:** `07_Coding_Standards.md`, `11_Implementation_Rules.md`.
- **Testing references:** `12_Project_Test_Plan.md`.
- **Software architecture references:** Clean Architecture and Ports-and-Adapters (Hexagonal Architecture) literature — UAQE's layering is a direct application of both.
- **Edge AI / embedded AI references:** TensorFlow Lite Micro documentation (relevant to the `tflite-micro` runtime used by every embedded profile); general quantization-aware training and post-training quantization literature.
- **FPGA references:** vendor datasheets for Xilinx Artix-7/Zynq-7000/Kintex-7 and Intel Cyclone V — the exact source for the `fpga_resources` figures in `05_Hardware_Profile_Spec.md` §3.

---

## 28. Complete Project Checklist

- [ ] Architecture: `01`–`09` reviewed and `09_Architecture_Lock.md` formally signed off.
- [ ] Implementation: all modules in §8 built against `03_API_Specification.md`, in the order in §9.
- [ ] Testing: 1:1 unit test mirror complete; contract tests for every interface implementation; integration suite covering every `HardwareClass` and every input format.
- [ ] Benchmarking: real-hardware or simulation-fallback benchmark results captured for every supported profile.
- [ ] Documentation: every public class/method has Google-style docstrings; this onboarding guide's Module Ownership table reflects current status.
- [ ] Deployment: CLI, REST, and batch entry points all exercised end-to-end via `CompositionRoot`.
- [ ] Release: coverage gates met, CI green, release notes drafted, tag applied.
- [ ] Maintenance: RFC process (`09_Architecture_Lock.md` §14) active and followed for any change to a locked item.

---

## 29. New Developer Quick Start

1. Read §1–§4 of this guide (10 minutes) for orientation.
2. Skim `02_Folder_Structure.md` to know where things live.
3. Open `09_Architecture_Lock.md` — this is what you cannot casually change.
4. Pick a module from §8 whose status matches the current active roadmap phase (`08_Project_Roadmap.md`).
5. Read that module's entry in `10_Module_Development_Guide.md` and its exact contract in `03_API_Specification.md`.
6. Branch (§11), implement against `11_Implementation_Rules.md`'s construction patterns, write mirrored tests, and open a PR (§6, §24).

If you've done steps 1–4 within 30 minutes, you're ready to write your first test and your first implementation stub.

---

## 30. Final Notes

UAQE's philosophy is separation of concerns taken seriously: decision logic, mechanics, and hardware knowledge are deliberately decoupled, and that separation is what makes the system extensible without becoming fragile. Quality expectations are high by design — full type hints, `mypy --strict`, 85%/70% coverage gates, and contract tests for interchangeability are not bureaucracy, they are what makes "add a new board" or "add a new framework" a same-day, additive change instead of a risky core rewrite.

Coding expectations follow directly from `07_Coding_Standards.md` and `11_Implementation_Rules.md`: every contributor, human or AI, is expected to implement against the contract, not around it. Long-term maintainability comes from the lock/RFC discipline in `09_Architecture_Lock.md` — it is deliberately harder to change something locked than to add something new, and that asymmetry is intentional.

The vision for the future is a system where adding hardware, frameworks, or strategies is routine, and where every artifact UAQE produces comes with a legible explanation of why it was built that way.

---

**End of `13_Developer_Onboarding.md`.**
