# 07_Coding_Standards.md
## Universal AI Quantization Engine — Coding Standards

**Depends on:** `01_Project_Architecture.md` through `06_Config_Spec.md`
**Status:** DRAFT — pending Architecture Lock
**Scope:** Binding conventions for every contributor and every future Claude conversation implementing a module against `03_API_Specification.md`.

---

## 1. Python Version

- Target: **Python 3.11** minimum, tested up to 3.12.
- No use of syntax or stdlib features unavailable in 3.11 (e.g. no `match` fallthrough tricks assuming 3.12-only behavior).
- `pyproject.toml` pins `requires-python = ">=3.11,<3.13"`.

## 2. PEP8 & Formatting

- `black` (line length 100) is the sole formatter; no manual formatting debates — if `black` and a reviewer disagree, `black` wins.
- `isort` (profile `black`) orders imports: stdlib → third-party → `uaqe.*` (absolute imports only, never relative `from . import`).
- `flake8` or `ruff` enforced in CI for anything `black`/`isort` don't cover (unused imports, shadowed names).

## 3. Naming Convention

| Element | Convention | Example |
|---|---|---|
| Module (file) | `snake_case.py` | `model_loader.py` |
| Package (folder) | `snake_case` | `quantization/` |
| Class | `PascalCase` | `QuantizationEngine` |
| Interface (abstract class) | `IPascalCase` | `IQuantizationStrategy` |
| Method / function | `snake_case` | `execute()`, `load()` |
| Constant | `UPPER_SNAKE_CASE` | `HARDWARE_SCHEMA_VERSION` |
| Private attribute | `_leading_underscore` at implementation level; documented as `-` in `03_API_Specification.md` | `self._cache` |
| Protected attribute | single leading underscore, documented as `#` | shared base-class state |
| Enum member | `UPPER_SNAKE_CASE` | `Precision.INT8` |
| Dataclass | `PascalCase`, noun phrase | `AnalysisResult`, `ReadinessScore` |

No abbreviations beyond those already present in `02_Folder_Structure.md` / `03_API_Specification.md` (e.g. `IMR` is an established acronym and stays; do not invent new ones like `QCfg` for `QuantizationConfig`).

## 4. Folder Naming

- All folders `snake_case`, matching `02_Folder_Structure.md` exactly — no contributor may rename, merge, or split a locked folder without an RFC (`09_Architecture_Lock.md`).
- Every package folder contains an `__init__.py`, even if empty, to keep import paths explicit (no implicit namespace packages).

## 5. Import Strategy

- Absolute imports only: `from uaqe.domain.quantization.quantization_engine import QuantizationEngine`.
- `uaqe.domain` and `uaqe.common` MUST NOT import anything from `uaqe.infrastructure` or `uaqe.interface` (enforced by `05_Coding_Standards` import-linter config, e.g. `import-linter` contract in `pyproject.toml`).
- Only `uaqe.interface.composition_root` may import concrete classes from `uaqe.infrastructure`.
- No wildcard imports (`from x import *`) anywhere.
- No circular imports; `import-linter` runs in CI to catch layering violations mechanically, not just by review.

## 6. Logging Format

- All logging goes through `ILogger` (never bare `print()` or the stdlib `logging` module directly from domain/application code).
- Structured, JSON-lines format: every log call carries `run_id`, `stage_name` (where applicable), and arbitrary `**fields` as shown in `03_API_Specification.md` §1.6.
- Log levels map strictly to severity: `debug` (verbose internals), `info` (stage start/end, milestones), `warning` (recoverable issue, `StageResult.warnings` entry), `error` (stage failed, exception raised), `critical` (pipeline-halting, unrecoverable).

## 7. Error Handling

- Every raised exception in `uaqe.domain` and `uaqe.application` MUST be a subclass of `UAQEError` (`03_API_Specification.md` §1.4). Third-party exceptions from framework adapters MUST be caught and re-raised as the appropriate `UAQEError` subclass (e.g. a `torch` load failure becomes `ModelLoadError`, never leaks a raw `torch` exception past `infrastructure/framework_adapters/`).
- Every `UAQEError` MUST set `code`, `message`, and `stage`; `remediation_hint` is strongly encouraged wherever a corrective action is known.
- No bare `except:` or `except Exception:` swallowing; catch specific exceptions, re-raise as `UAQEError` subclasses with context preserved (`raise ... from err`).
- `PipelineOrchestrator` is the only component that interprets `ExecutionConfig.on_error`; individual stages always raise on failure and never silently continue.

## 8. Type Hints

- Full type hints on every public method signature, matching `03_API_Specification.md` exactly (return types, parameter types, `Optional[...]`, `List[...]`, `Dict[..., ...]`).
- `mypy --strict` enforced in CI. No `# type: ignore` without an inline comment explaining why.
- Dataclasses and value objects use `@dataclass(frozen=True)` for anything documented as `@immutable` in the API spec.

## 9. Documentation Style

- Google-style docstrings on every public class and method: `Args:`, `Returns:`, `Raises:` sections mandatory wherever the method has parameters, a return value, or can raise.
- Module-level docstring at the top of every file stating its single responsibility in one sentence, consistent with its entry in `02_Folder_Structure.md`.
- No docstring may restate the type hint without adding semantic meaning (e.g. not `"path: str - the path"` but `"path: Filesystem path to the source model file"`).

## 10. Testing Strategy

- Every module under `src/uaqe/` has a mirrored test module under `tests/unit/` (per `02_Folder_Structure.md` §14) — 1:1 file correspondence enforced by CI (a source file without a matching test file fails the build).
- `pytest` is the sole test runner; `pytest-cov` enforces a minimum 85% line coverage on `uaqe.domain` and `uaqe.application`, 70% on `uaqe.infrastructure` (lower bar due to hardware-dependent code paths that require mocked I/O).
- All `IFrameworkAdapter`, `IExporterBackend`, `IQuantizationStrategy`, `ICompressionStrategy`, and `IReportRenderer` implementations MUST have a contract test that runs the same test suite against every concrete implementation of that interface (shared parametrized test base), guaranteeing interchangeability.
- Integration tests (`tests/integration/pipeline_end_to_end/`) run the full pipeline against `tests/fixtures/sample_models/` for at least one model per supported input format and one hardware profile per `HardwareClass`.

## 11. Git Workflow

- Trunk-based development on `main`; all work happens on short-lived feature branches merged via pull request. No direct commits to `main`.
- `main` is always deployable; CI (lint, type-check, tests, import-linter) MUST pass before merge.

## 12. Branch Strategy

- Branch naming: `<type>/<module>-<short-description>`, e.g. `feat/quantization-sensitivity-analyzer`, `fix/exporter-artix7-overflow`, `docs/config-spec-update`.
- `<type>` ∈ `feat`, `fix`, `docs`, `refactor`, `test`, `chore` — matches commit type below.
- One module (per `02_Folder_Structure.md` ownership) per branch where practical; cross-module branches require sign-off from both module owners.

## 13. Commit Naming Convention

Conventional Commits format: `<type>(<scope>): <description>`

```
feat(quantization): add per-layer sensitivity threshold override
fix(exporter): guard against artix7 BRAM overflow on export
docs(config): document reports.json output_format field
refactor(pipeline): extract stage resolution into PipelineBuilder
test(compression): add contract tests for ICompressionStrategy
chore(deps): bump onnx adapter dependency pin
```

`<scope>` matches a `02_Folder_Structure.md` module name (`quantization`, `exporter`, `pipeline`, `config`, `compression`, etc.).

## 14. Code Review Rules

1. No PR merges without at least one approval from a reviewer who is not the module's primary owner (cross-pollination requirement).
2. Any PR touching a file listed in `09_Architecture_Lock.md` requires two approvals plus an explicit RFC reference in the PR description.
3. Reviewers check three things every time, in order: (a) does this match the signature in `03_API_Specification.md`, (b) does this violate any Data Flow Rule in `04_Data_Flow.md`, (c) is test coverage adequate per §10 above. Style nits are secondary to these three.
4. No PR may introduce a new top-level dependency without updating `pyproject.toml` and stating the justification in the PR description; dependencies must be scoped to the correct layer (e.g. `torch` may only appear in `infrastructure/framework_adapters/torch_adapter.py`'s dependency group, never in `uaqe.domain`).

---

**End of `07_Coding_Standards.md`.**
