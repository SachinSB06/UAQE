# UAQE GitHub Readiness & Repository Audit Report

**Universal AI Quantization Engine (UAQE)**  
**Audit Date**: 2026-09-09  
**Execution Host**: Windows AMD64 (Python 3.13.7, Node v24.13.0)  
**Status**: **GITHUB READY — VERIFIED**

---

## A. Original Project Size
- **Total Workspace Size**: **33,814.19 MB (33.02 GB)**
- **Total Files**: **151,096 files** across 10 top-level directories.
- **Primary Bloat Drivers**:
  - `archive/`: 17,466.54 MB (historical experiment runs, previous pipeline caches)
  - `output/`: 13,689.92 MB (runtime job executions, uploaded datasets, local candidates)
  - `src/output/`: 2,239.87 MB (intermediate execution outputs generated within `src`)
  - `frontend/node_modules/`: 177.51 MB (third-party Node modules)
  - `scratch/`: 31.01 MB (temporary test outputs)

---

## B. Final Source Repository Size
- **Expected Tracked Size**: **158.63 MB (166,330,578 bytes)**
- **Total Space Reduction**: **32.87 GB (99.5% reduction)**
- **Tracked Files**: Source code, core configuration, test suites, documentation, lightweight calibration sets, and verified demonstration models.

---

## C. Files & Directories Kept
1. **Core Source (`src/`)**: Complete `uaqe` Python package (analyzers, optimizers, compression packagers, runtime benchmarks, telemetry monitors, hardware target registry, FastAPI server).
2. **Web Frontend (`frontend/`)**: React 19 + TypeScript + Vite dashboard, components, charts, and test specs.
3. **Tests (`tests/` & `src/uaqe/tests/`)**: Authoritative regression suites covering candidate metric identity, latency provenance, candidate switching, artifact downloads, and XNNPACK integration.
4. **Verified Models (`models/`)**: Two verified demonstration INT8 models (`1.85 MB` each) with `models/README.md`.
5. **Datasets (`datasets/`)**: Lightweight SEM 9-class calibration set (`47 MB`), hackathon test dataset, and `datasets/README.md`.
6. **Launchers & CLI**: `start_uaqe.bat`, `start_uaqe.ps1`, `uaqe.py`, `pytest.py`, `pytest.bat`, `pytest.ps1`.
7. **Documentation (`docs/` & root)**: `README.md`, `LICENSE`, `docs/project_structure.md`.
8. **Configuration & CI**: `requirements.txt`, `requirements-runtime.txt`, `.gitignore`, `.github/workflows/tests.yml`.

---

## D. Files & Directories Excluded from Git
1. `archive/` (17.06 GB) — Historical build logs and experiment scratch files.
2. `output/` (13.37 GB) — Generated optimization jobs, candidate artifacts, and upload caches.
3. `src/output/`, `src/outputs/`, `src/logs/`, `src/reports/` (2.23 GB) — Legacy internal script outputs.
4. `scratch/` (31 MB) — Active test scratch directories.
5. `frontend/node_modules/` (177 MB) & `frontend/dist/` — Node packages and compiled build assets.
6. `__pycache__/`, `*.pyc`, `.pytest_cache/` — Python bytecode and runner caches.
7. Large raw datasets (`datasets/imagenet*`, raw archives).

---

## E. Files Placed
- `models/mobilenetv3_sem_9class_qat_int8.tflite`: Reference QAT INT8 model (SHA256 `10d51d4f...`).
- `models/mobilenetv3_sem_9class_xnnpack_int8.tflite`: XNNPACK-compatible INT8 model (SHA256 `93a54e8f...`).
- `models/README.md`: Model format support and checkpoint policy guide.
- `datasets/README.md`: Dataset layout, adapter guide, and calibration registry.
- `start_uaqe.ps1`: Native PowerShell launcher.
- `requirements.txt`: Unified dependencies specification.
- `LICENSE`: Apache License 2.0.
- `.github/workflows/tests.yml`: Automated CI pipeline.

---

## F. Security Audit Findings
- **Potential Secrets / API Keys**: **0 found** across entire repository.
- **Private Keys / Certificates**: **0 found**.
- **Credentials / `.env` files**: **0 found**.
- **Result**: **PASS** (Zero credentials or security vulnerabilities detected).

---

## G. Hardcoded Path Audit Findings
- **`README.md`**: Removed hardcoded `cd "d:/Quantization embedded"` in favor of generic repository cloning.
- **`docs/project_structure.md`**: Replaced absolute machine paths with repository-relative hierarchy.
- **`datasets/calibration/calibration_manifest.json`**: Changed `calibration_root` from `D:/Quantization embedded/...` to `datasets/calibration/dataset`.
- **`src/uaqe/tests/test_no_mock_optimization.py`** & **`src/uaqe/tests/test_vit_large_imagenet_pipeline.py`**: Replaced fixed `D:\imagenet_10k_split` with `os.environ.get("IMAGENET_SPLIT_PATH", ...)` with automated `skipTest` protection.
- **`start_uaqe.bat`** & **`start_uaqe.ps1`**: Dynamically resolve repository root via `%~dp0` and `$PSScriptRoot`.

---

## H. Dependency Audit Findings
- Generated unified `requirements.txt` covering runtime, server, and test suites.
- Preserved minimal runtime specification `requirements-runtime.txt`.
- Frontend `package.json` verified: Vite 8, React 19, TypeScript 6.

---

## I. Startup File Audit
- **`start_uaqe.bat`**: Preserved and hardened:
  1. Resolves repository root dynamically via `%~dp0`.
  2. Discovers Python interpreter (`.venv` -> `venv` -> system `python`).
  3. Verifies Python availability and imports (`fastapi`, `uvicorn`).
  4. Verifies `npm` availability and runs `npm install` in `frontend/` if `node_modules` is missing.
  5. Automatically recreates missing runtime output directories (`output\jobs`, `output\uploads\models`, `output\uploads\datasets`, `scratch`).
  6. Launches backend on `http://127.0.0.1:8000` and frontend on `http://localhost:3000`.
  7. Prints URLs and clear shutdown guidance.
- **`start_uaqe.ps1`**: Created matching native PowerShell script using `$PSScriptRoot`.

---

## J. Startup Execution Results
- **Execution from Repo Root**: **PASS** (Backend and Frontend both started and responded HTTP 200).
- **Execution from Outside Working Directory**: **PASS** (Tested from Windows Temp directory; root resolved and backend responded HTTP 200).
- **Automatic Runtime Directory Recreation**: **PASS** (Directories removed/renamed were automatically recreated upon initialization).
- **Clean Shutdown**: **PASS** (Processes cleanly terminated).
- **Restart Verification**: **PASS** (Server restarted and responded HTTP 200 on port 8000).

---

## K. Backend Health
- `GET http://127.0.0.1:8000/docs`: **HTTP 200 OK**
- `GET http://127.0.0.1:8000/api/status`: **HTTP 200 OK** (`status: READY, version: 1.0.0`)

---

## L. Frontend Health
- `GET http://localhost:3000/`: **HTTP 200 OK** (HTML bundle loaded with `<title>UAQE</title>`).

---

## M. Candidate-Selection Bug Fix (Section 20)
- **Data-Flow Fix**:
  - In `ResultsPage.tsx`, replaced winner fallback `optimized={telemetry?.final}` with candidate-specific `optimized={activeCandidateTelemetry}`.
  - In `ResourceComparisonCard.tsx`, prohibited winner substitution when evaluating an active candidate. Missing values display `NOT AVAILABLE` / `—` rather than another candidate's metrics.
  - In `candidateMetrics.ts`, removed hardcoded operator counts (`204`, `0`).

---

## N. Artifact-Download Bug Fix (Section 21)
- **Data-Flow & API Fix**:
  - Candidate identity is strictly bound to `job_id + candidate_id + canonical artifact metadata`.
  - Artifact download URL routes to `GET /api/jobs/{job_id}/candidates/{candidate_id}/download`, ensuring Candidate 1 downloads `cand_001` artifact and Candidate 4 downloads `cand_004` artifact.
  - Non-existent universal `.onnx` request returns clean **HTTP 404** without fabricating mock bytes.
  - Strict path traversal guards block `..` and directory manipulation.

---

## O. Test Results
- **Candidate Selection & Download Regression Suite (10 Tests)**: **PASS** (`tests/test_candidate_selection_and_download.py` — 0.615s).
- **Candidate Latency Provenance Suite (15 Tests)**: **PASS** (`tests/test_candidate_latency_provenance.py`).
- **Candidate Metric Identity Suite (24 Tests)**: **PASS** (`tests/test_candidate_metric_identity.py`).
- **Frontend Candidate Normalization Tests (10 Forms)**: **PASS** (`frontend/src/tests/test_candidate_normalization.mjs`).
- **Frontend Candidate Metrics Selector Tests**: **PASS** (`frontend/src/tests/test_candidate_metrics.mjs`).
- **Frontend Linter (`oxlint`)**: **PASS** (0 errors).

---

## P. Frontend Build
- **`npm run build`**: **PASS** (Completed in 7.02s with TypeScript compilation and Vite minification; generated `frontend/dist`).

---

## Q. Large File Audit
- **Files > 100 MB in Tracked Set**: **0 files** (Zero files exceed GitHub's hard limit).
- **Files > 50 MB in Tracked Set**: **1 file**:
  - `src/models/resnet50/model.safetensors` (**97.74 MB**).
  - *Note*: This baseline model is required by the ResNet regression tests. It is within GitHub's 100 MB hard limit. If preferred, it can be migrated to Git LFS or external download script prior to pushing.

---

## R. Model Artifact Decisions
- Small verified demonstration models (`~1.85 MB` each) are preserved in `models/` with exact SHA-256 hashes matching test assertions:
  - Reference: `10d51d4f243c554734e7347f9010297fb182a64c4ad89d5551e24830846dab1d`
  - XNNPACK: `93a54e8f1408b62c41fe595e4a88b7d6ac2dbf9b368569bc506fb5f1f5da78fd`
- Large intermediate training checkpoints and multi-gigabyte files are excluded from Git.

---

## S. Dataset Decisions
- Small representative calibration sets (`~50 MB`) are preserved to allow immediate offline functionality.
- Multi-gigabyte ImageNet datasets are excluded via `.gitignore` and documented in `datasets/README.md`.

---

## T. Remaining Limitations & Honest Disclaimers
1. **Hardware Acceleration**: INT8 reduces footprint by ~4×. Latency speedup over FP32 is dependent on CPU vector extensions (AVX-512 VNNI or ARM NEON DotProd) and is not unconditionally faster across all CPU architectures.
2. **Vision Transformers**: ViT attention operations are recognized by probers but guarded as unsupported for TFLite edge micro-kernels.
3. **Hardware Measurements**: Telemetry and timings reflect Host CPU (AMD64) execution; physical on-device Raspberry Pi 5 measurements remain marked pending physical bench runs.

---

## U. Final GitHub Push Checklist

Before executing `git push`:

1. **Verify Git Status**:
   ```bash
   git status
   ```
2. **Ensure Virtual Environment & Output are Ignored**:
   Confirm that `output/`, `scratch/`, `archive/`, and `node_modules/` do not appear in untracked files.
3. **Stage Changes**:
   ```bash
   git add .
   ```
4. **Commit**:
   ```bash
   git commit -m "feat: Initial public release of UAQE (Universal AI Quantization Engine)"
   ```
5. **Set Main Branch and Add Remote**:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-org-or-username>/UAQE.git
   ```
6. **Push to GitHub**:
   ```bash
   git push -u origin main
   ```
