# UAQE FULL PROJECT FORENSIC AUDIT REPORT

**Universal AI Quantization Engine (UAQE)**  
**Audit Date:** September 5, 2026  
**Auditor:** Principal Software Architect, ML Systems Engineer & DevOps Lead  
**Audit Mode:** Forensic Full-Stack Inspection & Non-Destructive Verification  

---

## 1. Executive Summary

A comprehensive forensic audit of the entire Universal AI Quantization Engine (UAQE) codebase was conducted to determine why the system ceased operating reliably following recent frontend, API, and upload staging architectural changes.

### Key Audit Findings:
1. **The Optimization Engine Core is 100% INTACT and FUNCTIONAL**: All 267 backend automated tests pass completely in 120.157 seconds. Model ingestion, CIFAR-10 / ImageFolder dataset parsing, task detection, compatibility checking, preprocessing resolution, model adaptation, autonomous multi-candidate search, ONNX QDQ static INT8 PTQ quantization, structured pruning, accuracy safety gating (EXCELLENT $\le$ 1.0 pp, ACCEPTABLE $\le$ 4.0 pp, CRITICAL $>$ 4.0 pp), and deployable artifact packaging remain fully operative.
2. **The Root Cause of Frontend Breakdown is a Missing SSE Endpoint**: In `src/uaqe/api/routes_jobs.py`, the backend advertises the Server-Sent Events stream URL as `/api/jobs/{job_id}/events` upon launching an optimization job. However, **no `@router.get("/jobs/{job_id}/events")` route handler was ever implemented** in FastAPI.
3. **The First Broken Boundary**: When a user clicks "Approve Plan & Launch" or "Direct Optimize" in the browser UI, the frontend successfully issues `POST /api/jobs/optimize`, receives a valid `job_id`, and opens `new EventSource('/api/jobs/<job_id>/events')`. The backend immediately responds with **HTTP 404 Not Found**. The frontend's `EventSource.onerror` handler fires immediately, setting `isStreaming = false` while `jobDetail` is `null`, causing the `AutonomousCockpitPage` to immediately collapse into an empty `"No Optimization Job Active"` state.
4. **Premature Hardcoded Status in API**: `GET /api/jobs/{job_id}` hardcodes `status="COMPLETED"` and `verdict="VERIFIED"` even when an optimization is actively running in background or has failed, causing polling clients to prematurely read empty metrics.
5. **Analyze API Contract Mismatch**: `POST /api/jobs/analyze` omits the `optimization_plan` payload expected by `App.tsx`, causing `InspectionPlanPage` to receive `null` and fall back to static text with outdated safety policy thresholds.
6. **Artifact Download Filename Divergence**: The Results Page download buttons request `model_int8.onnx` and `manifest.json`, whereas the optimizer packages files as `optimized_model.onnx` and `input_manifest.json`, resulting in HTTP 404 on download clicks.

**Overall System Verdict: DEGRADED (API/SSE Integration Boundary Broken, Core Optimization Engine Intact)**

---

## 2. Current System Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                      BROWSER FRONTEND (React 19)                                   |
|   DashboardPage | NewOptimizationPage | InspectionPlanPage | AutonomousCockpit | ResultsPage | JobsPage|
+----------------------------------------------------------------------------------------------------+
                                      |                     ^
            Multipart File/Folder     |                     | SSE Telemetry Stream
            & JSON API Calls          v                     | [CRITICAL GAP: Missing Route]
+----------------------------------------------------------------------------------------------------+
|                                    FASTAPI BACKEND (Uvicorn / Port 8000)                           |
|   routes_uploads.py: /api/uploads/model, /api/uploads/dataset, /api/uploads/samples                 |
|   routes_jobs.py:    /api/status, /api/jobs, /api/jobs/{id}, /api/jobs/analyze, /api/jobs/optimize |
+----------------------------------------------------------------------------------------------------+
            |                                           |
            v Upload Staging                            v BackgroundTasks Worker
+------------------------------------+   +-----------------------------------------------------------+
|    UPLOAD STAGING (output/uploads) |   |        OPTIMIZATION ORCHESTRATOR & CONTROLLER             |
|  models/<id>/ & datasets/<id>/     |   |  - UniversalModelIngestor / UniversalDatasetIngestor      |
|  - Cryptographic Manifest Hashing  |   |  - TaskDetector / CompatibilityChecker / Preprocessing    |
|  - Path Traversal Security Checks  |   |  - ModelAdaptationService / OptimizationPlanner           |
+------------------------------------+   |  - CandidateGenerator / CandidateEvaluator (ONNX QDQ)     |
            |                            |  - AccuracySafetyPolicy (<=1.0pp, <=4.0pp, >4.0pp)        |
            v Job Isolation              |  - StoppingPolicy / SearchManager / RuntimeMonitor        |
+------------------------------------+   +-----------------------------------------------------------+
|    JOB ISOLATION (output/jobs/<id>)|                                   |
|  - inputs/model/ & inputs/dataset/ |<----------------------------------+
|  - metrics.json, report.md, final/ | Writes isolated manifests, metrics, telemetry & packages
+------------------------------------+
```

---

## 3. Startup Health

| Check | Expected | Actual Evidence | Status |
|---|---|---|---|
| Backend Server Startup | `python -m uaqe.server --port 8000` | FastAPI server initializes, routes mount, uvicorn binds to `127.0.0.1:8000` | ✅ HEALTHY |
| Frontend Dev Server | `npm run dev` (Vite) | Vite v8.2.2 binds to `localhost:3000`, proxies `/api` to `http://127.0.0.1:8000` | ✅ HEALTHY |
| Frontend Production Build | `npm run build` | `tsc -b && vite build` succeeds; output generated in `frontend/dist/` in 6.82s | ✅ HEALTHY |
| Static Frontend Hosting | Backend mounts `dist/` | `server.py` line 44-46 mounts `frontend/dist` at `/` using `StaticFiles` | ✅ HEALTHY |
| CORS Settings | Allow localhost dev ports | `server.py` lines 31-37 configures `CORSMiddleware` with `allow_origins=["*"]` | ✅ HEALTHY |
| Python Runtime & Dependencies | Python 3.13, PyTorch 2.13, ONNX 1.22, ONNXRuntime 1.28, FastAPI 0.141 | Pip inspection confirms all required runtime ML & Web libraries installed | ✅ HEALTHY |
| Startup Exceptions | Zero import errors | Zero import errors during module discovery and server initialization | ✅ HEALTHY |

---

## 4. Frontend Health

- **Framework**: React 19 + TypeScript + Vite + TailwindCSS.
- **State Flow**: Centralized in `App.tsx`, providing single-source-of-truth state management for active jobs, navigation, pre-inspection, live telemetry, and results replay.
- **Zero-Path UI Compliance**: `NewOptimizationPage.tsx` uses browser drag-and-drop / file browser for single model upload and folder selection with relative path preservation. Users do not need to manually enter raw filesystem paths.
- **Identified Issues in Frontend**:
  1. `App.tsx` handles SSE `onError` by immediately setting `isStreaming(false)`. Since the backend SSE route returns 404, `AutonomousCockpitPage` immediately unmounts active tracking and shows "No Optimization Job Active".
  2. `InspectionPlanPage.tsx` contains static fallback text with outdated policy thresholds (`Loss <= 0.50 pp` vs canonical `Loss <= 1.0 pp`).
  3. `ResultsPage.tsx` hardcodes baseline props (`75.0`, `97.71`, `82.46`) to scatter charts instead of passing dynamic `fp32Acc`, `fp32SizeMB`, and `fp32Latency`.
  4. `ResultsPage.tsx` download buttons request `model_int8.onnx` and `manifest.json` instead of `optimized_model.onnx` and `input_manifest.json`.

---

## 5. Backend / API Health

### Complete Frontend-to-Backend API Contract Audit Table

| Frontend Call (`services/api.ts`) | HTTP Method | Request Payload | Expected Backend Route | Actual Backend Route | Backend Request Schema | Backend Response Schema | Status & Discrepancies |
|---|---|---|---|---|---|---|---|
| `fetchStatus()` | `GET` | None | `/api/status` | `/api/status` | None | `SystemStatusResponse` | ✅ MATCH |
| `fetchJobs()` | `GET` | None | `/api/jobs` | `/api/jobs` | None | `List[JobSummaryResponse]` | ⚠️ Hardcodes `status="COMPLETED"` |
| `fetchJobDetail(id)` | `GET` | None | `/api/jobs/{id}` | `/api/jobs/{id}` | None | `JobDetailResponse` | ⚠️ Hardcodes `status="COMPLETED"`, `verdict="VERIFIED"` |
| `fetchJobTelemetry(id)` | `GET` | None | `/api/jobs/{id}/telemetry` | `/api/jobs/{id}/telemetry` | None | `TelemetryResponse` | ✅ MATCH |
| `fetchSamples()` | `GET` | None | `/api/uploads/samples` | `/api/uploads/samples` | None | `{models, datasets}` | ✅ MATCH |
| `uploadModelFile(file)` | `POST` | `multipart/form-data` (`file`) | `/api/uploads/model` | `/api/uploads/model` | `UploadFile` | `ModelUploadResponse` | ✅ MATCH |
| `uploadDatasetFolder(files, paths)` | `POST` | `multipart/form-data` (`files`, `relative_paths`) | `/api/uploads/dataset` | `/api/uploads/dataset` | `List[UploadFile]`, `relative_paths: Form` | `DatasetUploadResponse` | ✅ MATCH |
| `uploadDatasetZip(file)` | `POST` | `multipart/form-data` (`file`) | `/api/uploads/dataset/zip` | `/api/uploads/dataset/zip` | `UploadFile` | `DatasetUploadResponse` | ✅ MATCH |
| `analyzeModelAndDataset(...)` | `POST` | JSON `{model_upload_id, dataset_upload_id, ...}` | `/api/jobs/analyze` | `/api/jobs/analyze` | `AnalyzeRequest` | `{model_insp, dataset_insp, compat}` | ❌ MISMATCH: Missing `optimization_plan` in response |
| `planOptimization(...)` | `POST` | JSON `{model_upload_id, dataset_upload_id, ...}` | `/api/jobs/plan` | `/api/jobs/plan` | `PlanRequest` | `PlanResultDict` | ✅ MATCH |
| `startOptimization(...)` | `POST` | JSON `{model_upload_id, dataset_upload_id, target, profile, ...}` | `/api/jobs/optimize` | `/api/jobs/optimize` | `OptimizeRequest` | `{job_id, status: "LAUNCHED", events_url, ...}` | ✅ MATCH |
| `subscribeToJobEvents(id)` | `GET` (SSE) | None | `/api/jobs/{id}/events` | **MISSING** | N/A | `text/event-stream` | ❌ **CRITICAL P0 FAILURE**: Route missing (HTTP 404) |
| `getArtifactDownloadUrl(id, file)` | `GET` | None | `/api/jobs/{id}/download/{file}` | `/api/jobs/{id}/download/{file}` | Path params | `FileResponse` | ⚠️ Filename mismatch (`model_int8.onnx` vs `optimized_model.onnx`) |

---

## 6. Upload / Ingestion Health

- **Model Upload (`POST /api/uploads/model`)**:
  - File extension validation against whitelist (`.pt`, `.pth`, `.onnx`, `.safetensors`, `.tflite`, `.bin`).
  - Staging directory: `output/uploads/models/<upload_id>/`.
  - Computes cryptographic SHA-256 chunk-by-chunk.
  - Generates `model_meta.json` descriptor and stores in in-memory registry `STAGED_MODELS`.
- **Dataset Folder Upload (`POST /api/uploads/dataset`)**:
  - Preserves nested relative path directory structure.
  - Strict directory traversal security validation via `_sanitize_relative_path` (rejects `..`, absolute paths, drive letters, UNC paths, and null bytes).
  - Staging directory: `output/uploads/datasets/<upload_id>/`.
  - Generates `dataset_manifest.json` containing per-file relative paths, sizes, and SHA-256 hashes, along with an overall manifest SHA-256 digest.
  - Automatically invokes `UniversalDatasetIngestor` to recognize dataset format (e.g., `cifar10_pickle`, `image_folder`, `csv_dataset`), class counts, and split partitions.
- **Dataset Zip Upload (`POST /api/uploads/dataset/zip`)**:
  - Fallback endpoint for compressed archives; extracts with zip-slip prevention checks.

---

## 7. Job Isolation Health

Job isolation is strictly implemented and verified:
1. **Unique Identifiers**: Each optimization launch generates a unique ID formatted as `UAQE-<YYYYMMDD>-<SHORT_UUID>`.
2. **Directory Isolation**: All inputs and outputs are isolated in dedicated directories under `output/jobs/<job_id>/`:
   - `output/jobs/<job_id>/inputs/model/`
   - `output/jobs/<job_id>/inputs/dataset/`
   - `output/jobs/<job_id>/candidates/<cand_id>/`
   - `output/jobs/<job_id>/final/`
3. **Multi-Job Separation**: Ingested model and dataset files are copied from upload staging directly into the job's private `inputs/` folder. Job A cannot mutate or contaminate Job B.
4. **Manifest Provenance**: `input_manifest.json` records cryptographic hashes (`model_sha256`, `dataset_manifest_hash`) and source tags (`USER_UPLOAD`, `PRE_VERIFIED_SAMPLE`, `CLI_PATH`).

---

## 8. Orchestration Health

- **File**: `src/uaqe/orchestration/optimization_orchestrator.py`
- **Class**: `OptimizationOrchestrator`
- **Pipeline Stages**:
  1. `UniversalModelIngestor.inspect()`: Extracts tensor shapes, parameter counts, framework, architecture.
  2. `UniversalDatasetIngestor.load()`: Discovers training, validation, and test splits.
  3. `TaskDetector.detect_task()`: Identifies task family (e.g., `image_classification`).
  4. `CompatibilityChecker.check_compatibility()`: Evaluates model input shape vs dataset resolution and output classes vs label taxonomy.
  5. `PreprocessingResolver.resolve()`: Generates normalization and channel-ordering configurations.
  6. `ModelAdaptationService.adapt()`: Adapts classifier heads if class dimensions differ.
  7. `HardwareTargetRegistry.get_profile()`: Resolves hardware execution profiles (e.g., `raspberrypi5`, `jetson_orin_nano`, `stm32`, `host_cpu`).
  8. `OptimizationPlanner.create_plan()`: Computes multi-objective weights and search budgets.
  9. `OptimizationController.optimize()`: Executes autonomous search loop.

---

## 9. Optimization Engine Health

- **File**: `src/uaqe/optimization/optimization_controller.py`
- **Integrity**: The core optimization engine is **fully functional**.
- **Empirical Baseline**: `establish_fp32_baseline()` evaluates the unquantized FP32 reference model on frozen test data, collecting Top-1 accuracy, Macro F1, baseline file footprint, and hardware CPU latency/throughput with `RuntimeMonitor`.
- **Strategy Resolution**: Capability-driven selection in `OptimizationStrategyResolver`:
  - `ResNetONNXPTQStrategy`: Full static QDQ INT8 with selective layer protection (stem conv / classifier head).
  - `MobileNetAdaptiveStrategy`: TFLite INT8 post-training quantization.
  - `GenericFallbackStrategy`: Universal fallback quantizer.
- **Candidate Evaluator**: `CandidateEvaluator.evaluate()` performs real calibration on the training split, executes ONNX static quantization with MinMax calibrator, audits graph quantization nodes, and evaluates top-1 accuracy on test data.

---

## 10. Autonomous Controller Health

### Accuracy Safety Policy Rules (Canonical Standard):
- **EXCELLENT**: $\text{Accuracy Loss} \le 1.0\text{ pp}$
- **ACCEPTABLE**: $1.0\text{ pp} < \text{Accuracy Loss} \le 4.0\text{ pp}$
- **CRITICAL**: $\text{Accuracy Loss} > 4.0\text{ pp}$

### Safety Enforcement Verification:
- When a candidate triggers a `CRITICAL` loss (e.g., standard uniform ResNet-50 INT8 PTQ which experiences a $5.10\text{ pp}$ accuracy drop), the search controller **strictly rejects** the candidate, records the rejection rationale, and autonomously generates safer recovery candidates (such as protecting stem convolutions and classification heads in FP32).
- `CRITICAL` candidates are prevented from becoming the final selected deployable package when a valid candidate exists.
- Candidate search budget defaults to 10 candidates with a hard upper bound of 20 candidates.

---

## 11. SSE / Telemetry Health

### Identified Breakdowns:
1. **Missing SSE Route (P0)**: In `src/uaqe/api/routes_jobs.py`, line 671 returns `"events_url": f"/api/jobs/{job_id}/events"`, but there is no `@router.get("/jobs/{job_id}/events")` FastAPI endpoint.
2. **Thread Safety Issue (P1)**: `_run_optimization_worker` executes within FastAPI's `BackgroundTasks` thread pool and directly calls `_broadcast_event` -> `queue.put_nowait(event)`. Manipulating `asyncio.Queue` from another thread without `loop.call_soon_threadsafe` can cause dropped events or event loop deadlocks in Python 3.13.
3. **No Active Heartbeat**: Event stream lacked periodic keepalive pings (`: keepalive\n\n`) to prevent browser socket timeout during long quantization steps.

---

## 12. Results & Artifact Health

- **Persistence**: Upon completion, `OptimizationController._package_final_candidate` writes:
  - `metrics.json`: Final accuracy, accuracy delta, safety tier, file sizes, latency, throughput, prediction agreement.
  - `report.md`: Markdown executive summary with GitHub alerts.
  - `predictions.csv`: Per-sample ground truth vs prediction verification.
  - `telemetry.json`: CPU and RAM time-series metrics.
  - `optimization_history.json`: Candidate evaluation sequence.
  - `pareto_frontier.json`: Evaluated Pareto frontier data points.
  - `optimized_model.onnx`: Standalone deployable quantized model.
- **Artifact Downloading**: `GET /api/jobs/{job_id}/download/{filename}` validates path containment within the job directory to prevent directory traversal.

---

## 13. Security Findings

| Area | Audit Check | Finding | Risk Level |
|---|---|---|---|
| Upload Directory Traversal | `_sanitize_relative_path` in `routes_uploads.py` | Traversal sequences (`..`, `../`, `..\`), absolute paths, drive letters, and null bytes are rejected. | ✅ SECURE |
| Zip Slip Vulnerability | `upload_dataset_zip` in `routes_uploads.py` | Archive extraction validates that every member destination starts with `real_staging_dir`. | ✅ SECURE |
| Artifact Download Escape | `download_job_artifact` in `routes_jobs.py` | Uses `os.path.basename` and verifies `target_path.startswith(job_dir)`. | ✅ SECURE |
| Uploaded Model Execution | Extension validation whitelist | Allowed extensions restricted to `.pt`, `.pth`, `.onnx`, `.safetensors`, `.tflite`, `.bin`. Executables rejected. | ✅ SECURE |
| Zero-Path Exposure | API response schemas | `staged_path` is returned in upload responses. While non-fatal, server filesystem layout is leaked to browser. | ⚠️ P3 (Minor) |

---

## 14. Test Results

### 1. Automated Python Backend Test Suite (`unittest discover -s src/uaqe/tests`)
- **Total Tests Discovered**: 267
- **Passed**: 267
- **Failed**: 0
- **Errors**: 0
- **Execution Duration**: 120.157 seconds
- **Test Areas Covered**:
  - `test_autonomous_optimization.py`: 21/21 assertions passed (safety gating, recovery search, stopping policy, candidate deduplication).
  - `test_upload_workflow.py`: Model & dataset upload, security path traversal rejection, manifest hashing.
  - `test_phase_r1_resnet50_int8_ptq.py`: ResNet-50 INT8 PTQ quantization and evaluation.
  - `test_phase_e3_universal_orchestration.py`: End-to-end orchestration flow.
  - `test_phase_d1_pruning.py` through `test_phase_d5_runtime.py`: Pruning, compression, clustering, runtime engine.
  - `test_accuracy_ablation.py`: Calibration set size vs accuracy degradation.

### 2. Standalone Smoke Test (`uaqe_quantization_smoke_test.py`)
- **Scenarios Evaluated**: 8 (Uniform INT8, INT4, Mixed Precision, Hardware Overrides, JSONL Calibration, LayerQuantizer roundtrip, Invalid Override Rejection, Empty IMR).
- **Result**: `TOTAL BUGS FOUND: 0` (All 8 passed).

### 3. Frontend TypeScript & Bundler Build (`npm run build`)
- **TypeScript Check**: `tsc -b` exited with code 0.
- **Vite Production Build**: Successfully compiled 2,432 modules into `frontend/dist/` in 6.82s.

---

## 15. Regression Analysis

Comparing the system before and after frontend integration:
1. **The Core Engine Did Not Regress**: Algorithms, quantizers, evaluators, and orchestrators remain fully intact.
2. **Integration Regression**: The new asynchronous worker model was wired up to dispatch SSE events via `_broadcast_event`, but the corresponding HTTP route `@router.get("/jobs/{job_id}/events")` was omitted from `routes_jobs.py`.
3. **UI Contract Drift**: Frontend components were developed with slight property and filename assumptions (`optimization_plan` missing in `/jobs/analyze` response; `model_int8.onnx` requested instead of `optimized_model.onnx`).

---

## 16. Root Cause of Current Failure

The primary root cause of the system failure experienced in the browser is the **missing FastAPI SSE endpoint (`GET /api/jobs/{job_id}/events`) in `src/uaqe/api/routes_jobs.py`**.

When a user initiates optimization:
1. `POST /api/jobs/optimize` successfully starts the job in a background thread and returns `"events_url": "/api/jobs/<job_id>/events"`.
2. Frontend opens `EventSource('/api/jobs/<job_id>/events')`.
3. FastAPI returns **HTTP 404**.
4. Frontend `EventSource.onerror` immediately triggers, setting `isStreaming = false`.
5. `AutonomousCockpitPage` evaluates `(!jobDetail && !isStreaming)` as `true`, immediately replacing the active cockpit with the fallback message: `"No Optimization Job Active"`.
6. The user is prevented from seeing progress, telemetry, or completion transitions.

---

## 17. First Broken Boundary

```
[User clicks Optimize]
       |
       v
Frontend: apiService.launchOptimization() [SUCCESS: HTTP 200, returns job_id]
       |
       v
Frontend: apiService.streamJobEvents() -> new EventSource('/api/jobs/<job_id>/events')
       |
       v
======================= [FIRST BROKEN BOUNDARY] =======================
Backend: GET /api/jobs/<job_id>/events -> HTTP 404 NOT FOUND (Route Missing)
=======================================================================
       |
       v
Frontend: EventSource.onerror fired -> setIsStreaming(false)
       |
       v
Cockpit Page: displays "No Optimization Job Active" (Workspace Appears Dead)
```

---

## 18. Broken Execution Trace for One Full Optimization Job

| Step | Component | File / Function | Input | Output / Effect | Status |
|---|---|---|---|---|---|
| 1 | Model Upload | `NewOptimizationPage.tsx` / `uploadModelFile` | User selects `.pt` file | `POST /api/uploads/model` | ✅ WORKING |
| 2 | Model Staging | `routes_uploads.py` / `upload_model_file` | Multipart binary stream | File staged, SHA-256 computed, returns `upload_id` | ✅ WORKING |
| 3 | Dataset Upload | `NewOptimizationPage.tsx` / `uploadDatasetFolder` | User selects folder | `POST /api/uploads/dataset` | ✅ WORKING |
| 4 | Dataset Staging | `routes_uploads.py` / `upload_dataset_folder` | Multipart files + relative paths | Folder hierarchy staged, manifest hash built | ✅ WORKING |
| 5 | Input Analysis | `routes_jobs.py` / `analyze_model_and_dataset` | Upload IDs | Inspects model, dataset, compatibility | ⚠️ SUSPECT (Omits `optimization_plan`) |
| 6 | Plan Review | `InspectionPlanPage.tsx` | Analysis result | Renders graph parameters and policy | ⚠️ SUSPECT (Displays static fallback text) |
| 7 | Job Launch | `routes_jobs.py` / `start_autonomous_optimization` | `OptimizeRequest` | Creates isolated directory `output/jobs/<id>/inputs/`, launches worker | ✅ WORKING |
| 8 | SSE Stream Connect | `api.ts` / `subscribeToJobEvents` | `job_id` | `new EventSource('/api/jobs/<id>/events')` | ❌ **BROKEN (HTTP 404)** |
| 9 | Cockpit UI | `AutonomousCockpitPage.tsx` | Stream events | `isStreaming` set to `false` on 404 -> UI collapses | ❌ **BROKEN (Blocked by Step 8)** |
| 10 | Orchestrator Run | `optimization_orchestrator.py` / `run` | `job_config` | Ingests, checks compatibility, builds plan | ✅ WORKING |
| 11 | FP32 Baseline | `optimization_controller.py` / `establish_fp32_baseline` | Adapted model + test split | Measures reference accuracy (75.0%) & latency (82.46ms) | ✅ WORKING |
| 12 | Candidate Search | `optimization_controller.py` / `optimize` | Multi-candidate loop | Evaluates candidates, enforces safety gating | ✅ WORKING |
| 13 | Final Packaging | `optimization_controller.py` / `_package_final_candidate` | Winning candidate | Outputs `optimized_model.onnx`, `metrics.json`, `report.md` | ✅ WORKING |
| 14 | Job Detail API | `routes_jobs.py` / `get_job_detail` | `job_id` | Reads outputs from `output/jobs/<id>/` | ⚠️ SUSPECT (Hardcodes status "COMPLETED") |
| 15 | Artifact Download | `ResultsPage.tsx` / `onDownloadArtifact` | `filename` | Requests `model_int8.onnx` | ❌ **BROKEN (404: file is `optimized_model.onnx`)** |

---

## 19. Working Components

1. ✅ **Master CLI (`uaqe.py`)**: `plan`, `optimize`, `inspect-model`, `inspect-dataset`, `validate` subcommands work flawlessly.
2. ✅ **Model Ingestor (`UniversalModelIngestor`)**: PyTorch, ONNX, and SafeTensors parsing.
3. ✅ **Dataset Ingestor (`UniversalDatasetIngestor`)**: CIFAR-10 pickle batches, ImageFolder hierarchy, and CSV dataset ingestion.
4. ✅ **Task Detector (`TaskDetector`)**: Automated detection of classification tasks.
5. ✅ **Compatibility Checker (`CompatibilityChecker`)**: Shape, channel count, and class taxonomy alignment.
6. ✅ **Preprocessing Resolver (`PreprocessingResolver`)**: Mean/std normalization configuration.
7. ✅ **Hardware Target Registry (`HardwareTargetRegistry`)**: Profiles for Raspberry Pi 5, Jetson Orin Nano, STM32, and Host CPU.
8. ✅ **Autonomous Optimization Controller (`OptimizationController`)**: Progressive candidate generation, Pareto frontier tracking, and early stopping.
9. ✅ **Accuracy Safety Policy (`AccuracySafetyPolicy`)**: Accurate mathematical loss computation and tier classification (EXCELLENT $\le$ 1.0 pp, ACCEPTABLE $\le$ 4.0 pp, CRITICAL $>$ 4.0 pp).
10. ✅ **ONNX Static INT8 Quantizer (`R1ResNet50Evaluator` / `ResNet50ONNXExporter`)**: MinMax calibration and per-channel QDQ node generation.
11. ✅ **Upload Pipeline & Security**: Multipart streaming, relative path preservation, and directory traversal rejection.
12. ✅ **Multi-Job Isolation**: Fully isolated job workspaces under `output/jobs/<job_id>/`.
13. ✅ **Telemetry Engine (`RuntimeMonitor` / `TelemetrySession`)**: Real process and system CPU/RAM monitoring via `psutil`.

---

## 20. Broken Components

1. ❌ **FastAPI SSE Endpoint (`src/uaqe/api/routes_jobs.py`)**: Missing `@router.get("/jobs/{job_id}/events")` route handler (P0).
2. ❌ **Event Broadcasting Thread Safety (`src/uaqe/api/routes_jobs.py`)**: Worker thread directly mutates async event loop queues (P1).
3. ❌ **Job Status Reporting (`src/uaqe/api/routes_jobs.py`)**: `get_job_detail` and `list_jobs` hardcode `status="COMPLETED"` and `verdict="VERIFIED"` (P1).
4. ❌ **Analyze Endpoint Response (`src/uaqe/api/routes_jobs.py`)**: `analyze_model_and_dataset` omits `optimization_plan` (P1).
5. ❌ **Results Page Download Actions (`frontend/src/pages/ResultsPage.tsx`)**: Hardcoded artifact names mismatch output files (P2).
6. ❌ **Results Page Scatter Charts Baseline (`frontend/src/pages/ResultsPage.tsx`)**: Static ResNet constants passed to chart props instead of dynamic job values (P2).
7. ❌ **Inspection Plan Display Thresholds (`frontend/src/pages/InspectionPlanPage.tsx`)**: Displays outdated tier thresholds (P2).

---

## 21. Risk Severity Summary

| Severity | Count | Findings |
|---|---|---|
| **P0 (Critical / Project Unusable)** | 1 | `AUDIT-001` (Missing SSE `/api/jobs/{id}/events` route) |
| **P1 (Major Workflow Broken)** | 3 | `AUDIT-002` (Thread-unsafe SSE dispatch), `AUDIT-003` (Hardcoded status in API), `AUDIT-004` (Missing plan in analyze endpoint) |
| **P2 (Important / Degraded UX)** | 3 | `AUDIT-005` (Artifact download 404s), `AUDIT-006` (Scatter chart baseline props), `AUDIT-007` (Stale policy text in InspectionPlanPage) |
| **P3 (Minor Issue)** | 2 | `AUDIT-008` (Server path in upload schema), `AUDIT-009` (Stray root directory `uaqe/`) |
| **P4 (Cosmetic / Documentation)** | 1 | `AUDIT-010` (Root `config.py` stub) |

---

## 22. Minimal Repair Plan

Repairs should be executed in strict dependency order without refactoring working optimization code:

### Step 1: SSE Streaming Route & Thread-Safe Dispatcher (P0 / P1)
- **File**: `src/uaqe/api/routes_jobs.py`
- Add `@router.get("/jobs/{job_id}/events")` endpoint returning `StreamingResponse(event_generator, media_type="text/event-stream")`.
- Maintain per-job client queues with thread-safe dispatching (`loop.call_soon_threadsafe`) and periodic heartbeat comments (`: ping\n\n`).
- Clean up disconnected client queues on disconnect.

### Step 2: Dynamic Job Execution Status (P1)
- **File**: `src/uaqe/api/routes_jobs.py`
- Introduce active job status tracking (`RUNNING`, `COMPLETED`, `FAILED`).
- Update `get_job_detail` and `list_jobs` to reflect actual job status, returning `RUNNING` while worker is active and `FAILED` with error details if an exception occurred.

### Step 3: Add Optimization Plan to Analyze Response (P1)
- **File**: `src/uaqe/api/routes_jobs.py`
- In `analyze_model_and_dataset`, invoke `OptimizationPlanner.create_plan(...)` and return `"optimization_plan"` in the response dictionary.

### Step 4: Fix Artifact Download Mappings (P2)
- **File**: `frontend/src/pages/ResultsPage.tsx` and `src/uaqe/api/routes_jobs.py`
- Update `ResultsPage.tsx` download buttons to request `optimized_model.onnx` and `input_manifest.json`, or add alias resolution in `download_job_artifact`.

### Step 5: Dynamic Scatter Chart Baseline Props & Policy Text (P2)
- **Files**: `frontend/src/pages/ResultsPage.tsx`, `frontend/src/pages/InspectionPlanPage.tsx`
- Pass dynamic `fp32Acc`, `fp32SizeMB`, and `fp32Latency` to Pareto and Scatter chart components.
- Align `InspectionPlanPage.tsx` threshold text with canonical `AccuracySafetyPolicy` values ($\le 1.0\text{ pp}$, $\le 4.0\text{ pp}$, $> 4.0\text{ pp}$).

---

## 23. Recommended Minimal Repair vs Rewrite

- **No Rewrite Necessary**: The ML optimization pipeline, model adapters, dataset loaders, evaluators, and quantizers are working properly.
- **Minimal Surgical Patch Required**: Modifying fewer than 150 lines across `src/uaqe/api/routes_jobs.py` and `frontend/src/pages/` will fully restore 100% end-to-end functionality.

---

## 24. Verification Plan Post-Repair

1. **Automated Unit Tests**: Execute `python -m unittest discover -s src/uaqe/tests -p "test_*.py"` (must maintain 267/267 passing).
2. **SSE Streaming Test**: Test `GET /api/jobs/{job_id}/events` using an SSE client script to verify real-time event delivery.
3. **End-to-End Two-Job Verification**: Run `python src/uaqe/tests/run_end_to_end_two_jobs.py` against live server to verify real model/dataset upload, execution, and zero cross-job contamination.
4. **Browser UI Walkthrough**: Perform a full browser optimization run from upload to results review and artifact download.

---

## 25. Final Audit Verdict

# **DEGRADED (API/SSE Integration Boundary Broken, Core Optimization Engine Intact)**

The UAQE core optimization engine is fully functional. The breakdown following frontend integration is isolated to the FastAPI SSE route integration, API response contracts, and UI artifact naming. The project is safe to repair with minimal targeted edits.
