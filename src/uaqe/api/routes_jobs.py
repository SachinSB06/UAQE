"""Job Management, Execution, Telemetry, and SSE Streaming Routes.

Provides REST and SSE endpoints for UAQE God Mode Frontend.
"""

from __future__ import annotations

import os
import json
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from uaqe.telemetry.process_metrics import get_host_environment_info
from uaqe.telemetry.runtime_monitor import RuntimeMonitor
from uaqe.orchestration.hardware_target_registry import HardwareTargetRegistry
from uaqe.orchestration.optimization_orchestrator import OptimizationOrchestrator
from uaqe.orchestration.universal_model_ingestor import UniversalModelIngestor
from uaqe.orchestration.universal_dataset_ingestor import UniversalDatasetIngestor
from uaqe.orchestration.task_detector import TaskDetector
from uaqe.orchestration.compatibility_checker import CompatibilityChecker
from uaqe.orchestration.optimization_planner import OptimizationPlanner
from uaqe.orchestration.hardware_target_registry import HardwareTargetRegistry
from uaqe.api.routes_uploads import get_staged_model, get_staged_dataset, get_verified_samples
from .schemas import (
    ProvenanceLabel,
    ProvenanceMetric,
    InputSourceType,
    SystemStatusResponse,
    JobSummaryResponse,
    JobDetailResponse,
    ModelInspectionResponse,
    DatasetInspectionResponse,
    OptimizationPlanResponse,
    CandidateSummary,
    CandidateArtifactMetadata,
    TelemetryResponse,
    TelemetryPhaseData,
    TelemetrySample,
    AnalyzeRequest,
    PlanRequest,
    OptimizeRequest
)

router = APIRouter(prefix="/api", tags=["jobs"])

# ---------------------------------------------------------------------------
# In-memory job state registry
# States: QUEUED | RUNNING | COMPLETED | FAILED
# ---------------------------------------------------------------------------
ACTIVE_JOB_STATUS: Dict[str, str] = {}

# ---------------------------------------------------------------------------
# In-memory registry of active SSE queues per job.
# Structure: { job_id: [asyncio.Queue, ...] }
# Queues are created in the asyncio event loop that serves the SSE endpoint.
# The background worker uses the thread-safe helper _broadcast_event_threadsafe.
# ---------------------------------------------------------------------------
ACTIVE_EVENT_QUEUES: Dict[str, List[asyncio.Queue]] = {}

# Reference to the running asyncio event loop for thread-safe dispatch.
# Set once when the first SSE connection arrives; valid for the lifetime of
# the server process since FastAPI/Uvicorn runs a single event loop.
_SERVER_LOOP: Optional[asyncio.AbstractEventLoop] = None


from pathlib import Path

# Repo root is 4 levels above routes_jobs.py (api -> uaqe -> src -> repo)
REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)


def _get_jobs_root() -> str:
    """Return the absolute path to output/jobs directory."""
    jobs_dir = os.path.join(REPO_ROOT, "output", "jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    return jobs_dir


def _broadcast_event_threadsafe(job_id: str, event_data: Dict[str, Any]) -> None:
    """Thread-safe broadcast of an SSE event to all connected clients for this job.

    Called from background worker threads.  Uses call_soon_threadsafe so the
    asyncio.Queue is only touched from the event-loop thread, avoiding the race
    condition described in AUDIT-002.
    """
    loop = _SERVER_LOOP
    if loop is None or not loop.is_running():
        return

    queues = ACTIVE_EVENT_QUEUES.get(job_id)
    if not queues:
        return

    for q in list(queues):
        try:
            loop.call_soon_threadsafe(q.put_nowait, event_data)
        except Exception:
            pass


# Keep the old name as an alias so existing code inside the worker still works.
def _broadcast_event(job_id: str, event_data: Dict[str, Any]) -> None:
    """Deprecated: alias for _broadcast_event_threadsafe."""
    _broadcast_event_threadsafe(job_id, event_data)


# ---------------------------------------------------------------------------
# Helper: resolve real-time status for a job
# ---------------------------------------------------------------------------
def _get_job_status(job_id: str) -> str:
    """Return the current status string for a job.

    Precedence:
    1. In-memory ACTIVE_JOB_STATUS (authoritative for running/queued jobs).
    2. Disk: metrics.json exists -> COMPLETED.
    3. Disk: job directory exists -> RUNNING (still writing outputs).
    4. Default: UNKNOWN.
    """
    if job_id in ACTIVE_JOB_STATUS:
        return ACTIVE_JOB_STATUS[job_id]
    # Completed jobs on disk (server restarted after completion)
    jobs_root = _get_jobs_root()
    job_dir = os.path.join(jobs_root, job_id)
    if os.path.isdir(job_dir):
        if os.path.exists(os.path.join(job_dir, "metrics.json")):
            return "COMPLETED"
        return "RUNNING"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# SYSTEM STATUS
# ---------------------------------------------------------------------------
@router.get("/status", response_model=SystemStatusResponse)
def get_system_status() -> SystemStatusResponse:
    """Return system readiness, supported formats, targets, and profiles."""
    hw_targets = list(HardwareTargetRegistry._PROFILES.values())
    return SystemStatusResponse(
        status="READY",
        version="1.0.0",
        supported_model_formats=["pytorch_checkpoint", "onnx", "safetensors", "tflite"],
        supported_tasks=["image_classification"],
        supported_profiles=["balanced", "accuracy_first", "size_first", "latency_first", "storage_first"],
        supported_targets=hw_targets,
        host_environment=get_host_environment_info()
    )


# ---------------------------------------------------------------------------
# JOB LIST
# ---------------------------------------------------------------------------
@router.get("/jobs", response_model=List[JobSummaryResponse])
def list_jobs() -> List[JobSummaryResponse]:
    """List all historical optimization jobs discovered in output/jobs/."""
    jobs_root = _get_jobs_root()
    if not os.path.exists(jobs_root):
        return []

    summaries: List[JobSummaryResponse] = []
    job_folders = sorted(
        [f for f in os.listdir(jobs_root) if os.path.isdir(os.path.join(jobs_root, f))],
        reverse=True
    )

    for job_id in job_folders:
        job_dir = os.path.join(jobs_root, job_id)
        manifest_path = os.path.join(job_dir, "input_manifest.json")
        metrics_path = os.path.join(job_dir, "metrics.json")
        history_path = os.path.join(job_dir, "optimization_history.json")

        created_at = datetime.fromtimestamp(os.path.getctime(job_dir)).isoformat()
        model_name = "Unknown Model"
        dataset_name = "Unknown Dataset"
        target_hw = "raspberrypi5"
        profile = "balanced"

        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                    model_name = os.path.basename(manifest.get("model_path", model_name))
                    dataset_name = os.path.basename(manifest.get("dataset_path", dataset_name))
                    target_hw = manifest.get("target_hardware", target_hw)
                    profile = manifest.get("optimization_profile", profile)
            except Exception:
                pass

        fp32_acc = None
        final_acc = None
        acc_loss_pp = None
        safety_class = None
        orig_size = None
        opt_size = None
        size_red = None
        fp32_lat = None
        opt_lat = None
        lat_change = None
        selected_cand_name = None
        stopping_reason = None
        verdict = "PENDING"

        # Determine real status from registry / disk
        status = _get_job_status(job_id)

        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                    fp32_acc = m.get("fp32_accuracy")
                    final_acc = m.get("optimized_accuracy")
                    acc_loss_pp = m.get("accuracy_loss_pp")
                    safety_class = m.get("accuracy_safety_classification")
                    orig_size = m.get("original_size_bytes")
                    opt_size = m.get("optimized_size_bytes")
                    size_red = m.get("storage_reduction_percent")
                    fp32_lat = m.get("fp32_latency_ms")
                    opt_lat = m.get("optimized_latency_ms")
                    lat_change = m.get("latency_change_percent")
                    selected_cand_name = m.get("selected_candidate_name")
                    stopping_reason = m.get("stopping_reason")
                    if m.get("baseline_status") == "INVALID_BASELINE" or m.get("validation_status") == "FAILED" or m.get("verdict") == "FAILED":
                        verdict = "FAILED"
                        status = "FAILED"
                    else:
                        verdict = "VERIFIED WITH CAVEATS" if not m.get("accuracy_constraint_satisfied", True) else "VERIFIED"
            except Exception:
                pass
        elif status in ("RUNNING", "QUEUED"):
            verdict = "IN_PROGRESS"

        candidate_count = 0
        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    hist = json.load(f)
                    candidate_count = len(hist)
            except Exception:
                pass

        summaries.append(JobSummaryResponse(
            job_id=job_id,
            created_at=created_at,
            model_name=model_name,
            dataset_name=dataset_name,
            target_hardware=target_hw,
            optimization_profile=profile,
            status=status,
            verdict=verdict,
            fp32_accuracy=fp32_acc,
            final_accuracy=final_acc,
            accuracy_loss_pp=acc_loss_pp,
            accuracy_classification=safety_class,
            original_size_bytes=orig_size,
            optimized_size_bytes=opt_size,
            size_reduction_percent=size_red,
            fp32_latency_ms=fp32_lat,
            optimized_latency_ms=opt_lat,
            latency_change_percent=lat_change,
            candidates_evaluated=candidate_count,
            selected_candidate_name=selected_cand_name,
            stopping_reason=stopping_reason
        ))

    return summaries


# ---------------------------------------------------------------------------
# JOB COMPARE
# ---------------------------------------------------------------------------
@router.get("/jobs/compare")
def compare_jobs(job_a: str, job_b: str) -> Dict[str, Any]:
    """Provide a structured side-by-side technical comparison between two optimization jobs."""
    jobs_root = _get_jobs_root()
    dir_a = os.path.join(jobs_root, job_a)
    dir_b = os.path.join(jobs_root, job_b)

    if not os.path.isdir(dir_a):
        raise HTTPException(status_code=404, detail=f"Job A '{job_a}' not found.")
    if not os.path.isdir(dir_b):
        raise HTTPException(status_code=404, detail=f"Job B '{job_b}' not found.")

    def load_job_summary(jdir: str, jid: str) -> Dict[str, Any]:
        metrics_file = os.path.join(jdir, "metrics.json")
        model_insp_file = os.path.join(jdir, "model_inspection.json")
        dataset_insp_file = os.path.join(jdir, "dataset_inspection.json")
        telemetry_file = os.path.join(jdir, "telemetry.json")
        manifest_file = os.path.join(jdir, "input_manifest.json")

        metrics = {}
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except Exception:
                pass

        model_insp = {}
        if os.path.exists(model_insp_file):
            try:
                with open(model_insp_file, "r", encoding="utf-8") as f:
                    model_insp = json.load(f)
            except Exception:
                pass

        dataset_insp = {}
        if os.path.exists(dataset_insp_file):
            try:
                with open(dataset_insp_file, "r", encoding="utf-8") as f:
                    dataset_insp = json.load(f)
            except Exception:
                pass

        telemetry = {}
        if os.path.exists(telemetry_file):
            try:
                with open(telemetry_file, "r", encoding="utf-8") as f:
                    telemetry = json.load(f)
            except Exception:
                pass

        manifest = {}
        if os.path.exists(manifest_file):
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                pass

        base_tel = telemetry.get("baseline") or {}
        opt_tel = telemetry.get("final") or {}

        return {
            "job_id": jid,
            "architecture": model_insp.get("architecture") or model_insp.get("model_name", "Unknown"),
            "dataset": dataset_insp.get("dataset_name") or dataset_insp.get("format", "Unknown"),
            "target": manifest.get("target_hardware", "Unknown"),
            "profile": manifest.get("optimization_profile", "balanced"),
            "strategy": metrics.get("selected_strategy") or metrics.get("selected_candidate_name", "Unknown"),
            "verdict": metrics.get("verdict", "PENDING"),
            "baseline_accuracy": metrics.get("fp32_accuracy"),
            "optimized_accuracy": metrics.get("optimized_accuracy"),
            "accuracy_delta_pp": metrics.get("accuracy_delta_pp"),
            "accuracy_loss_pp": metrics.get("accuracy_loss_pp"),
            "safety_status": metrics.get("safety_status") or metrics.get("accuracy_safety_classification", "PENDING"),
            "original_size_bytes": metrics.get("original_size_bytes"),
            "optimized_size_bytes": metrics.get("optimized_size_bytes"),
            "storage_reduction_percent": metrics.get("storage_reduction_percent"),
            "fp32_latency_ms": metrics.get("fp32_latency_ms"),
            "optimized_latency_ms": metrics.get("optimized_latency_ms"),
            "latency_change_percent": metrics.get("latency_change_percent"),
            "throughput_ips": metrics.get("throughput_images_per_sec"),
            "baseline_cpu_avg_pct": base_tel.get("avg_cpu_percent") or base_tel.get("cpu_avg_pct"),
            "baseline_cpu_peak_pct": base_tel.get("peak_cpu_percent") or base_tel.get("cpu_peak_pct"),
            "optimized_cpu_avg_pct": opt_tel.get("avg_cpu_percent") or opt_tel.get("cpu_avg_pct"),
            "optimized_cpu_peak_pct": opt_tel.get("peak_cpu_percent") or opt_tel.get("cpu_peak_pct"),
            "baseline_ram_avg_mb": base_tel.get("avg_ram_mb") or base_tel.get("ram_avg_mb"),
            "baseline_ram_peak_mb": base_tel.get("peak_ram_mb") or base_tel.get("ram_peak_mb"),
            "optimized_ram_avg_mb": opt_tel.get("avg_ram_mb") or opt_tel.get("ram_avg_mb"),
            "optimized_ram_peak_mb": opt_tel.get("peak_ram_mb") or opt_tel.get("ram_peak_mb"),
            "checkpoint_mode": metrics.get("checkpoint_mode", "USER_UPLOAD"),
            "weight_source": metrics.get("weight_source", "ORIGINAL_MODEL"),
            "adaptation_status": metrics.get("adaptation_status", "NOT_REQUIRED"),
            "baseline_status": metrics.get("baseline_status", "PENDING"),
        }

    return {
        "job_a": load_job_summary(dir_a, job_a),
        "job_b": load_job_summary(dir_b, job_b),
        "comparison_timestamp": time.time()
    }


# ---------------------------------------------------------------------------
# JOB DETAIL
# ---------------------------------------------------------------------------
@router.get("/jobs/{job_id}", response_model=JobDetailResponse)
def get_job_detail(job_id: str) -> JobDetailResponse:
    """Retrieve the complete job bundle with provenance metadata."""
    jobs_root = _get_jobs_root()
    job_dir = os.path.join(jobs_root, job_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    # 1. Model inspection
    model_insp_path = os.path.join(job_dir, "model_inspection.json")
    model_raw = {}
    if os.path.exists(model_insp_path):
        with open(model_insp_path, "r", encoding="utf-8") as f:
            model_raw = json.load(f)

    model_insp = ModelInspectionResponse(
        format=ProvenanceMetric(value=model_raw.get("format", "unknown"), unit="", source=ProvenanceLabel.DETECTED, origin="model_inspection.json", description="Model file storage format"),
        framework=ProvenanceMetric(value=model_raw.get("framework", "PyTorch"), unit="", source=ProvenanceLabel.DETECTED, origin="model_inspection.json", description="Underlying DL framework"),
        architecture=ProvenanceMetric(value=model_raw.get("architecture", "Unknown"), unit="", source=ProvenanceLabel.DETECTED, origin="model_inspection.json", description="Identified neural network architecture"),
        task=ProvenanceMetric(value=model_raw.get("task", "image_classification"), unit="", source=ProvenanceLabel.DETECTED, origin="model_inspection.json", description="Task family"),
        input_shape=ProvenanceMetric(value=model_raw.get("input_shape", [1, 3, 224, 224]), unit="", source=ProvenanceLabel.DETECTED, origin="model_inspection.json", description="Input tensor dimension"),
        output_shape=ProvenanceMetric(value=model_raw.get("output_shape", [1, 10]), unit="", source=ProvenanceLabel.DETECTED, origin="model_inspection.json", description="Output logits dimension"),
        parameter_count=ProvenanceMetric(value=model_raw.get("parameter_count", 0), unit="parameters", source=ProvenanceLabel.DETECTED, origin="model_inspection.json", description="Total parameter count"),
        tensor_count=ProvenanceMetric(value=model_raw.get("tensor_count", 0), unit="tensors", source=ProvenanceLabel.DETECTED, origin="model_inspection.json", description="Total weight tensors"),
        file_size_bytes=ProvenanceMetric(value=model_raw.get("file_size_bytes", 0), unit="bytes", source=ProvenanceLabel.DETECTED, origin="model_inspection.json", description="Baseline file size"),
        sha256=ProvenanceMetric(value=model_raw.get("source_sha256", ""), unit="", source=ProvenanceLabel.DETECTED, origin="model_inspection.json", description="Source artifact SHA-256"),
        dtype=ProvenanceMetric(value=model_raw.get("dtype", "float32"), unit="", source=ProvenanceLabel.DETECTED, origin="model_inspection.json", description="Default tensor precision"),
        raw_descriptor=model_raw
    )

    # 2. Dataset inspection
    dataset_insp_path = os.path.join(job_dir, "dataset_inspection.json")
    dataset_raw = {}
    if os.path.exists(dataset_insp_path):
        with open(dataset_insp_path, "r", encoding="utf-8") as f:
            dataset_raw = json.load(f)

    dataset_insp = DatasetInspectionResponse(
        dataset_name=ProvenanceMetric(value=dataset_raw.get("dataset_name", "Unknown"), unit="", source=ProvenanceLabel.DETECTED, origin="dataset_inspection.json", description="Dataset identifier"),
        detected_format=ProvenanceMetric(value=dataset_raw.get("detected_format", "unknown"), unit="", source=ProvenanceLabel.DETECTED, origin="dataset_inspection.json", description="Dataset packaging format"),
        adapter_type=ProvenanceMetric(value=dataset_raw.get("adapter_type", "UniversalAdapter"), unit="", source=ProvenanceLabel.DETECTED, origin="dataset_inspection.json", description="Resolved ingestion adapter"),
        class_count=ProvenanceMetric(value=dataset_raw.get("class_count", 10), unit="classes", source=ProvenanceLabel.DETECTED, origin="dataset_inspection.json", description="Number of distinct label classes"),
        class_names=ProvenanceMetric(value=dataset_raw.get("class_names", []), unit="", source=ProvenanceLabel.DETECTED, origin="dataset_inspection.json", description="Class name taxonomy"),
        splits=ProvenanceMetric(value=dataset_raw.get("splits", {}), unit="samples", source=ProvenanceLabel.DETECTED, origin="dataset_inspection.json", description="Train/Val/Test sample split counts"),
        class_distribution=ProvenanceMetric(value=dataset_raw.get("class_distribution", {}), unit="distribution", source=ProvenanceLabel.DETECTED, origin="dataset_inspection.json", description="Class distribution per split"),
        raw_descriptor=dataset_raw
    )

    # 3. Optimization plan
    plan_path = os.path.join(job_dir, "optimization_plan.json")
    plan_raw = {}
    if os.path.exists(plan_path):
        with open(plan_path, "r", encoding="utf-8") as f:
            plan_raw = json.load(f)

    opt_plan = OptimizationPlanResponse(
        target_hardware=ProvenanceMetric(value=plan_raw.get("target_hardware", "raspberrypi5"), unit="", source=ProvenanceLabel.CONFIGURED, origin="optimization_plan.json", description="Selected execution target"),
        optimization_profile=ProvenanceMetric(value=plan_raw.get("profile", "balanced"), unit="", source=ProvenanceLabel.CONFIGURED, origin="optimization_plan.json", description="Optimization trade-off profile"),
        max_candidates_budget=ProvenanceMetric(value=plan_raw.get("search_budget", {}).get("max_candidates", 10), unit="candidates", source=ProvenanceLabel.POLICY, origin="optimization_plan.json", description="Maximum autonomous candidate budget"),
        max_allowed_accuracy_loss_pp=ProvenanceMetric(value=plan_raw.get("accuracy_safety", {}).get("max_acceptable_loss_pp", 4.0), unit="pp", source=ProvenanceLabel.POLICY, origin="optimization_plan.json", description="Hard safety boundary for profile"),
        safety_policy_excellent_pp=ProvenanceMetric(value=1.0, unit="pp", source=ProvenanceLabel.POLICY, origin="accuracy_safety_policy", description="EXCELLENT classification threshold"),
        safety_policy_acceptable_pp=ProvenanceMetric(value=4.0, unit="pp", source=ProvenanceLabel.POLICY, origin="accuracy_safety_policy", description="ACCEPTABLE classification threshold"),
        safety_policy_critical_pp=ProvenanceMetric(value=4.0, unit="pp", source=ProvenanceLabel.POLICY, origin="accuracy_safety_policy", description="CRITICAL safety threshold"),
        objective_weights=ProvenanceMetric(value=plan_raw.get("objective_weights", {"accuracy": 0.5, "size": 0.25, "latency": 0.25}), unit="weights", source=ProvenanceLabel.POLICY, origin="optimization_plan.json", description="Multi-objective scoring weights"),
        calib_samples=ProvenanceMetric(value=plan_raw.get("calibration", {}).get("samples", 256), unit="samples", source=ProvenanceLabel.CONFIGURED, origin="optimization_plan.json", description="Calibration sample count"),
        test_samples=ProvenanceMetric(value=plan_raw.get("validation", {}).get("test_samples", 1000), unit="samples", source=ProvenanceLabel.CONFIGURED, origin="optimization_plan.json", description="Independent test evaluation sample count"),
        raw_plan=plan_raw
    )

    # 4. Metrics & Provenance
    metrics_path = os.path.join(job_dir, "metrics.json")
    metrics_provenance: Dict[str, ProvenanceMetric[Any]] = {}
    metrics_raw = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_raw = json.load(f)

        metrics_provenance = {
            "fp32_accuracy": ProvenanceMetric(value=metrics_raw.get("fp32_accuracy"), unit="ratio", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="FP32 Reference Accuracy"),
            "optimized_accuracy": ProvenanceMetric(value=metrics_raw.get("optimized_accuracy"), unit="ratio", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="Final Optimized Accuracy"),
            "accuracy_loss_pp": ProvenanceMetric(value=metrics_raw.get("accuracy_loss_pp"), unit="pp", source=ProvenanceLabel.CALCULATED, origin="metrics.json", description="Accuracy Loss (FP32 - Optimized) in Percentage Points"),
            "accuracy_delta_pp": ProvenanceMetric(value=metrics_raw.get("accuracy_delta_pp"), unit="pp", source=ProvenanceLabel.CALCULATED, origin="metrics.json", description="Accuracy Delta (Optimized - FP32) in Percentage Points"),
            "accuracy_safety_classification": ProvenanceMetric(value=metrics_raw.get("accuracy_safety_classification"), unit="", source=ProvenanceLabel.CALCULATED, origin="metrics.json", description="Safety Quality Tier"),
            "safety_status": ProvenanceMetric(value=metrics_raw.get("safety_status", metrics_raw.get("accuracy_safety_classification")), unit="", source=ProvenanceLabel.CALCULATED, origin="metrics.json", description="Safety Status"),
            "baseline_status": ProvenanceMetric(value=metrics_raw.get("baseline_status", "VALID"), unit="", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="FP32 Baseline Validity Status"),
            "candidate_status": ProvenanceMetric(value=metrics_raw.get("candidate_status", "EVALUATED"), unit="", source=ProvenanceLabel.CALCULATED, origin="metrics.json", description="Candidate Evaluation Status"),
            "winner": ProvenanceMetric(value=metrics_raw.get("winner", "NONE"), unit="", source=ProvenanceLabel.CALCULATED, origin="metrics.json", description="Selected Winner Candidate"),
            "baseline_threshold_percent": ProvenanceMetric(value=metrics_raw.get("baseline_threshold_percent", 25.0), unit="%", source=ProvenanceLabel.POLICY, origin="metrics.json", description="Active Baseline Accuracy Validity Threshold"),
            "checkpoint_mode": ProvenanceMetric(value=metrics_raw.get("checkpoint_mode", "USER_UPLOAD"), unit="", source=ProvenanceLabel.CONFIGURED, origin="metrics.json", description="Model Checkpoint Mode (USER_UPLOAD or VERIFIED_BENCHMARK)"),
            "weight_source": ProvenanceMetric(value=metrics_raw.get("weight_source", "ORIGINAL_MODEL"), unit="", source=ProvenanceLabel.DETECTED, origin="metrics.json", description="Model Weight Lifecycle Source"),
            "adaptation_status": ProvenanceMetric(value=metrics_raw.get("adaptation_status", "NOT_REQUIRED"), unit="", source=ProvenanceLabel.DETECTED, origin="metrics.json", description="Classifier Adaptation Status"),
            "accuracy_constraint_satisfied": ProvenanceMetric(value=metrics_raw.get("accuracy_constraint_satisfied"), unit="", source=ProvenanceLabel.CALCULATED, origin="metrics.json", description="Whether accuracy meets profile constraint"),
            "original_size_bytes": ProvenanceMetric(value=metrics_raw.get("original_size_bytes"), unit="bytes", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="FP32 Model Disk Footprint"),
            "optimized_size_bytes": ProvenanceMetric(value=metrics_raw.get("optimized_size_bytes"), unit="bytes", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="Final Optimized Model Footprint"),
            "storage_reduction_percent": ProvenanceMetric(value=metrics_raw.get("storage_reduction_percent"), unit="%", source=ProvenanceLabel.CALCULATED, origin="metrics.json", description="Disk Footprint Reduction Percentage"),
            "fp32_latency_ms": ProvenanceMetric(value=metrics_raw.get("fp32_latency_ms"), unit="ms", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="Host CPU Pure FP32 Inference Latency"),
            "fp32_p50_latency_ms": ProvenanceMetric(value=metrics_raw.get("fp32_p50_latency_ms"), unit="ms", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="Host CPU FP32 P50 Latency"),
            "fp32_p95_latency_ms": ProvenanceMetric(value=metrics_raw.get("fp32_p95_latency_ms"), unit="ms", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="Host CPU FP32 P95 Latency"),
            "fp32_throughput_img_s": ProvenanceMetric(value=metrics_raw.get("fp32_throughput_img_s"), unit="img/s", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="Host CPU Pure FP32 Throughput"),
            "optimized_latency_ms": ProvenanceMetric(value=metrics_raw.get("optimized_latency_ms"), unit="ms", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="Host CPU Optimized Inference Latency"),
            "pure_inference_latency_ms": ProvenanceMetric(value=metrics_raw.get("pure_inference_latency_ms", metrics_raw.get("optimized_latency_ms")), unit="ms", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="Host CPU Pure Optimized Inference Latency"),
            "int8_latency_ms": ProvenanceMetric(value=metrics_raw.get("int8_latency_ms", metrics_raw.get("optimized_latency_ms")), unit="ms", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="Host CPU INT8 Pure Inference Latency"),
            "int8_p50_latency_ms": ProvenanceMetric(value=metrics_raw.get("int8_p50_latency_ms"), unit="ms", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="Host CPU INT8 P50 Latency"),
            "int8_p95_latency_ms": ProvenanceMetric(value=metrics_raw.get("int8_p95_latency_ms"), unit="ms", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="Host CPU INT8 P95 Latency"),
            "int8_throughput_img_s": ProvenanceMetric(value=metrics_raw.get("int8_throughput_img_s", metrics_raw.get("throughput_images_per_sec")), unit="img/s", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="Host CPU INT8 Pure Throughput"),
            "latency_change_percent": ProvenanceMetric(value=metrics_raw.get("latency_change_percent"), unit="%", source=ProvenanceLabel.CALCULATED, origin="metrics.json", description="Latency Speedup Percentage"),
            "throughput_images_per_sec": ProvenanceMetric(value=metrics_raw.get("throughput_images_per_sec"), unit="img/s", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="Host CPU Inference Throughput"),
            "prediction_agreement_percent": ProvenanceMetric(value=metrics_raw.get("prediction_agreement_percent"), unit="%", source=ProvenanceLabel.CALCULATED, origin="metrics.json", description="Prediction Match Rate with FP32 Baseline"),
            "selected_strategy": ProvenanceMetric(value=metrics_raw.get("selected_strategy"), unit="", source=ProvenanceLabel.CONFIGURED, origin="metrics.json", description="Selected Optimization Strategy"),
            "accuracy_eval_total_ms": ProvenanceMetric(value=metrics_raw.get("accuracy_eval_total_ms"), unit="ms", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="Dataset Accuracy Evaluation Total Duration"),
            "accuracy_eval_per_image_ms": ProvenanceMetric(value=metrics_raw.get("accuracy_eval_per_image_ms"), unit="ms", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="Dataset Accuracy Evaluation Per-Sample Duration"),
            "benchmark_provenance": ProvenanceMetric(value=metrics_raw.get("benchmark_provenance"), unit="", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="Benchmark Provenance Specification"),
            "fp32_benchmark_provenance": ProvenanceMetric(value=metrics_raw.get("fp32_benchmark_provenance"), unit="", source=ProvenanceLabel.MEASURED, origin="metrics.json", description="FP32 Benchmark Provenance Specification"),
            "validation_status": ProvenanceMetric(value=metrics_raw.get("validation_status", "PASSED" if metrics_raw.get("accuracy_constraint_satisfied", True) else "FAILED"), unit="", source=ProvenanceLabel.CALCULATED, origin="metrics.json", description="Validation Gate Status"),
            "verdict": ProvenanceMetric(value=metrics_raw.get("verdict", "VERIFIED"), unit="", source=ProvenanceLabel.CALCULATED, origin="metrics.json", description="Final Verification Verdict")
        }

    # 5. Candidates history
    history_path = os.path.join(job_dir, "optimization_history.json")
    candidate_summaries: List[CandidateSummary] = []
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            hist_raw = json.load(f)
            for c in hist_raw:
                rejection_reason = None
                action_taken = "Candidate accepted."
                if c.get("is_critical", False):
                    rejection_reason = f"Exceeded safety threshold ({c.get('accuracy_loss_pp', 0.0):.2f} pp loss > safety limit)"
                    action_taken = "Rejected by safety policy."
                elif not c.get("is_satisfied", True):
                    rejection_reason = f"Did not meet target profile constraint ({c.get('accuracy_loss_pp', 0.0):.2f} pp loss)"
                    action_taken = "Rejected by profile constraint."

                # Dynamically resolve canonical candidate artifact metadata
                cand_id = c.get("candidate_id", "")
                cand_dir = os.path.join(job_dir, "candidates", cand_id)
                cand_model_path = c.get("model_path")
                target_cand_file = None

                if cand_model_path:
                    fname = os.path.basename(cand_model_path)
                    p_in_cand = os.path.join(cand_dir, fname)
                    if os.path.exists(p_in_cand):
                        target_cand_file = p_in_cand
                    elif os.path.exists(cand_model_path):
                        target_cand_file = cand_model_path

                if not target_cand_file and os.path.isdir(cand_dir):
                    cand_files = [
                        f for f in os.listdir(cand_dir)
                        if os.path.isfile(os.path.join(cand_dir, f)) and not f.endswith(('.json', '.txt', '.log', '.csv'))
                    ]
                    if cand_files:
                        target_cand_file = os.path.join(cand_dir, cand_files[0])

                cand_artifact_meta = None
                if target_cand_file and os.path.exists(target_cand_file):
                    fname = os.path.basename(target_cand_file)
                    ext = os.path.splitext(fname)[1].lstrip('.').lower() or 'bin'
                    file_sz = os.path.getsize(target_cand_file)
                    rel_p = os.path.relpath(target_cand_file, job_dir).replace('\\', '/')

                    cand_sha = (
                        c.get("artifact_metadata", {}).get("candidate_artifact_sha256")
                        or c.get("benchmark_provenance", {}).get("artifact_sha256")
                    )
                    if not cand_sha:
                        try:
                            import hashlib
                            h = hashlib.sha256()
                            with open(target_cand_file, 'rb') as fp:
                                while chunk := fp.read(65536):
                                    h.update(chunk)
                            cand_sha = h.hexdigest()
                        except Exception:
                            cand_sha = None

                    cand_artifact_meta = CandidateArtifactMetadata(
                        filename=fname,
                        format=ext,
                        size_bytes=file_sz,
                        sha256=cand_sha,
                        download_url=f"/api/jobs/{job_id}/candidates/{cand_id}/download",
                        relative_path=rel_p
                    )
                elif cand_id:
                    raw_fname = os.path.basename(cand_model_path) if cand_model_path else f"{cand_id}.bin"
                    ext = os.path.splitext(raw_fname)[1].lstrip('.').lower() or 'bin'
                    cand_artifact_meta = CandidateArtifactMetadata(
                        filename=raw_fname,
                        format=ext,
                        size_bytes=c.get("model_size_bytes", 0),
                        sha256=c.get("artifact_metadata", {}).get("candidate_artifact_sha256") or c.get("benchmark_provenance", {}).get("artifact_sha256"),
                        download_url=f"/api/jobs/{job_id}/candidates/{cand_id}/download",
                        relative_path=f"candidates/{cand_id}/{raw_fname}"
                    )

                candidate_summaries.append(CandidateSummary(
                    candidate_id=c.get("candidate_id", ""),
                    candidate_name=c.get("candidate_name", ""),
                    strategy_type=c.get("strategy_type", ""),
                    top1_accuracy=c.get("top1_accuracy", 0.0),
                    accuracy_loss_pp=c.get("accuracy_loss_pp", 0.0),
                    safety_classification=c.get("safety_classification", "UNKNOWN"),
                    model_size_bytes=c.get("model_size_bytes", 0),
                    size_reduction_percent=round(c.get("size_reduction", 0.0) * 100.0, 2),
                    latency_mean_ms=c.get("latency_mean_ms", 0.0),
                    latency_reduction_percent=round(c.get("latency_reduction", 0.0) * 100.0, 2),
                    throughput_ips=c.get("throughput_ips", 0.0),
                    composite_score=c.get("composite_score", 0.0),
                    is_satisfied=c.get("is_satisfied", False),
                    is_critical=c.get("is_critical", False),
                    rejection_reason=rejection_reason,
                    action_taken=action_taken,
                    artifact_metadata=c.get("artifact_metadata", {}),
                    benchmark_provenance=c.get("benchmark_provenance"),
                    artifact=cand_artifact_meta
                ))

    # 6. Pareto frontier
    pareto_path = os.path.join(job_dir, "pareto_frontier.json")
    pareto_raw = []
    if os.path.exists(pareto_path):
        with open(pareto_path, "r", encoding="utf-8") as f:
            pareto_raw = json.load(f)

    # 7. Report markdown
    report_path = os.path.join(job_dir, "report.md")
    report_md = ""
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_md = f.read()

    # 8. Available artifacts
    available_artifacts = []
    for root, _, files in os.walk(job_dir):
        for file in files:
            full_p = os.path.join(root, file)
            rel_p = os.path.relpath(full_p, job_dir)
            available_artifacts.append({
                "filename": file,
                "relative_path": rel_p,
                "size_bytes": os.path.getsize(full_p)
            })

    created_at = datetime.fromtimestamp(os.path.getctime(job_dir)).isoformat()

    # Dynamic status from registry
    real_status = _get_job_status(job_id)

    # Verdict: only VERIFIED when metrics exist confirming completion
    if metrics_raw:
        if metrics_raw.get("baseline_status") == "INVALID_BASELINE":
            verdict_str = "FAILED"
            real_status = "FAILED"
        elif metrics_raw.get("verdict") == "FAILED":
            verdict_str = "FAILED"
            real_status = "COMPLETED"
        else:
            verdict_str = metrics_raw.get("verdict", "VERIFIED" if metrics_raw.get("accuracy_constraint_satisfied", True) else "VERIFIED WITH CAVEATS")
            real_status = "COMPLETED"
    elif real_status in ("RUNNING", "QUEUED"):
        verdict_str = "IN_PROGRESS"
    else:
        verdict_str = "PENDING"

    return JobDetailResponse(
        job_id=job_id,
        job_dir=job_dir,
        created_at=created_at,
        status=real_status,
        verdict=verdict_str,
        model_inspection=model_insp,
        dataset_inspection=dataset_insp,
        optimization_plan=opt_plan,
        metrics=metrics_provenance,
        candidates=candidate_summaries,
        pareto_frontier=pareto_raw,
        stopping_reason=metrics_raw.get("stopping_reason", "TARGET_CONSTRAINTS_SATISFIED"),
        stopping_description=metrics_raw.get("stopping_description", ""),
        selected_candidate_id=metrics_raw.get("selected_candidate_id", ""),
        report_markdown=report_md,
        available_artifacts=available_artifacts,
        host_telemetry_status="MEASURED",
        target_hardware_status="PENDING"
    )


# ---------------------------------------------------------------------------
# TELEMETRY
# ---------------------------------------------------------------------------
# TELEMETRY
# ---------------------------------------------------------------------------
@router.get("/jobs/{job_id}/telemetry")
def get_job_telemetry(job_id: str) -> Dict[str, Any]:
    """Retrieve detailed CPU/RAM time-series telemetry for a job.
    
    Returns genuine measured data from telemetry.json. Never fabricates values.
    """
    jobs_root = _get_jobs_root()
    job_dir = os.path.join(jobs_root, job_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    telemetry_path = os.path.join(job_dir, "telemetry.json")
    if os.path.exists(telemetry_path):
        try:
            with open(telemetry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "job_id": job_id,
        "status": "NOT_AVAILABLE",
        "message": "Telemetry not yet recorded or pending execution.",
        "environment_info": get_host_environment_info(),
        "baseline": None,
        "final": None,
        "comparison": None,
        "phases": {},
        "execution_samples": [],
        "target_hardware_status": "PENDING",
        "host_validation_status": "MEASURED"
    }


# ---------------------------------------------------------------------------
# SSE STREAMING  ←  AUDIT-001 FIX
# ---------------------------------------------------------------------------
@router.get("/jobs/{job_id}/events")
async def stream_job_events(job_id: str, request: Request):
    """Stream real-time Server-Sent Events for a live optimization job.

    Fixes AUDIT-001 (missing route) and AUDIT-002 (thread safety).

    Behaviour:
    - Validates job_id; returns 404 for completely unknown jobs.
    - Registers an asyncio.Queue for this SSE client under ACTIVE_EVENT_QUEUES[job_id].
    - Drains events from the queue in real-time (no busy-polling).
    - Sends a lightweight `: ping` comment every 15 s as a keepalive heartbeat.
    - Terminates cleanly on client disconnect or after receiving a terminal event.
    - Removes the client queue from ACTIVE_EVENT_QUEUES on teardown.
    """
    global _SERVER_LOOP

    # Capture the running event loop for thread-safe worker -> queue dispatch.
    _SERVER_LOOP = asyncio.get_running_loop()

    jobs_root = _get_jobs_root()
    job_dir = os.path.join(jobs_root, job_id)

    # Accept if job is in the in-memory registry (just launched) OR already on disk.
    job_known = (job_id in ACTIVE_JOB_STATUS) or os.path.isdir(job_dir)
    if not job_known:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    # Create a per-client queue and register it.
    client_queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    ACTIVE_EVENT_QUEUES.setdefault(job_id, []).append(client_queue)

    # If job already reached terminal state before connection, deliver terminal event immediately
    current_status = _get_job_status(job_id)
    if current_status in ("COMPLETED", "FAILED"):
        metrics_p = os.path.join(job_dir, "metrics.json")
        metrics_data = {}
        if os.path.exists(metrics_p):
            try:
                with open(metrics_p, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
            except Exception:
                pass
        client_queue.put_nowait({
            "type": "complete" if current_status == "COMPLETED" else "error",
            "job_id": job_id,
            "status": current_status,
            "metrics": metrics_data,
            "message": f"Job {job_id} already in terminal state {current_status}."
        })

    async def event_generator():
        try:
            HEARTBEAT_INTERVAL = 15  # seconds
            last_heartbeat = time.monotonic()

            while True:
                # Check client disconnect
                if await request.is_disconnected():
                    break

                # Deliver heartbeat if queue has been idle long enough
                now = time.monotonic()
                if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                    yield ": ping\n\n"
                    last_heartbeat = now

                # Non-blocking drain of queued events
                try:
                    event_data = client_queue.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.1)
                    continue

                # Serialise and emit the event
                payload = json.dumps(event_data)
                yield f"data: {payload}\n\n"
                last_heartbeat = time.monotonic()

                # Terminal event types — close stream after delivering
                evt_type = event_data.get("type", "")
                if evt_type in ("complete", "error", "OPTIMIZATION_COMPLETED", "FINAL_RESULT"):
                    break

        finally:
            # Deregister this client's queue
            queues = ACTIVE_EVENT_QUEUES.get(job_id, [])
            if client_queue in queues:
                queues.remove(client_queue)
            if job_id in ACTIVE_EVENT_QUEUES and not ACTIVE_EVENT_QUEUES[job_id]:
                del ACTIVE_EVENT_QUEUES[job_id]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# ---------------------------------------------------------------------------
# INPUT RESOLUTION HELPERS
# ---------------------------------------------------------------------------
def _resolve_model(request: Union[AnalyzeRequest, PlanRequest, OptimizeRequest]) -> Tuple[str, str, str]:
    """Resolve model file path, sha256, and input source type."""
    import hashlib

    # 1. Check model_upload_id
    if request.model_upload_id:
        meta = get_staged_model(request.model_upload_id)
        if not meta or not os.path.exists(meta.get("staged_path", "")):
            raise HTTPException(status_code=404, detail=f"Uploaded model '{request.model_upload_id}' not found in staging.")
        return meta["staged_path"], meta.get("sha256", ""), InputSourceType.USER_UPLOAD.value

    # 2. Check model_id (sample)
    if request.model_id:
        samples = get_verified_samples()
        for m in samples.get("models", []):
            if m["id"] == request.model_id or m["name"] == request.model_id:
                return m["path"], m.get("sha256", ""), InputSourceType.PRE_VERIFIED_SAMPLE.value
        raise HTTPException(status_code=404, detail=f"Sample model '{request.model_id}' not found.")

    # 3. Check model_path (CLI/backward compatibility)
    if request.model_path:
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=400, detail=f"Model path does not exist: {request.model_path}")
        hasher = hashlib.sha256()
        with open(request.model_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return request.model_path, hasher.hexdigest(), InputSourceType.CLI_PATH.value

    raise HTTPException(
        status_code=400,
        detail="No model input specified. Please upload a model file or select a verified sample."
    )


def _resolve_dataset(request: Union[AnalyzeRequest, PlanRequest, OptimizeRequest]) -> Tuple[str, str, str]:
    """Resolve dataset path, manifest hash, and input source type."""
    # 1. Check dataset_upload_id
    if request.dataset_upload_id:
        meta = get_staged_dataset(request.dataset_upload_id)
        if not meta or not os.path.exists(meta.get("staged_path", "")):
            raise HTTPException(status_code=404, detail=f"Uploaded dataset '{request.dataset_upload_id}' not found in staging.")
        return meta["staged_path"], meta.get("manifest_hash", "staged_dataset_hash"), InputSourceType.USER_UPLOAD.value

    # 2. Check dataset_id (sample)
    if request.dataset_id:
        samples = get_verified_samples()
        for d in samples.get("datasets", []):
            if d["id"] == request.dataset_id or d["name"] == request.dataset_id:
                return d["path"], "pre_verified_sample_hash", InputSourceType.PRE_VERIFIED_SAMPLE.value
        raise HTTPException(status_code=404, detail=f"Sample dataset '{request.dataset_id}' not found.")

    # 3. Check dataset_path (CLI/backward compatibility)
    if request.dataset_path:
        if not os.path.exists(request.dataset_path):
            raise HTTPException(status_code=400, detail=f"Dataset path does not exist: {request.dataset_path}")
        return request.dataset_path, "cli_path_hash", InputSourceType.CLI_PATH.value

    raise HTTPException(
        status_code=400,
        detail="No dataset input specified. Please upload a dataset folder or select a verified sample."
    )


# ---------------------------------------------------------------------------
# ANALYZE  ←  AUDIT-004 FIX: now returns optimization_plan
# ---------------------------------------------------------------------------
@router.post("/jobs/analyze")
def analyze_model_and_dataset(request: AnalyzeRequest):
    """Inspect model, dataset, compatibility, and generate optimization plan.

    Fixes AUDIT-004: optimization_plan is now included in the response so
    InspectionPlanPage.tsx receives it via App.tsx setOptimizationPlan().
    """
    try:
        model_path, model_sha256, model_source = _resolve_model(request)
        dataset_path, dataset_hash, dataset_source = _resolve_dataset(request)

        model_ingestor = UniversalModelIngestor(model_path)
        model_desc = model_ingestor.inspect()
        try:
            model_caps = model_ingestor.get_capabilities()
        except Exception:
            model_caps = {}

        dataset_ingestor = UniversalDatasetIngestor(dataset_path)
        dataset_desc = dataset_ingestor.get_descriptor()
        task_info = TaskDetector.detect_task(model_desc, dataset_desc)
        compat = CompatibilityChecker.check_compatibility(model_desc, dataset_desc, task_info)

        # Check for Transformer / ViT architectures
        arch_lower = model_desc.get("architecture", "").lower()
        if "vit" in arch_lower or "transformer" in arch_lower or "patch" in arch_lower:
            compat["compatible"] = False
            compat["issues"].append(
                f"UNSUPPORTED_MODEL_ARCHITECTURE: Architecture '{model_desc.get('architecture')}' is a Vision Transformer "
                f"(~{round(model_desc.get('parameter_count', 0) / 1e6, 1)}M parameters). Autonomous quantization in UAQE "
                f"currently supports Convolutional Neural Network (CNN) architectures (ResNet-18/34/50 and MobileNetV2/V3). "
                f"Transformer attention QKV quantization is not supported in the current engine."
            )

        # Attach provenance tags
        model_desc["source_provenance"] = model_source
        model_desc["sha256"] = model_sha256
        dataset_desc["source_provenance"] = dataset_source
        dataset_desc["manifest_hash"] = dataset_hash

        # Generate optimization plan (AUDIT-004 fix)
        hw_profile = HardwareTargetRegistry.get_profile(request.target or "raspberrypi5")
        optimization_plan = OptimizationPlanner.create_plan(
            model_descriptor=model_desc,
            model_capabilities=model_caps,
            dataset_descriptor=dataset_desc,
            compatibility_report=compat,
            hardware_profile=hw_profile,
            profile_name=request.profile or "balanced"
        )

        return {
            "model_inspection": model_desc,
            "dataset_inspection": dataset_desc,
            "compatibility": compat,
            "model_source": model_source,
            "dataset_source": dataset_source,
            "optimization_plan": optimization_plan
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model/dataset analysis failed: {str(e)}")


# ---------------------------------------------------------------------------
# PLAN (dry-run)
# ---------------------------------------------------------------------------
@router.post("/jobs/plan")
def generate_optimization_plan(request: PlanRequest):
    """Generate optimization plan without executing."""
    model_path, model_sha256, model_source = _resolve_model(request)
    dataset_path, dataset_hash, dataset_source = _resolve_dataset(request)

    orchestrator = OptimizationOrchestrator(output_root=_get_jobs_root())
    checkpoint_mode = "VERIFIED_BENCHMARK" if model_source == InputSourceType.PRE_VERIFIED_SAMPLE.value else "USER_UPLOAD"
    job_config = {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "target_hardware": request.target,
        "optimization_profile": request.profile,
        "max_candidates": request.max_candidates,
        "auto_approve": True,
        "dry_run": True,
        "calib_samples": request.calib_samples,
        "test_samples": request.test_samples,
        "checkpoint_mode": checkpoint_mode
    }
    results = orchestrator.run(job_config)
    return results


# ---------------------------------------------------------------------------
# BACKGROUND WORKER
# ---------------------------------------------------------------------------
def _run_optimization_worker(
    job_id: str,
    model_path: str,
    dataset_path: str,
    target: str,
    profile: str,
    max_candidates: int,
    calib_samples: int,
    test_samples: int,
    checkpoint_mode: str = "USER_UPLOAD"
):
    """Background worker executing the real autonomous optimization and streaming events.

    Thread-safety: uses _broadcast_event_threadsafe (AUDIT-002 fix) which
    delegates to asyncio.loop.call_soon_threadsafe so queues are only mutated
    from the event-loop thread.
    """
    worker_monitor = None

    def progress_callback(event: Dict[str, Any]):
        event["job_id"] = job_id
        event["timestamp"] = time.time()
        stage = event.get("stage")
        if stage and worker_monitor:
            worker_monitor.set_phase(stage)
        _broadcast_event_threadsafe(job_id, event)

    try:
        # Mark job as RUNNING (AUDIT-003 fix)
        ACTIVE_JOB_STATUS[job_id] = "RUNNING"

        # Continuous host telemetry monitor running throughout worker execution
        def on_telemetry_sample(sample: Dict[str, Any]):
            telemetry_event = {
                "type": "telemetry",
                "job_id": job_id,
                "timestamp": sample.get("timestamp", time.time()),
                "cpu_system_pct": sample.get("system_cpu_percent", 0.0),
                "cpu_process_pct": sample.get("process_cpu_percent", 0.0),
                "ram_used_pct": sample.get("system_ram_percent", 0.0),
                "ram_process_mb": sample.get("process_ram_mb", 0.0),
                "ram_system_used_mb": sample.get("system_ram_used_mb", 0.0),
                "ram_system_available_mb": sample.get("system_ram_available_mb", 0.0),
                "peak_ram_process_mb": sample.get("peak_process_ram_mb", sample.get("process_ram_mb", 0.0)),
                "cpu_count_logical": sample.get("cpu_count_logical", 1),
                "cpu_frequency_current_mhz": sample.get("cpu_frequency_current_mhz", 0.0),
                "phase": sample.get("phase", "INITIALIZING")
            }
            _broadcast_event_threadsafe(job_id, telemetry_event)

        worker_monitor = RuntimeMonitor(
            sample_interval_sec=0.5,
            sample_callback=on_telemetry_sample,
            job_id=job_id
        )
        worker_monitor.start()

        progress_callback({"type": "stage_start", "stage": "INITIALIZING", "message": "Initializing optimization orchestrator..."})

        orchestrator = OptimizationOrchestrator(output_root=_get_jobs_root())
        job_config = {
            "job_id": job_id,
            "model_path": model_path,
            "dataset_path": dataset_path,
            "target_hardware": target,
            "optimization_profile": profile,
            "max_candidates": max_candidates,
            "auto_approve": True,
            "dry_run": False,
            "calib_samples": calib_samples,
            "test_samples": test_samples,
            "checkpoint_mode": checkpoint_mode,
            "progress_callback": progress_callback,
            "runtime_mode": "performance",
            "performance_enabled": True,
            "num_threads": 2
        }

        results = orchestrator.run(job_config)

        # Stop worker monitor and record execution samples to telemetry.json
        execution_samples = worker_monitor.stop()
        job_dir = os.path.join(_get_jobs_root(), job_id)
        telemetry_file = os.path.join(job_dir, "telemetry.json")
        if os.path.exists(telemetry_file):
            try:
                with open(telemetry_file, "r+", encoding="utf-8") as f:
                    tel_data = json.load(f)
                    if not tel_data.get("execution_samples"):
                        tel_data["execution_samples"] = execution_samples
                    f.seek(0)
                    json.dump(tel_data, f, indent=2)
                    f.truncate()
            except Exception:
                pass

        # Mark job as COMPLETED or FAILED
        if results.get("status") == "FAILED" or results.get("metrics", {}).get("baseline_status") == "INVALID_BASELINE":
            ACTIVE_JOB_STATUS[job_id] = "FAILED"
        else:
            ACTIVE_JOB_STATUS[job_id] = "COMPLETED"

        progress_callback({
            "type": "complete",
            "stage": "COMPLETED",
            "job_id": job_id,
            "results": results,
            "metrics": results.get("metrics", {}),
            "verdict": results.get("verdict", "VERIFIED"),
            "message": "Autonomous optimization complete"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Mark job as FAILED (AUDIT-003 fix)
        ACTIVE_JOB_STATUS[job_id] = "FAILED"
        progress_callback({
            "type": "error",
            "stage": "FAILED",
            "job_id": job_id,
            "error": str(e)
        })


def _copy_or_link_file(src: str, dst: str) -> None:
    """Hardlink a file if on same filesystem (0 extra bytes) or copy safely."""
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _copy_or_link_tree(src_dir: str, dst_dir: str) -> None:
    """Recursively hardlink directory tree (0 extra bytes) or copy safely."""
    os.makedirs(dst_dir, exist_ok=True)
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)
        if os.path.isdir(s):
            _copy_or_link_tree(s, d)
        else:
            _copy_or_link_file(s, d)


# ---------------------------------------------------------------------------
# OPTIMIZE (launch)
# ---------------------------------------------------------------------------
@router.post("/jobs/optimize")
def start_autonomous_optimization(request: OptimizeRequest, background_tasks: BackgroundTasks):
    """Launch autonomous optimization job in background worker with job-isolated inputs."""
    import shutil

    model_path, model_sha256, model_source = _resolve_model(request)
    dataset_path, dataset_hash, dataset_source = _resolve_dataset(request)

    # Inspect model architecture to verify optimization capability before launch
    try:
        model_ingestor = UniversalModelIngestor(model_path)
        model_desc = model_ingestor.inspect()
        arch = model_desc.get("architecture", "")
        arch_lower = arch.lower()
        if "vit" in arch_lower or "transformer" in arch_lower or "patch" in arch_lower:
            param_str = f"~{round(model_desc.get('parameter_count', 0) / 1e6, 1)}M parameters"
            raise HTTPException(
                status_code=400,
                detail=(
                    f"UNSUPPORTED_MODEL_ARCHITECTURE: Architecture '{arch}' is a Vision Transformer ({param_str}). "
                    f"Autonomous quantization in UAQE currently supports Convolutional Neural Network (CNN) architectures "
                    f"(ResNet-18/34/50 and MobileNetV2/V3). Vision Transformer attention QKV quantization is not supported in the current engine."
                )
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model pre-optimization inspection failed: {str(e)}")

    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    import uuid
    short_uuid = uuid.uuid4().hex[:8].upper()
    job_id = f"UAQE-{timestamp_str}-{short_uuid}"

    jobs_root = _get_jobs_root()

    # Pre-check available disk space
    try:
        _, _, free_bytes = shutil.disk_usage(jobs_root)
        model_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
        if free_bytes < (model_size + 100 * 1024 * 1024):
            free_mb = round(free_bytes / (1024 * 1024), 1)
            needed_mb = round((model_size + 100 * 1024 * 1024) / (1024 * 1024), 1)
            raise HTTPException(
                status_code=400,
                detail=(
                    f"INSUFFICIENT_STORAGE: Host disk space ({free_mb} MB free) is insufficient to stage "
                    f"job inputs (requires at least {needed_mb} MB free). Please free up disk space on the host."
                )
            )
    except HTTPException:
        raise
    except Exception:
        pass

    job_dir = os.path.join(jobs_root, job_id)
    inputs_dir = os.path.join(job_dir, "inputs")
    model_input_dir = os.path.join(inputs_dir, "model")
    dataset_input_dir = os.path.join(inputs_dir, "dataset")

    try:
        os.makedirs(model_input_dir, exist_ok=True)
        os.makedirs(dataset_input_dir, exist_ok=True)

        # Copy or hardlink staged model into job-isolated input folder (zero extra bytes on same volume)
        job_model_path = os.path.join(model_input_dir, os.path.basename(model_path))
        _copy_or_link_file(model_path, job_model_path)

        # Isolate dataset inputs with zero-copy hardlinking
        if model_source == InputSourceType.USER_UPLOAD.value or dataset_source == InputSourceType.USER_UPLOAD.value:
            _copy_or_link_tree(dataset_path, dataset_input_dir)
            job_dataset_path = dataset_input_dir
        else:
            job_dataset_path = dataset_path

        # Write input_manifest.json with provenance identities
        input_manifest = {
            "job_id": job_id,
            "created_at": datetime.now().isoformat(),
            "model_sha256": model_sha256,
            "dataset_manifest_hash": dataset_hash,
            "model_source": model_source,
            "dataset_source": dataset_source,
            "model_path": job_model_path,
            "dataset_path": job_dataset_path,
            "target_hardware": request.target,
            "optimization_profile": request.profile,
            "max_candidates": request.max_candidates,
            "calib_samples": request.calib_samples,
            "test_samples": request.test_samples
        }
        with open(os.path.join(job_dir, "input_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(input_manifest, f, indent=2)

    except OSError as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        if getattr(e, "winerror", None) == 112 or "space" in str(e).lower():
            raise HTTPException(
                status_code=400,
                detail=f"INSUFFICIENT_STORAGE: Host disk space exhausted while staging job inputs: {e}"
            )
        raise HTTPException(status_code=400, detail=f"Job staging I/O error: {e}")
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Job initialization error: {e}")

    # Register job as QUEUED immediately (AUDIT-003 fix)
    ACTIVE_JOB_STATUS[job_id] = "QUEUED"

    checkpoint_mode = "VERIFIED_BENCHMARK" if model_source == InputSourceType.PRE_VERIFIED_SAMPLE.value else "USER_UPLOAD"

    background_tasks.add_task(
        _run_optimization_worker,
        job_id=job_id,
        model_path=job_model_path,
        dataset_path=job_dataset_path,
        target=request.target,
        profile=request.profile,
        max_candidates=request.max_candidates,
        calib_samples=request.calib_samples,
        test_samples=request.test_samples,
        checkpoint_mode=checkpoint_mode
    )

    return {
        "job_id": job_id,
        "status": "LAUNCHED",
        "message": "Autonomous optimization job launched successfully.",
        "events_url": f"/api/jobs/{job_id}/events",
        "model_sha256": model_sha256,
        "dataset_manifest_hash": dataset_hash,
        "model_source": model_source,
        "dataset_source": dataset_source
    }


# ---------------------------------------------------------------------------
# ARTIFACT & CANDIDATE DOWNLOAD
# ---------------------------------------------------------------------------
@router.get("/jobs/{job_id}/candidates/{candidate_id}/download")
def download_candidate_artifact(job_id: str, candidate_id: str):
    """Securely download an artifact for a specific candidate with strict validation and isolation.
    
    Resolves artifact dynamically using job_id + candidate_id, never relying on global filename uniqueness.
    """
    jobs_root = _get_jobs_root()
    job_dir = os.path.realpath(os.path.join(jobs_root, job_id))

    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    # Strict validation of candidate_id against path traversal
    safe_cand_id = os.path.basename(candidate_id)
    if safe_cand_id != candidate_id or not candidate_id.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail=f"Invalid candidate identifier: {candidate_id}")

    cand_dir = os.path.realpath(os.path.join(job_dir, "candidates", safe_cand_id))
    if not cand_dir.startswith(job_dir) or not os.path.isdir(cand_dir):
        raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found in job {job_id}.")

    target_path = None
    target_filename = None

    # 1. Check optimization_history.json to resolve the exact candidate model filename
    history_path = os.path.join(job_dir, "optimization_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            for c in history:
                if c.get("candidate_id") == candidate_id:
                    mp = c.get("model_path")
                    if mp:
                        fname = os.path.basename(mp)
                        p = os.path.join(cand_dir, fname)
                        if os.path.exists(p):
                            target_path = p
                            target_filename = fname
                            break
                        elif os.path.exists(mp):
                            target_path = os.path.realpath(mp)
                            target_filename = fname
                            break
        except Exception:
            pass

    # 2. If not found via history, inspect candidate directory for model artifacts
    if not target_path or not os.path.exists(target_path):
        candidates_files = [
            f for f in os.listdir(cand_dir)
            if os.path.isfile(os.path.join(cand_dir, f)) and not f.endswith(('.json', '.txt', '.log', '.csv'))
        ]
        if candidates_files:
            target_filename = candidates_files[0]
            target_path = os.path.join(cand_dir, target_filename)

    if not target_path or not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"Artifact for candidate {candidate_id} not found in job {job_id}.")

    target_path = os.path.realpath(target_path)
    # Strict job containment check (Job isolation rule)
    if not target_path.startswith(cand_dir):
        raise HTTPException(status_code=403, detail="Access denied: path traversal detected.")

    return FileResponse(
        path=target_path,
        filename=target_filename,
        media_type="application/octet-stream"
    )


@router.get("/jobs/{job_id}/download/{filename:path}")
def download_job_artifact(job_id: str, filename: str, candidate_id: Optional[str] = None):
    """Securely download an artifact from the job directory with strict path sanitization.
    
    Supports candidate-specific downloads via candidate_id query param or relative candidate paths.
    """
    jobs_root = _get_jobs_root()
    job_dir = os.path.realpath(os.path.join(jobs_root, job_id))

    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    # 1. If candidate_id is specified, delegate to candidate directory resolution
    if candidate_id:
        return download_candidate_artifact(job_id=job_id, candidate_id=candidate_id)

    # 2. Check normalized path inside job_dir
    norm_path = filename.replace("\\", "/").lstrip("/")
    if ".." in norm_path:
        raise HTTPException(status_code=403, detail="Access denied: path traversal detected.")

    target_path = os.path.realpath(os.path.join(job_dir, norm_path))

    if not target_path.startswith(job_dir) or not os.path.exists(target_path) or os.path.isdir(target_path):
        # Check basename at root of job_dir
        safe_filename = os.path.basename(filename)
        root_path = os.path.realpath(os.path.join(job_dir, safe_filename))
        if root_path.startswith(job_dir) and os.path.exists(root_path) and not os.path.isdir(root_path):
            target_path = root_path
        else:
            # Fallback to candidates directory if file exists there (backward compatibility)
            import glob
            cand_matches = glob.glob(os.path.join(job_dir, "candidates", "*", safe_filename))
            if cand_matches and os.path.exists(cand_matches[0]):
                target_path = os.path.realpath(cand_matches[0])
            else:
                raise HTTPException(status_code=404, detail=f"Artifact {safe_filename} not found in job {job_id}.")

    return FileResponse(
        path=target_path,
        filename=os.path.basename(target_path),
        media_type="application/octet-stream"
    )

