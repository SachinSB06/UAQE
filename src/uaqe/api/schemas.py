"""Data contracts and schemas for UAQE API endpoints.

Enforces measurement provenance labels across all returned metrics:
DETECTED, CONFIGURED, MEASURED, CALCULATED, POLICY, PENDING, NOT_AVAILABLE.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Any, Optional, Generic, TypeVar, Union
from pydantic import BaseModel, Field

T = TypeVar("T")


class ProvenanceLabel(str, Enum):
    """Data provenance source classification."""
    DETECTED = "DETECTED"
    CONFIGURED = "CONFIGURED"
    MEASURED = "MEASURED"
    CALCULATED = "CALCULATED"
    POLICY = "POLICY"
    PENDING = "PENDING"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class ProvenanceMetric(BaseModel, Generic[T]):
    """Wrapper for a parameter/metric with explicit measurement provenance."""
    value: Optional[T] = None
    unit: str = ""
    source: ProvenanceLabel = ProvenanceLabel.NOT_AVAILABLE
    origin: str = ""
    description: str = ""


class SystemStatusResponse(BaseModel):
    """System capability status descriptor."""
    status: str = "READY"
    version: str = "1.0.0"
    supported_model_formats: List[str] = ["pytorch_checkpoint", "onnx", "safetensors", "tflite"]
    supported_tasks: List[str] = ["image_classification"]
    supported_profiles: List[str] = ["balanced", "accuracy_first", "size_first", "latency_first", "storage_first"]
    supported_targets: List[Dict[str, Any]] = []
    host_environment: Dict[str, Any] = {}


class InputSourceType(str, Enum):
    """Input provenance classification."""
    USER_UPLOAD = "USER_UPLOAD"
    PRE_VERIFIED_SAMPLE = "PRE_VERIFIED_SAMPLE"
    CLI_PATH = "CLI_PATH"


class ModelUploadResponse(BaseModel):
    """Response model for uploaded model file."""
    upload_id: str
    filename: str
    size_bytes: int
    sha256: str
    format: str
    status: str = "READY"
    source: str = "USER_UPLOAD"
    staged_path: Optional[str] = None


class DatasetManifestFile(BaseModel):
    """File entry in dataset manifest."""
    relative_path: str
    size_bytes: int
    sha256: str


class DatasetUploadResponse(BaseModel):
    """Response model for uploaded dataset folder/archive."""
    upload_id: str
    folder_name: str
    file_count: int
    total_size_bytes: int
    manifest_hash: str
    detected_format: str = "unknown"
    class_count: int = 0
    splits: Dict[str, int] = {}
    status: str = "READY"
    source: str = "USER_UPLOAD"
    staged_path: Optional[str] = None
    structure_valid: bool = True
    validation_message: str = "Dataset structure recognized and validated"


class AnalyzeRequest(BaseModel):
    """Request model for analyzing a model and dataset."""
    model_upload_id: Optional[str] = None
    model_id: Optional[str] = None
    model_path: Optional[str] = None

    dataset_upload_id: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_path: Optional[str] = None

    target: str = "raspberrypi5"
    profile: str = "balanced"
    calib_samples: int = 256
    test_samples: int = 1000


class PlanRequest(BaseModel):
    """Request model for generating an autonomous optimization plan."""
    model_upload_id: Optional[str] = None
    model_id: Optional[str] = None
    model_path: Optional[str] = None

    dataset_upload_id: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_path: Optional[str] = None

    target: str = "raspberrypi5"
    profile: str = "balanced"
    max_candidates: int = 10
    calib_samples: int = 256
    test_samples: int = 1000


class OptimizeRequest(BaseModel):
    """Request model for starting an autonomous optimization run."""
    model_upload_id: Optional[str] = None
    model_id: Optional[str] = None
    model_path: Optional[str] = None

    dataset_upload_id: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_path: Optional[str] = None

    target: str = "raspberrypi5"
    profile: str = "balanced"
    max_candidates: int = 10
    auto_approve: bool = True
    calib_samples: int = 256
    test_samples: int = 1000


class ModelInspectionResponse(BaseModel):
    """Structured inspection details for a model."""
    format: ProvenanceMetric[str]
    framework: ProvenanceMetric[str]
    architecture: ProvenanceMetric[str]
    task: ProvenanceMetric[str]
    input_shape: ProvenanceMetric[List[int]]
    output_shape: ProvenanceMetric[List[int]]
    parameter_count: ProvenanceMetric[int]
    tensor_count: ProvenanceMetric[int]
    file_size_bytes: ProvenanceMetric[int]
    sha256: ProvenanceMetric[str]
    dtype: ProvenanceMetric[str]
    raw_descriptor: Dict[str, Any] = {}


class DatasetInspectionResponse(BaseModel):
    """Structured inspection details for a dataset."""
    dataset_name: ProvenanceMetric[str]
    detected_format: ProvenanceMetric[str]
    adapter_type: ProvenanceMetric[str]
    class_count: ProvenanceMetric[int]
    class_names: ProvenanceMetric[List[str]]
    splits: ProvenanceMetric[Dict[str, int]]
    class_distribution: ProvenanceMetric[Dict[str, Dict[str, int]]]
    raw_descriptor: Dict[str, Any] = {}


class OptimizationPlanResponse(BaseModel):
    """Structured autonomous optimization plan details."""
    target_hardware: ProvenanceMetric[Union[str, Dict[str, Any]]]
    optimization_profile: ProvenanceMetric[str]
    max_candidates_budget: ProvenanceMetric[int]
    max_allowed_accuracy_loss_pp: ProvenanceMetric[float]
    safety_policy_excellent_pp: ProvenanceMetric[float]
    safety_policy_acceptable_pp: ProvenanceMetric[float]
    safety_policy_critical_pp: ProvenanceMetric[float]
    objective_weights: ProvenanceMetric[Dict[str, float]]
    calib_samples: ProvenanceMetric[int]
    test_samples: ProvenanceMetric[int]
    raw_plan: Dict[str, Any] = {}


class CandidateArtifactMetadata(BaseModel):
    """Canonical artifact metadata bound to a specific optimization candidate."""
    filename: str
    format: str
    size_bytes: int
    sha256: Optional[str] = None
    download_url: str
    relative_path: Optional[str] = None


class CandidateSummary(BaseModel):
    """Summary of an executed optimization candidate."""
    candidate_id: str
    candidate_name: str
    strategy_type: str
    top1_accuracy: float
    accuracy_loss_pp: float
    safety_classification: str
    model_size_bytes: int
    size_reduction_percent: float
    latency_mean_ms: float
    latency_reduction_percent: float
    throughput_ips: float
    composite_score: float
    is_satisfied: bool
    is_critical: bool
    rejection_reason: Optional[str] = None
    action_taken: str = ""
    artifact_metadata: Dict[str, Any] = {}
    benchmark_provenance: Optional[Dict[str, Any]] = None
    artifact: Optional[CandidateArtifactMetadata] = None


class JobSummaryResponse(BaseModel):
    """Summary record for a job listed in history."""
    job_id: str
    created_at: str
    model_name: str
    dataset_name: str
    target_hardware: str
    optimization_profile: str
    status: str
    verdict: str
    fp32_accuracy: Optional[float] = None
    final_accuracy: Optional[float] = None
    accuracy_loss_pp: Optional[float] = None
    accuracy_classification: Optional[str] = None
    original_size_bytes: Optional[int] = None
    optimized_size_bytes: Optional[int] = None
    size_reduction_percent: Optional[float] = None
    fp32_latency_ms: Optional[float] = None
    optimized_latency_ms: Optional[float] = None
    latency_change_percent: Optional[float] = None
    candidates_evaluated: int = 0
    selected_candidate_name: Optional[str] = None
    stopping_reason: Optional[str] = None


class JobDetailResponse(BaseModel):
    """Complete detail bundle for a specific job."""
    job_id: str
    job_dir: str
    created_at: str
    status: str
    verdict: str
    model_inspection: ModelInspectionResponse
    dataset_inspection: DatasetInspectionResponse
    optimization_plan: OptimizationPlanResponse
    metrics: Dict[str, ProvenanceMetric[Any]] = {}
    candidates: List[CandidateSummary] = []
    pareto_frontier: List[Dict[str, Any]] = []
    stopping_reason: str = ""
    stopping_description: str = ""
    selected_candidate_id: Optional[str] = None
    report_markdown: str = ""
    available_artifacts: List[Dict[str, Any]] = []
    host_telemetry_status: str = "MEASURED"
    target_hardware_status: str = "PENDING"


class TelemetrySample(BaseModel):
    """Individual telemetry measurement sample."""
    timestamp: float
    process_cpu_percent: float
    system_cpu_percent: float
    process_ram_mb: float
    process_vms_mb: float
    system_ram_used_mb: float
    system_ram_percent: float


class TelemetryPhaseData(BaseModel):
    """Telemetry data for a specific phase."""
    phase_id: str
    phase_name: str
    model_state: str
    candidate_id: Optional[str] = None
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    throughput_ips: float
    batch_size: int
    warmup_runs: int
    measured_runs: int
    avg_cpu_percent: float
    peak_cpu_percent: float
    avg_ram_mb: float
    peak_ram_mb: float
    duration_sec: float
    samples: List[TelemetrySample] = []


class TelemetryResponse(BaseModel):
    """Telemetry response with environment metadata and time-series samples."""
    job_id: str
    environment_info: Dict[str, Any]
    target_hardware_status: str = "PENDING"
    host_validation_status: str = "MEASURED"
    baseline: Optional[TelemetryPhaseData] = None
    final: Optional[TelemetryPhaseData] = None
    phases: Dict[str, TelemetryPhaseData] = {}
