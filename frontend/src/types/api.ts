/**
 * UAQE Frontend TypeScript API Data Contracts
 * Enforces measurement provenance and rigorous type safety across all components.
 */

export type ProvenanceLabel =
  | 'DETECTED'
  | 'CONFIGURED'
  | 'MEASURED'
  | 'CALCULATED'
  | 'POLICY'
  | 'PENDING'
  | 'NOT_AVAILABLE';

export type InputSourceType = 'USER_UPLOAD' | 'PRE_VERIFIED_SAMPLE' | 'CLI_PATH';

export type UploadState =
  | 'IDLE'
  | 'SELECTED'
  | 'UPLOADING'
  | 'VERIFYING'
  | 'RECONSTRUCTING'
  | 'VALIDATING'
  | 'READY'
  | 'FAILED';

export interface ModelUploadResult {
  upload_id: string;
  filename: string;
  size_bytes: number;
  sha256: string;
  format: string;
  status: string;
  source: string;
  staged_path?: string;
}

export interface DatasetUploadResult {
  upload_id: string;
  folder_name: string;
  file_count: number;
  total_size_bytes: number;
  manifest_hash: string;
  detected_format: string;
  class_count: number;
  splits: Record<string, number>;
  status: string;
  source: string;
  staged_path?: string;
  structure_valid?: boolean;
  validation_message?: string;
}

export interface ProvenanceMetric<T = any> {
  value: T | null;
  unit: string;
  source: ProvenanceLabel;
  origin: string;
  description: string;
}


export interface SystemStatus {
  status: string;
  version: string;
  supported_model_formats: string[];
  supported_tasks: string[];
  supported_profiles: string[];
  supported_targets: TargetProfile[];
  host_environment: HostEnvironment;
}

export interface TargetProfile {
  target_id: string;
  name: string;
  hardware_class: string;
  ram_bytes?: number;
  sram_bytes?: number;
  flash_bytes?: number;
  supported_precisions: string[];
  preferred_export_format: string;
  max_model_size_bytes: number;
  is_physical_measurement: boolean;
  description: string;
}

export interface HostEnvironment {
  os: string;
  os_release?: string;
  architecture: string;
  processor?: string;
  cpu_count_logical: number;
  cpu_count_physical: number;
  total_ram_mb: number;
  python_version?: string;
  environment: string;
}

export interface ModelInspection {
  format: ProvenanceMetric<string>;
  framework: ProvenanceMetric<string>;
  architecture: ProvenanceMetric<string>;
  task: ProvenanceMetric<string>;
  input_shape: ProvenanceMetric<number[]>;
  output_shape: ProvenanceMetric<number[]>;
  parameter_count: ProvenanceMetric<number>;
  tensor_count: ProvenanceMetric<number>;
  file_size_bytes: ProvenanceMetric<number>;
  sha256: ProvenanceMetric<string>;
  dtype: ProvenanceMetric<string>;
  raw_descriptor: Record<string, any>;
}

export interface DatasetInspection {
  dataset_name: ProvenanceMetric<string>;
  detected_format: ProvenanceMetric<string>;
  adapter_type: ProvenanceMetric<string>;
  class_count: ProvenanceMetric<number>;
  class_names: ProvenanceMetric<string[]>;
  splits: ProvenanceMetric<Record<string, number>>;
  class_distribution: ProvenanceMetric<Record<string, Record<string, number>>>;
  raw_descriptor: Record<string, any>;
}

export interface OptimizationPlan {
  target_hardware: ProvenanceMetric<string | Record<string, any>>;
  optimization_profile: ProvenanceMetric<string>;
  max_candidates_budget: ProvenanceMetric<number>;
  max_allowed_accuracy_loss_pp: ProvenanceMetric<number>;
  safety_policy_excellent_pp: ProvenanceMetric<number>;
  safety_policy_acceptable_pp: ProvenanceMetric<number>;
  safety_policy_critical_pp: ProvenanceMetric<number>;
  objective_weights: ProvenanceMetric<Record<string, number>>;
  calib_samples: ProvenanceMetric<number>;
  test_samples: ProvenanceMetric<number>;
  raw_plan: Record<string, any>;
}

export interface CandidateArtifact {
  filename: string;
  format: string;
  size_bytes: number;
  sha256?: string;
  download_url: string;
  relative_path?: string;
}

export interface CandidateSummary {
  candidate_id: string;
  candidate_name: string;
  strategy_type: string;
  top1_accuracy: number;
  accuracy_loss_pp: number;
  safety_classification: 'EXCELLENT' | 'ACCEPTABLE' | 'CRITICAL' | string;
  model_size_bytes: number;
  size_reduction_percent: number;
  latency_mean_ms: number;
  latency_reduction_percent: number;
  throughput_ips: number;
  composite_score: number;
  is_satisfied: boolean;
  is_critical: boolean;
  rejection_reason: string | null;
  action_taken: string;
  artifact_metadata: Record<string, any>;
  benchmark_provenance?: Record<string, any>;
  artifact?: CandidateArtifact;
}

export interface JobSummary {
  job_id: string;
  created_at: string;
  model_name: string;
  dataset_name: string;
  target_hardware: string;
  optimization_profile: string;
  status: string;
  verdict: 'VERIFIED' | 'VERIFIED WITH CAVEATS' | 'NOT VERIFIED' | string;
  fp32_accuracy: number | null;
  final_accuracy: number | null;
  accuracy_loss_pp: number | null;
  accuracy_classification: string | null;
  original_size_bytes: number | null;
  optimized_size_bytes: number | null;
  size_reduction_percent: number | null;
  fp32_latency_ms: number | null;
  optimized_latency_ms: number | null;
  latency_change_percent: number | null;
  candidates_evaluated: number;
  selected_candidate_name: string | null;
  stopping_reason: string | null;
}

export interface JobArtifact {
  filename: string;
  relative_path: string;
  size_bytes: number;
}

export interface JobDetail {
  job_id: string;
  job_dir: string;
  created_at: string;
  status: string;
  verdict: string;
  model_inspection: ModelInspection;
  dataset_inspection: DatasetInspection;
  optimization_plan: OptimizationPlan;
  metrics: Record<string, ProvenanceMetric>;
  candidates: CandidateSummary[];
  pareto_frontier: any[];
  stopping_reason: string;
  stopping_description: string;
  selected_candidate_id: string | null;
  report_markdown: string;
  available_artifacts: JobArtifact[];
  host_telemetry_status: string;
  target_hardware_status: string;
}

export interface TelemetrySample {
  timestamp: number;
  process_cpu_percent: number;
  system_cpu_percent: number;
  process_ram_mb: number;
  process_vms_mb: number;
  system_ram_used_mb: number;
  system_ram_percent: number;
}

export interface TelemetryPhaseData {
  phase_id: string;
  phase_name: string;
  model_state: string;
  candidate_id?: string | null;
  latency_mean_ms: number;
  latency_median_ms: number;
  latency_p95_ms: number;
  throughput_ips: number;
  batch_size: number;
  warmup_runs: number;
  measured_runs: number;
  avg_cpu_percent: number;
  peak_cpu_percent: number;
  avg_ram_mb: number;
  peak_ram_mb: number;
  duration_sec: number;
  samples: TelemetrySample[];
}

export interface TelemetryResponse {
  job_id: string;
  environment_info: HostEnvironment;
  target_hardware_status: string;
  host_validation_status: string;
  baseline: TelemetryPhaseData | null;
  final: TelemetryPhaseData | null;
  phases: Record<string, TelemetryPhaseData>;
}

export interface OptimizationEvent {
  type: string;
  job_id?: string;
  timestamp?: number;
  stage?: string;
  message?: string;
  candidate_id?: string;
  candidate_name?: string;
  result?: Record<string, any>;
  candidate?: Record<string, any>;
  is_satisfied?: boolean;
  stopping_reason?: string;
  stopping_desc?: string;
  plan?: Record<string, any>;
  results?: Record<string, any>;
  metrics?: Record<string, any>;
  verdict?: string;
  error?: string;
  // Live telemetry fields
  cpu_system_pct?: number;
  cpu_process_pct?: number;
  ram_used_pct?: number;
  ram_process_mb?: number;
  ram_system_used_mb?: number;
  ram_system_available_mb?: number;
  peak_ram_process_mb?: number;
  cpu_count_logical?: number;
  cpu_frequency_current_mhz?: number;
  phase?: string;
}
