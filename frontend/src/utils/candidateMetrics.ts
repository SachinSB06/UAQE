import type { JobDetail, CandidateSummary, TelemetryResponse } from '../types/api';
import { safeNumber, safeString } from './candidateNormalizer.ts';

export interface CanonicalArtifact {
  filename: string;
  format: string;
  formatLabel: string;
  sizeBytes: number | null;
  sizeMB: number | null;
  sha256: string | null;
  downloadUrl: string;
  relativePath: string | null;
}

export interface ActiveCandidateMetrics {
  candidateId: string | null;
  candidateName: string;
  strategyName: string;
  strategyType: string;
  isSatisfied: boolean;
  isCritical: boolean;
  safetyClassification: string;
  rejectionReason: string | null;
  actionTaken: string;

  // Accuracy
  fp32Accuracy: number | null;
  optAccuracy: number | null;
  accuracyLossPp: number | null;
  accuracyDeltaPp: number | null;

  // Footprint
  fp32SizeBytes: number | null;
  fp32SizeMB: number | null;
  optSizeBytes: number | null;
  optSizeMB: number | null;
  sizeReductionPercent: number | null;

  // Latency & Speedup
  fp32LatencyMs: number | null;
  optLatencyMs: number | null;
  optLatencyP95Ms: number | null;
  fp32ThroughputIps: number | null;
  optThroughputIps: number | null;
  speedupPercent: number | null;

  // Telemetry & Resource Efficiency (Host CPU & RAM)
  cpuAvgPct: number | null;
  cpuPeakPct: number | null;
  ramAvgMb: number | null;
  ramPeakMb: number | null;
  executionSamples: any[];

  // Benchmark Provenance
  runtime: string | null;
  delegate: string | null;
  delegatedOps: number | null;
  fallbackOps: number | null;
  threads: number;

  // Canonical Artifact
  artifact: CanonicalArtifact;
}

/**
 * Single source of truth selector helper for candidate-dependent UI metrics.
 * Eliminates metric mixing between active candidate, winner, and stale job-level fallbacks.
 */
export function getActiveCandidateMetrics(
  jobDetail: JobDetail | null,
  activeCandidate: CandidateSummary | null,
  telemetry: TelemetryResponse | null
): ActiveCandidateMetrics {
  const candId = activeCandidate?.candidate_id ?? null;
  const candName = safeString(activeCandidate?.candidate_name, 'Candidate');
  const stratType = safeString(activeCandidate?.strategy_type, '');
  const stratName = activeCandidate?.candidate_name || stratType || 'Quantization Candidate';

  // 1. Accuracy
  const fp32Acc = jobDetail?.metrics?.fp32_accuracy?.value != null
    ? safeNumber(jobDetail.metrics.fp32_accuracy.value) !== null
      ? (safeNumber(jobDetail.metrics.fp32_accuracy.value)! * 100)
      : null
    : null;

  const optAcc = activeCandidate != null
    ? (safeNumber(activeCandidate.top1_accuracy, null) !== null
        ? safeNumber(activeCandidate.top1_accuracy)! * 100
        : null)
    : (jobDetail?.metrics?.optimized_accuracy?.value != null
        ? safeNumber(jobDetail.metrics.optimized_accuracy.value)! * 100
        : null);

  const accuracyLossPp = activeCandidate != null
    ? safeNumber(activeCandidate.accuracy_loss_pp, null)
    : (jobDetail?.metrics?.accuracy_loss_pp?.value != null
        ? safeNumber(jobDetail.metrics.accuracy_loss_pp.value, null)
        : (fp32Acc !== null && optAcc !== null ? fp32Acc - optAcc : null));

  const accuracyDeltaPp = optAcc !== null && fp32Acc !== null ? optAcc - fp32Acc : null;

  // 2. Footprint
  const fp32SizeBytes = jobDetail?.metrics?.original_size_bytes?.value != null
    ? safeNumber(jobDetail.metrics.original_size_bytes.value, null)
    : (jobDetail?.model_inspection?.file_size_bytes?.value != null
        ? safeNumber(jobDetail.model_inspection.file_size_bytes.value, null)
        : null);

  const fp32SizeMB = fp32SizeBytes != null ? fp32SizeBytes / (1024 * 1024) : null;

  const optSizeBytes = activeCandidate != null
    ? safeNumber(activeCandidate.model_size_bytes, null)
    : (jobDetail?.metrics?.optimized_size_bytes?.value != null
        ? safeNumber(jobDetail.metrics.optimized_size_bytes.value, null)
        : null);

  const optSizeMB = optSizeBytes != null ? optSizeBytes / (1024 * 1024) : null;

  const sizeReductionPercent = activeCandidate != null
    ? safeNumber(activeCandidate.size_reduction_percent, null)
    : (jobDetail?.metrics?.storage_reduction_percent?.value != null
        ? safeNumber(jobDetail.metrics.storage_reduction_percent.value, null)
        : null);

  // 3. Latency & Throughput
  const fp32LatencyMs = jobDetail?.metrics?.fp32_latency_ms?.value != null
    ? safeNumber(jobDetail.metrics.fp32_latency_ms.value, null)
    : null;

  const optLatencyMs = activeCandidate != null
    ? safeNumber(activeCandidate.latency_mean_ms, null)
    : (jobDetail?.metrics?.optimized_latency_ms?.value != null
        ? safeNumber(jobDetail.metrics.optimized_latency_ms.value, null)
        : null);

  const fp32ThroughputIps = jobDetail?.metrics?.fp32_throughput_img_s?.value != null
    ? safeNumber(jobDetail.metrics.fp32_throughput_img_s.value, null)
    : (fp32LatencyMs && fp32LatencyMs > 0 ? 1000 / fp32LatencyMs : null);

  const optThroughputIps = activeCandidate != null
    ? safeNumber(activeCandidate.throughput_ips, null)
    : (jobDetail?.metrics?.throughput_images_per_sec?.value != null
        ? safeNumber(jobDetail.metrics.throughput_images_per_sec.value, null)
        : (optLatencyMs && optLatencyMs > 0 ? 1000 / optLatencyMs : null));

  const speedupPercent = (fp32LatencyMs && optLatencyMs && fp32LatencyMs > 0)
    ? ((fp32LatencyMs - optLatencyMs) / fp32LatencyMs) * 100
    : null;

  // 4. Telemetry & Hardware Resource Efficiency (Strict Active Candidate Binding)
  // Look up candidate phase in telemetry.phases[candId] or artifact_metadata.telemetry.summary
  const candPhase = candId && telemetry?.phases ? telemetry.phases[candId] : null;
  const telemSummary = activeCandidate?.artifact_metadata?.telemetry?.summary || null;

  const cpuAvgPct = safeNumber(
    candPhase?.avg_cpu_percent ?? telemSummary?.avg_cpu_percent ?? telemSummary?.cpu_avg_pct,
    null
  );

  const cpuPeakPct = safeNumber(
    candPhase?.peak_cpu_percent ?? telemSummary?.peak_cpu_percent ?? telemSummary?.cpu_peak_pct,
    null
  );

  const ramAvgMb = safeNumber(
    candPhase?.avg_ram_mb ?? telemSummary?.avg_ram_mb ?? telemSummary?.ram_avg_mb,
    null
  );

  const ramPeakMb = safeNumber(
    candPhase?.peak_ram_mb ?? telemSummary?.peak_ram_mb ?? telemSummary?.ram_peak_mb,
    null
  );

  const optLatencyP95Ms = safeNumber(
    candPhase?.latency_p95_ms ?? activeCandidate?.artifact_metadata?.latency_p95_ms,
    null
  );

  const diskTel = telemetry as any;
  const executionSamples = (candPhase?.samples && candPhase.samples.length > 0)
    ? candPhase.samples
    : (activeCandidate?.artifact_metadata?.telemetry?.samples || diskTel?.execution_samples || []);

  // 5. Benchmark Provenance
  const candProv = activeCandidate?.benchmark_provenance || null;
  const isXnnpack = stratType === 'XNNPACK_COMPATIBLE_INT8' || activeCandidate?.artifact_metadata?.delegate === 'XNNPACK';

  const runtime = candProv?.runtime
    ?? (isXnnpack
        ? 'TensorFlow Lite (XNNPACK)'
        : (activeCandidate ? 'TensorFlow Lite (BUILTIN_WITHOUT_DEFAULT_DELEGATES)' : (jobDetail?.metrics?.benchmark_provenance?.value?.runtime ?? null)));

  const delegate = candProv?.delegate
    ?? (isXnnpack
        ? 'XNNPACK'
        : (activeCandidate ? 'REFERENCE' : (jobDetail?.metrics?.benchmark_provenance?.value?.delegate ?? null)));

  const delegatedOps = activeCandidate?.artifact_metadata?.delegated_operator_count
    ?? candProv?.delegate_supported_ops?.length
    ?? null;

  const fallbackOps = activeCandidate?.artifact_metadata?.fallback_operator_count
    ?? candProv?.delegate_fallback_ops?.length
    ?? null;

  const threads = safeNumber(candProv?.num_threads ?? activeCandidate?.artifact_metadata?.threads, 2) ?? 2;

  // 6. Canonical Artifact Metadata (Strict Candidate Binding)
  const jobId = jobDetail?.job_id || '';
  let artifactFilename = activeCandidate?.artifact?.filename;
  let artifactFormat = activeCandidate?.artifact?.format;
  let artifactSha256 = activeCandidate?.artifact?.sha256
    ?? activeCandidate?.artifact_metadata?.candidate_artifact_sha256
    ?? candProv?.artifact_sha256
    ?? null;
  let artifactSizeBytes = activeCandidate?.artifact?.size_bytes ?? optSizeBytes;
  let artifactDownloadUrl = activeCandidate?.artifact?.download_url;
  let artifactRelativePath = activeCandidate?.artifact?.relative_path ?? null;

  if (!artifactFilename) {
    // Dynamic fallback based on candidate model path or architecture
    if (activeCandidate?.strategy_type?.toLowerCase().includes('onnx')) {
      artifactFilename = 'optimized_model.onnx';
      artifactFormat = 'onnx';
    } else {
      artifactFilename = 'optimized_model.tflite';
      artifactFormat = 'tflite';
    }
  }

  if (!artifactFormat) {
    artifactFormat = artifactFilename.split('.').pop()?.toLowerCase() || 'bin';
  }

  if (!artifactDownloadUrl || artifactDownloadUrl.includes('/api/jobs//')) {
    artifactDownloadUrl = candId && jobId
      ? `/api/jobs/${encodeURIComponent(jobId)}/candidates/${encodeURIComponent(candId)}/download`
      : jobId
      ? `/api/jobs/${encodeURIComponent(jobId)}/download/${encodeURIComponent(artifactFilename)}`
      : '';
  }

  const formatUpper = artifactFormat.toUpperCase();
  const formatLabel = formatUpper === 'TFLITE' ? 'TFLITE INT8' : formatUpper === 'ONNX' ? 'ONNX INT8' : `${formatUpper} INT8`;

  return {
    candidateId: candId,
    candidateName: candName,
    strategyName: stratName,
    strategyType: stratType,
    isSatisfied: Boolean(activeCandidate?.is_satisfied),
    isCritical: Boolean(activeCandidate?.is_critical),
    safetyClassification: safeString(activeCandidate?.safety_classification, 'UNKNOWN'),
    rejectionReason: activeCandidate?.rejection_reason ?? null,
    actionTaken: activeCandidate?.action_taken || '',

    fp32Accuracy: fp32Acc,
    optAccuracy: optAcc,
    accuracyLossPp,
    accuracyDeltaPp,

    fp32SizeBytes,
    fp32SizeMB,
    optSizeBytes,
    optSizeMB,
    sizeReductionPercent,

    fp32LatencyMs,
    optLatencyMs,
    optLatencyP95Ms,
    fp32ThroughputIps,
    optThroughputIps,
    speedupPercent,

    cpuAvgPct,
    cpuPeakPct,
    ramAvgMb,
    ramPeakMb,
    executionSamples,

    runtime: safeString(runtime, null as any),
    delegate: safeString(delegate, null as any),
    delegatedOps: safeNumber(delegatedOps, null),
    fallbackOps: safeNumber(fallbackOps, null),
    threads,

    artifact: {
      filename: artifactFilename,
      format: artifactFormat,
      formatLabel,
      sizeBytes: artifactSizeBytes,
      sizeMB: artifactSizeBytes != null ? artifactSizeBytes / (1024 * 1024) : null,
      sha256: artifactSha256,
      downloadUrl: artifactDownloadUrl,
      relativePath: artifactRelativePath,
    },
  };
}
