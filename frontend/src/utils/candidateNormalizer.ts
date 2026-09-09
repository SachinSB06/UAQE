import type { CandidateSummary } from '../types/api';

/**
 * Safely extracts a numeric value from raw numbers, strings, or ProvenanceMetric wrappers.
 * Returns null if the value is null, undefined, or not a finite number.
 */
export function safeNumber(val: any, fallback: number | null = null): number | null {
  if (val === null || val === undefined) return fallback;
  const unwrapped = typeof val === 'object' && val !== null && 'value' in val ? val.value : val;
  if (unwrapped === null || unwrapped === undefined || unwrapped === '') return fallback;
  const num = typeof unwrapped === 'number' ? unwrapped : Number(unwrapped);
  return Number.isFinite(num) ? num : fallback;
}

/**
 * Formats a number with toFixed if finite; otherwise returns the fallback string (e.g. 'NOT AVAILABLE' or 'PENDING').
 */
export function formatNumber(
  val: any,
  digits: number = 2,
  fallback: string = 'NOT AVAILABLE'
): string {
  const num = safeNumber(val);
  if (num === null) return fallback;
  return num.toFixed(digits);
}

/**
 * Safely extracts a string value from strings or objects.
 */
export function safeString(val: any, fallback: string = ''): string {
  if (val === null || val === undefined) return fallback;
  const unwrapped = typeof val === 'object' && val !== null && 'value' in val ? val.value : val;
  if (unwrapped === null || unwrapped === undefined) return fallback;
  return String(unwrapped);
}

/**
 * Defensively normalizes candidate payload shapes without recalculating or altering any live metrics.
 * Preserves genuine backend numbers and strings while providing safe field shapes for rendering.
 */
export function normalizeCandidate(raw: any, index: number = 0): CandidateSummary {
  if (!raw || typeof raw !== 'object') {
    return {
      candidate_id: `cand_${index + 1}`,
      candidate_name: `Candidate ${index + 1}`,
      strategy_type: 'UNKNOWN',
      top1_accuracy: 0,
      accuracy_loss_pp: 0,
      safety_classification: 'CRITICAL',
      model_size_bytes: 0,
      size_reduction_percent: 0,
      latency_mean_ms: 0,
      latency_reduction_percent: 0,
      throughput_ips: 0,
      composite_score: 0,
      is_satisfied: false,
      is_critical: true,
      rejection_reason: 'Malformed or missing candidate payload',
      action_taken: 'Rejected by parser',
      artifact_metadata: {},
    };
  }

  // Preserve all original properties in case subcomponents access them
  const base = { ...raw };

  const candidate_id = safeString(raw.candidate_id, `cand_${index + 1}`);
  const candidate_name = safeString(raw.candidate_name, `Candidate ${index + 1}`);
  const strategy_type = safeString(raw.strategy_type, 'Adaptive Quantization');

  const top1_accuracy = safeNumber(raw.top1_accuracy, 0) ?? 0;
  const accuracy_loss_pp = safeNumber(raw.accuracy_loss_pp, 0) ?? 0;
  const safety_classification = safeString(raw.safety_classification, 'EXCELLENT');

  const model_size_bytes = safeNumber(raw.model_size_bytes, 0) ?? 0;

  // Size reduction representation:
  // If size_reduction_percent is explicitly provided (e.g. from /api/jobs/{id}), use it.
  // Otherwise, if size_reduction ratio is provided (e.g. 0.6969 from CandidateEvaluationResult), scale by 100 for display percentage.
  let size_reduction_percent: number = 0;
  if (raw.size_reduction_percent != null) {
    size_reduction_percent = safeNumber(raw.size_reduction_percent, 0) ?? 0;
  } else if (raw.size_reduction != null) {
    const ratio = safeNumber(raw.size_reduction, 0) ?? 0;
    size_reduction_percent = ratio * 100.0;
  }

  const latency_mean_ms = safeNumber(raw.latency_mean_ms, 0) ?? 0;

  // Latency reduction representation:
  let latency_reduction_percent: number = 0;
  if (raw.latency_reduction_percent != null) {
    latency_reduction_percent = safeNumber(raw.latency_reduction_percent, 0) ?? 0;
  } else if (raw.latency_reduction != null) {
    latency_reduction_percent = safeNumber(raw.latency_reduction, 0) ?? 0;
  }

  const throughput_ips = safeNumber(raw.throughput_ips, 0) ?? 0;
  const composite_score = safeNumber(raw.composite_score, 0) ?? 0;

  const is_satisfied = Boolean(raw.is_satisfied);
  const is_critical = Boolean(raw.is_critical);

  let rejection_reason: string | null = raw.rejection_reason ?? null;
  if (!rejection_reason && is_critical) {
    rejection_reason = `Exceeded safety threshold (${accuracy_loss_pp.toFixed(2)} pp loss > safety limit)`;
  } else if (!rejection_reason && !is_satisfied) {
    rejection_reason = `Did not meet target profile constraint (${accuracy_loss_pp.toFixed(2)} pp loss)`;
  }

  let action_taken: string = raw.action_taken ?? (is_satisfied && !is_critical ? 'Selected as Optimal Candidate' : 'Gated by Policy');

  let artifact = raw.artifact ?? base.artifact ?? undefined;
  if (!artifact && raw.model_path) {
    const rawFname = String(raw.model_path).split(/[\\/]/).pop() || 'model.bin';
    const ext = rawFname.includes('.') ? rawFname.split('.').pop()?.toLowerCase() || 'bin' : 'bin';
    const jobIdStr = safeString(raw.job_id, '');
    artifact = {
      filename: rawFname,
      format: ext,
      size_bytes: model_size_bytes,
      sha256: raw.artifact_metadata?.candidate_artifact_sha256 ?? raw.benchmark_provenance?.artifact_sha256,
      download_url: jobIdStr ? `/api/jobs/${encodeURIComponent(jobIdStr)}/candidates/${encodeURIComponent(candidate_id)}/download` : undefined,
      relative_path: `candidates/${candidate_id}/${rawFname}`,
    };
  }

  return {
    ...base,
    candidate_id,
    candidate_name,
    strategy_type,
    top1_accuracy,
    accuracy_loss_pp,
    safety_classification,
    model_size_bytes,
    size_reduction_percent,
    latency_mean_ms,
    latency_reduction_percent,
    throughput_ips,
    composite_score,
    is_satisfied,
    is_critical,
    rejection_reason,
    action_taken,
    artifact_metadata: raw.artifact_metadata || {},
    benchmark_provenance: raw.benchmark_provenance ?? base.benchmark_provenance,
    artifact,
  };
}
