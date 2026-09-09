import assert from 'node:assert/strict';
import { getActiveCandidateMetrics } from '../utils/candidateMetrics.ts';
import { normalizeCandidate } from '../utils/candidateNormalizer.ts';

console.log('--- Starting Candidate Metrics Selector & Switching Tests ---');

// Mock job fixture with 2 candidates (Job UAQE-20260909-191140-11AA370D structure)
const mockJob = {
  job_id: 'UAQE-20260909-191140-11AA370D',
  model_id: 'mobilenetv3_large',
  metrics: {
    fp32_accuracy: { value: 0.94 },
    fp32_latency_ms: { value: 6.65 },
    fp32_throughput_img_s: { value: 150.31 },
    original_size_bytes: { value: 6123456 },
    optimized_accuracy: { value: 0.94 },
    optimized_latency_ms: { value: 149.91 }, // Winner (Candidate 1) snapshot
    throughput_images_per_sec: { value: 6.67 },
  },
  candidates: [
    {
      candidate_id: 'cand_001',
      candidate_name: 'Candidate 1 (Static INT8)',
      strategy_type: 'mobilenet_adaptive',
      top1_accuracy: 0.94,
      accuracy_loss_pp: 0.0,
      latency_mean_ms: 149.905,
      throughput_ips: 6.671,
      model_size_bytes: 2827080,
      is_satisfied: true,
      is_critical: false,
      artifact: {
        filename: 'optimized_model.tflite',
        format: 'tflite',
        size_bytes: 2827080,
        sha256: '10d51d4faaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        download_url: '/api/jobs/UAQE-20260909-191140-11AA370D/candidates/cand_001/download',
      },
      artifact_metadata: {
        telemetry: {
          summary: {
            avg_cpu_percent: 45.2,
            peak_cpu_percent: 52.0,
            avg_ram_mb: 412.5,
            peak_ram_mb: 430.0,
          },
        },
      },
    },
    {
      candidate_id: 'cand_004',
      candidate_name: 'Candidate 4 (XNNPACK INT8)',
      strategy_type: 'XNNPACK_COMPATIBLE_INT8',
      top1_accuracy: 0.94,
      accuracy_loss_pp: 0.0,
      latency_mean_ms: 9.818,
      throughput_ips: 101.86,
      model_size_bytes: 2827080,
      is_satisfied: true,
      is_critical: false,
      artifact: {
        filename: 'optimized_model.tflite',
        format: 'tflite',
        size_bytes: 2827080,
        sha256: '93a54e8fbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
        download_url: '/api/jobs/UAQE-20260909-191140-11AA370D/candidates/cand_004/download',
      },
      artifact_metadata: {
        telemetry: {
          summary: {
            avg_cpu_percent: 68.2,
            peak_cpu_percent: 75.0,
            avg_ram_mb: 449.66,
            peak_ram_mb: 470.0,
          },
        },
      },
    },
  ],
};

const mockTelemetry = {
  baseline: {
    latency_mean_ms: 6.65,
    avg_cpu_percent: 55.0,
    avg_ram_mb: 390.0,
  },
  final: {
    latency_mean_ms: 149.91,
    avg_cpu_percent: 45.2,
    avg_ram_mb: 412.5,
  },
  phases: {
    cand_001: {
      avg_cpu_percent: 45.2,
      peak_cpu_percent: 52.0,
      avg_ram_mb: 412.5,
      peak_ram_mb: 430.0,
    },
    cand_004: {
      avg_cpu_percent: 68.2,
      peak_cpu_percent: 75.0,
      avg_ram_mb: 449.66,
      peak_ram_mb: 470.0,
    },
  },
};

// Test 1: Candidate 1 Metrics
const cand1 = normalizeCandidate(mockJob.candidates[0]);
const metrics1 = getActiveCandidateMetrics(mockJob, cand1, mockTelemetry);
assert.equal(metrics1.candidateId, 'cand_001');
assert.equal(metrics1.optLatencyMs, 149.905);
assert.equal(metrics1.optThroughputIps, 6.671);
assert.equal(metrics1.cpuAvgPct, 45.2);
assert.equal(metrics1.ramAvgMb, 412.5);
assert.equal(metrics1.artifact.filename, 'optimized_model.tflite');
assert.equal(metrics1.artifact.format, 'tflite');
assert.equal(metrics1.artifact.downloadUrl, '/api/jobs/UAQE-20260909-191140-11AA370D/candidates/cand_001/download');
console.log('✓ Candidate 1 extraction passed.');

// Test 2: Candidate 4 Metrics (switching candidate MUST update all lower resource metrics)
const cand4 = normalizeCandidate(mockJob.candidates[1]);
const metrics4 = getActiveCandidateMetrics(mockJob, cand4, mockTelemetry);
assert.equal(metrics4.candidateId, 'cand_004');
assert.equal(metrics4.optLatencyMs, 9.818);
assert.equal(metrics4.optThroughputIps, 101.86);
assert.equal(metrics4.cpuAvgPct, 68.2);
assert.equal(metrics4.ramAvgMb, 449.66);
assert.equal(metrics4.artifact.filename, 'optimized_model.tflite');
assert.equal(metrics4.artifact.format, 'tflite');
assert.equal(metrics4.artifact.downloadUrl, '/api/jobs/UAQE-20260909-191140-11AA370D/candidates/cand_004/download');
console.log('✓ Candidate 4 extraction and switching passed (latency: 9.818ms, throughput: 101.86, CPU: 68.2%).');

// Test 3: Metric Isolation - metrics1 and metrics4 are completely independent
assert.notEqual(metrics1.optLatencyMs, metrics4.optLatencyMs);
assert.notEqual(metrics1.optThroughputIps, metrics4.optThroughputIps);
assert.notEqual(metrics1.cpuAvgPct, metrics4.cpuAvgPct);
assert.notEqual(metrics1.ramAvgMb, metrics4.ramAvgMb);
assert.notEqual(metrics1.artifact.downloadUrl, metrics4.artifact.downloadUrl);
console.log('✓ Strict metric isolation between candidates verified.');

// Test 4: Dynamic format detection for ONNX candidates
const onnxCand = normalizeCandidate({
  candidate_id: 'cand_onnx',
  candidate_name: 'ResNet INT8',
  model_path: '/models/quantized.onnx',
  top1_accuracy: 0.76,
  latency_mean_ms: 22.0,
  throughput_ips: 45.4,
  model_size_bytes: 25000000,
});
const onnxMetrics = getActiveCandidateMetrics(
  { job_id: 'JOB-ONNX', metrics: {} },
  onnxCand,
  null
);
assert.equal(onnxMetrics.artifact.format, 'onnx');
assert.equal(onnxMetrics.artifact.filename, 'quantized.onnx');
assert.equal(onnxMetrics.artifact.downloadUrl, '/api/jobs/JOB-ONNX/candidates/cand_onnx/download');
console.log('✓ ONNX candidate dynamic artifact detection passed.');

console.log('--- ALL CANDIDATE METRICS TESTS PASSED ---');
