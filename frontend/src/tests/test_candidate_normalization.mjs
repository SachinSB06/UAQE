import assert from 'node:assert/strict';
import {
  safeNumber,
  formatNumber,
  safeString,
  normalizeCandidate,
} from '../utils/candidateNormalizer.ts';

console.log('--- Starting Candidate Normalization & Safety Regression Tests (Forms A-J) ---');

// Form A: Complete candidate result (standard backend format)
const formA = {
  candidate_id: 'cand_001',
  candidate_name: 'MobileNetV3 Static INT8 PTQ',
  strategy_type: 'mobilenet_adaptive',
  model_path: '/path/to/model.tflite',
  top1_accuracy: 0.94,
  macro_f1: 0.94,
  model_size_bytes: 1855816,
  latency_mean_ms: 12.5,
  latency_median_ms: 12.4,
  latency_p95_ms: 13.0,
  throughput_ips: 80.0,
  prediction_agreement: 100.0,
  accuracy_loss_pp: 0.0,
  safety_classification: 'EXCELLENT',
  is_satisfied: true,
  is_critical: false,
  composite_score: 0.95,
  raw_score: 0.95,
  size_reduction: 0.6969,
  latency_reduction: -16.7943,
  execution_duration_sec: 10.98,
};

const normA = normalizeCandidate(formA);
assert.equal(normA.candidate_id, 'cand_001');
assert.equal(normA.size_reduction_percent, 69.69);
assert.equal(formatNumber(normA.size_reduction_percent, 1), '69.7');
assert.equal(normA.latency_mean_ms, 12.5);
assert.equal(formatNumber(normA.latency_mean_ms, 1), '12.5');
assert.equal(normA.is_satisfied, true);
console.log('✓ Form A (Complete Candidate Result) passed.');

// Form B: Missing optional metric (e.g. no latency_p95_ms, no size_reduction)
const formB = {
  candidate_id: 'cand_002',
  candidate_name: 'Partial Candidate',
  strategy_type: 'INT8',
  top1_accuracy: 0.85,
  accuracy_loss_pp: 1.5,
};

const normB = normalizeCandidate(formB);
assert.equal(normB.candidate_id, 'cand_002');
assert.equal(normB.size_reduction_percent, 0);
assert.equal(formatNumber(normB.size_reduction_percent, 1), '0.0');
assert.equal(normB.latency_mean_ms, 0);
assert.equal(normB.composite_score, 0);
console.log('✓ Form B (Missing Optional Metric) passed.');

// Form C: Null metric (fields explicitly set to null)
const formC = {
  candidate_id: 'cand_003',
  candidate_name: 'Null Metrics Candidate',
  strategy_type: null,
  top1_accuracy: null,
  accuracy_loss_pp: null,
  model_size_bytes: null,
  size_reduction: null,
  size_reduction_percent: null,
  latency_mean_ms: null,
  throughput_ips: null,
  composite_score: null,
  safety_classification: null,
};

const normC = normalizeCandidate(formC);
assert.equal(normC.candidate_id, 'cand_003');
assert.equal(formatNumber(normC.top1_accuracy, 2), '0.00');
assert.equal(formatNumber(normC.size_reduction_percent, 1), '0.0');
assert.equal(formatNumber(normC.latency_mean_ms, 1), '0.0');
assert.equal(formatNumber(normC.composite_score, 3), '0.000');
assert.equal(safeString(normC.safety_classification), 'EXCELLENT');
console.log('✓ Form C (Null Metric) passed.');

// Form D: Raw numeric metric (direct numbers, integers and floats)
const formD = {
  candidate_id: 'cand_004',
  candidate_name: 'Raw Numbers',
  top1_accuracy: 0.99,
  accuracy_loss_pp: 0.05,
  model_size_bytes: 2000000,
  size_reduction: 0.5,
  latency_mean_ms: 10.0,
  throughput_ips: 100.0,
  composite_score: 0.99,
};

const normD = normalizeCandidate(formD);
assert.equal(normD.size_reduction_percent, 50.0);
assert.equal(formatNumber(normD.size_reduction_percent, 1), '50.0');
assert.equal(normD.top1_accuracy, 0.99);
console.log('✓ Form D (Raw Numeric Metric) passed.');

// Form E: ProvenanceMetric metric (wrapped in { value, unit, source, origin })
const formE = {
  candidate_id: { value: 'cand_005' },
  candidate_name: { value: 'Wrapped Candidate' },
  strategy_type: { value: 'QDQ' },
  top1_accuracy: { value: 0.92, unit: 'ratio' },
  accuracy_loss_pp: { value: 0.8, unit: 'pp' },
  model_size_bytes: { value: 1500000, unit: 'bytes' },
  size_reduction: { value: 0.72, unit: 'ratio' },
  latency_mean_ms: { value: 14.2, unit: 'ms' },
  throughput_ips: { value: 70.4, unit: 'ips' },
  composite_score: { value: 0.88 },
  safety_classification: { value: 'ACCEPTABLE' },
};

const normE = normalizeCandidate(formE);
assert.equal(normE.candidate_id, 'cand_005');
assert.equal(normE.candidate_name, 'Wrapped Candidate');
assert.equal(normE.top1_accuracy, 0.92);
assert.equal(normE.accuracy_loss_pp, 0.8);
assert.equal(normE.size_reduction_percent, 72.0);
assert.equal(formatNumber(normE.size_reduction_percent, 1), '72.0');
assert.equal(normE.latency_mean_ms, 14.2);
assert.equal(normE.throughput_ips, 70.4);
assert.equal(normE.safety_classification, 'ACCEPTABLE');
console.log('✓ Form E (ProvenanceMetric Metric) passed.');

// Form F: Failed candidate
const formF = {
  candidate_id: 'cand_fail',
  candidate_name: 'Failed Candidate',
  strategy_type: 'FAILED',
  top1_accuracy: 0.1,
  accuracy_loss_pp: 80.0,
  safety_classification: 'CRITICAL',
  is_satisfied: false,
  is_critical: true,
  composite_score: 0.0,
};

const normF = normalizeCandidate(formF);
assert.equal(normF.is_satisfied, false);
assert.equal(normF.is_critical, true);
assert.equal(normF.safety_classification, 'CRITICAL');
assert.ok(normF.rejection_reason.includes('Exceeded safety threshold'));
console.log('✓ Form F (Failed Candidate) passed.');

// Form G: candidate_done SSE event wrapper
const formG = {
  type: 'candidate_done',
  stage: 'EVALUATING',
  candidate_id: 'cand_001',
  candidate_name: 'INT8 PTQ',
  result: {
    candidate_id: 'cand_001',
    candidate_name: 'INT8 PTQ',
    strategy_type: 'INT8_PTQ',
    top1_accuracy: 0.94,
    accuracy_loss_pp: 0.0,
    model_size_bytes: 1855816,
    size_reduction: 0.6969,
    latency_mean_ms: 111.97,
    throughput_ips: 8.93,
    composite_score: 0.6742,
    is_satisfied: true,
    is_critical: false,
    safety_classification: 'EXCELLENT',
  },
};

const normG = normalizeCandidate(formG.result);
assert.equal(normG.candidate_id, 'cand_001');
assert.equal(normG.size_reduction_percent, 69.69);
assert.equal(formatNumber(normG.size_reduction_percent, 1), '69.7');
console.log('✓ Form G (candidate_done Event) passed.');

// Form H: best_candidate_selected SSE event
const formH = {
  type: 'best_candidate_selected',
  stage: 'SELECTING',
  candidate_id: 'cand_001',
  candidate_name: 'INT8 PTQ',
  result: {
    candidate_id: 'cand_001',
    top1_accuracy: 0.94,
    accuracy_loss_pp: 0.0,
    size_reduction: 0.6969,
    latency_mean_ms: 111.97,
    composite_score: 0.6742,
    is_satisfied: true,
    is_critical: false,
  },
};

const normH = normalizeCandidate(formH.result);
assert.equal(normH.candidate_id, 'cand_001');
assert.equal(normH.size_reduction_percent, 69.69);
console.log('✓ Form H (best_candidate_selected Event) passed.');

// Form I: Malformed optional field (e.g. NaN, objects in unexpected places, booleans in string fields)
const formI = {
  candidate_id: 12345,
  candidate_name: null,
  strategy_type: undefined,
  top1_accuracy: NaN,
  accuracy_loss_pp: 'not-a-number',
  model_size_bytes: {},
  size_reduction: Infinity,
  latency_mean_ms: [],
  throughput_ips: 'invalid',
  composite_score: null,
  safety_classification: 999,
};

const normI = normalizeCandidate(formI);
assert.equal(normI.candidate_id, '12345');
assert.equal(normI.top1_accuracy, 0);
assert.equal(normI.accuracy_loss_pp, 0);
assert.equal(normI.size_reduction_percent, 0);
assert.equal(formatNumber(normI.size_reduction_percent, 1), '0.0');
assert.equal(normI.latency_mean_ms, 0);
assert.equal(normI.throughput_ips, 0);
assert.equal(normI.composite_score, 0);
assert.equal(normI.safety_classification, '999');
console.log('✓ Form I (Malformed Optional Field) passed.');

// Form J: Empty candidate array / null / empty object
const normJ1 = normalizeCandidate(null);
const normJ2 = normalizeCandidate(undefined);
const normJ3 = normalizeCandidate({});
assert.equal(normJ1.candidate_id, 'cand_1');
assert.equal(normJ2.candidate_id, 'cand_1');
assert.equal(normJ3.candidate_id, 'cand_1');
console.log('✓ Form J (Empty / Null Candidate) passed.');

console.log('--- ALL 10 CANDIDATE PAYLOAD REGRESSION TESTS PASSED (A to J) ---');
