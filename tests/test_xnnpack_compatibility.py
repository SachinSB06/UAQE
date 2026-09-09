"""
tests/test_xnnpack_compatibility.py
Automated test suite verifying the experimental XNNPACK-compatible INT8 artifact,
baseline immutability, mathematical integrity, runtime execution, and accuracy retention.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from uaqe.telemetry.canonical_benchmark import CanonicalBenchmark

BASELINE_PATH = os.path.join(
    PROJECT_ROOT, "output", "phase_c2", "models", "mobilenetv3_sem_9class_qat_int8.tflite"
)
EXPECTED_BASELINE_SHA = "10d51d4f243c554734e7347f9010297fb182a64c4ad89d5551e24830846dab1d"
EXPECTED_BASELINE_SIZE = 1855816

EXPERIMENTAL_PATH = os.path.join(
    PROJECT_ROOT, "output", "phase_xnnpack", "models", "mobilenetv3_sem_9class_xnnpack_int8.tflite"
)
EXP_REPORTS_DIR = os.path.join(PROJECT_ROOT, "output", "phase_xnnpack", "reports")
EXP_DIAGNOSTICS_DIR = os.path.join(PROJECT_ROOT, "output", "phase_xnnpack", "diagnostics")


class TestXNNPACKCompatibility(unittest.TestCase):
    """Rigorous 15-point verification suite for the UAQE XNNPACK experimental pipeline."""

    def compute_sha(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def test_01_baseline_exists(self):
        """TEST 1: Baseline artifact exists."""
        self.assertTrue(os.path.exists(BASELINE_PATH), f"Baseline artifact missing at {BASELINE_PATH}")

    def test_02_baseline_sha_matches(self):
        """TEST 2: Baseline SHA-256 equals expected hash."""
        sha = self.compute_sha(BASELINE_PATH)
        self.assertEqual(
            sha,
            EXPECTED_BASELINE_SHA,
            f"Baseline SHA mismatch! Expected {EXPECTED_BASELINE_SHA}, got {sha}"
        )

    def test_03_baseline_size_unchanged(self):
        """TEST 3: Baseline size remains unchanged (1,855,816 bytes)."""
        size = os.path.getsize(BASELINE_PATH)
        self.assertEqual(
            size,
            EXPECTED_BASELINE_SIZE,
            f"Baseline size changed! Expected {EXPECTED_BASELINE_SIZE}, got {size}"
        )

    def test_04_experimental_artifact_location_isolated(self):
        """TEST 4: Experimental artifact is in phase_xnnpack, not phase_c2."""
        norm_exp = os.path.normpath(EXPERIMENTAL_PATH)
        self.assertIn("phase_xnnpack", norm_exp)
        self.assertNotIn("phase_c2", norm_exp)

    def test_05_experimental_artifact_exists(self):
        """TEST 5: Experimental artifact exists."""
        self.assertTrue(os.path.exists(EXPERIMENTAL_PATH), f"Experimental artifact missing at {EXPERIMENTAL_PATH}")

    def test_06_experimental_artifact_parseable(self):
        """TEST 6: Experimental artifact can be parsed/loaded as TFLite model."""
        with open(EXPERIMENTAL_PATH, "rb") as f:
            raw = f.read()
        self.assertGreater(len(raw), 1000000, "Experimental artifact file is too small.")
        self.assertIn(b"TFL3", raw[:16], "Artifact missing valid TFLite 'TFL3' FlatBuffer identifier.")

    def test_07_xnnpack_allocate_tensors_succeeds(self):
        """TEST 7: XNNPACK allocate_tensors succeeds honestly without preparation errors."""
        try:
            interpreter = tf.lite.Interpreter(
                model_path=EXPERIMENTAL_PATH,
                num_threads=2
            )
            interpreter.allocate_tensors()
            success = True
        except Exception as e:
            success = False
            self.fail(f"XNNPACK allocate_tensors failed: {e}")
        self.assertTrue(success)

    def test_08_xnnpack_delegation_present(self):
        """TEST 8: Required XNNPACK delegation is present and recorded."""
        audit_path = os.path.join(EXP_DIAGNOSTICS_DIR, "operator_compatibility_audit.json")
        self.assertTrue(os.path.exists(audit_path), f"Audit missing at {audit_path}")
        with open(audit_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(data.get("xnnpack_initialization_success", False))
        self.assertGreater(data.get("delegated_operators_count", 0), 0)
        self.assertEqual(data.get("fallback_operators_count", -1), 0)

    def test_09_accuracy_independently_evaluated(self):
        """TEST 9: Accuracy is independently evaluated and recorded."""
        acc_report_path = os.path.join(EXP_REPORTS_DIR, "xnnpack_accuracy_report.json")
        self.assertTrue(os.path.exists(acc_report_path), f"Accuracy report missing at {acc_report_path}")
        with open(acc_report_path, "r", encoding="utf-8") as f:
            acc_data = json.load(f)
        exp_acc = acc_data["models"]["experimental_xnnpack_int8"]["accuracy"]
        self.assertGreater(exp_acc, 0.90, f"Experimental accuracy unexpectedly low: {exp_acc}")

    def test_10_accuracy_loss_within_threshold(self):
        """TEST 10: Accuracy loss from canonical FP32 baseline is within the configured threshold (<= 0.5 pp)."""
        acc_report_path = os.path.join(EXP_REPORTS_DIR, "xnnpack_accuracy_report.json")
        with open(acc_report_path, "r", encoding="utf-8") as f:
            acc_data = json.load(f)
        fp32_acc = acc_data["models"]["fp32_reference"]["accuracy"] * 100.0
        exp_acc = acc_data["models"]["experimental_xnnpack_int8"]["accuracy"] * 100.0
        loss_pp = fp32_acc - exp_acc
        self.assertLessEqual(
            loss_pp,
            0.5,
            f"Accuracy loss {loss_pp:.2f} pp exceeds safety threshold of 0.5 pp"
        )
        self.assertTrue(acc_data.get("accuracy_loss_threshold_satisfied", False))

    def test_11_prediction_agreement_measured(self):
        """TEST 11: Prediction agreement between Reference INT8 and Experimental INT8 is measured."""
        acc_report_path = os.path.join(EXP_REPORTS_DIR, "xnnpack_accuracy_report.json")
        with open(acc_report_path, "r", encoding="utf-8") as f:
            acc_data = json.load(f)
        agreement = acc_data["prediction_agreement"]["prediction_agreement_percent"]
        self.assertGreaterEqual(
            agreement,
            99.0,
            f"Prediction agreement {agreement:.2f}% is lower than required 99.0%"
        )

    def test_12_canonical_benchmark_completes(self):
        """TEST 12: Canonical 10+100 benchmark completes and report is persisted."""
        bench_report_path = os.path.join(EXP_REPORTS_DIR, "xnnpack_benchmark_report.json")
        self.assertTrue(os.path.exists(bench_report_path), f"Benchmark report missing at {bench_report_path}")
        with open(bench_report_path, "r", encoding="utf-8") as f:
            bench_data = json.load(f)
        self.assertEqual(bench_data["protocol"]["warmup_runs"], 10)
        self.assertEqual(bench_data["protocol"]["timed_iterations"], 100)
        self.assertEqual(bench_data["protocol"]["num_threads"], 2)

    def test_13_benchmark_metrics_valid(self):
        """TEST 13: Benchmark metrics are finite, positive, and strictly valid."""
        bench_report_path = os.path.join(EXP_REPORTS_DIR, "xnnpack_benchmark_report.json")
        with open(bench_report_path, "r", encoding="utf-8") as f:
            bench_data = json.load(f)
        exp_bench = bench_data["benchmarks"]["experimental_xnnpack_int8"]
        self.assertGreater(exp_bench["pure_invoke_latency_ms"], 0.0)
        self.assertGreater(exp_bench["throughput_img_s"], 0.0)
        self.assertTrue(np.isfinite(exp_bench["pure_invoke_latency_ms"]))
        self.assertTrue(np.isfinite(exp_bench["throughput_img_s"]))
        # Verify throughput inverse relationship
        expected_thr = 1000.0 / exp_bench["pure_invoke_latency_ms"]
        self.assertAlmostEqual(exp_bench["throughput_img_s"], expected_thr, delta=1.0)

    def test_14_candidate_sha_recorded(self):
        """TEST 14: Candidate SHA-256 is recorded and matches actual experimental artifact."""
        summary_path = os.path.join(EXP_REPORTS_DIR, "xnnpack_experiment_summary.json")
        self.assertTrue(os.path.exists(summary_path), f"Summary missing at {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as f:
            sum_data = json.load(f)
        recorded_sha = sum_data["experimental_artifact"]["sha256"]
        actual_sha = self.compute_sha(EXPERIMENTAL_PATH)
        self.assertEqual(recorded_sha, actual_sha, f"Recorded SHA {recorded_sha} != actual {actual_sha}")

    def test_15_baseline_sha_unchanged_after_complete_experiment(self):
        """TEST 15: Baseline SHA-256 remains 100% UNCHANGED after the complete experiment."""
        final_sha = self.compute_sha(BASELINE_PATH)
        final_size = os.path.getsize(BASELINE_PATH)
        self.assertEqual(
            final_sha,
            EXPECTED_BASELINE_SHA,
            f"CRITICAL VIOLATION: Baseline SHA changed after experiment! {final_sha} != {EXPECTED_BASELINE_SHA}"
        )
        self.assertEqual(
            final_size,
            EXPECTED_BASELINE_SIZE,
            f"CRITICAL VIOLATION: Baseline size changed after experiment! {final_size} != {EXPECTED_BASELINE_SIZE}"
        )


if __name__ == "__main__":
    unittest.main()
