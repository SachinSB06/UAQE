"""Test Suite for XNNPACK and Reference TFLite benchmark evaluation."""

import unittest
import os
import sys
import hashlib
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from uaqe.telemetry.canonical_benchmark import (
    CanonicalBenchmark,
    CanonicalBenchmarkResult,
    BenchmarkProvenance
)

ARTIFACT_PATH = "output/phase_c2/models/mobilenetv3_sem_9class_qat_int8.tflite"
EXPECTED_SHA = "10d51d4f243c554734e7347f9010297fb182a64c4ad89d5551e24830846dab1d"
EXPECTED_SIZE = 1855816


class TestXnnpackBenchmark(unittest.TestCase):
    """Rigorous tests for XNNPACK acceleration and Reference evaluation."""

    def test_01_artifact_sha_and_size_integrity(self):
        """Verify artifact SHA and byte size are completely unchanged."""
        self.assertTrue(os.path.exists(ARTIFACT_PATH), f"Artifact missing at {ARTIFACT_PATH}")
        with open(ARTIFACT_PATH, "rb") as f:
            data = f.read()
        sha = hashlib.sha256(data).hexdigest()
        self.assertEqual(sha, EXPECTED_SHA, f"SHA changed! Expected {EXPECTED_SHA}, got {sha}")
        self.assertEqual(len(data), EXPECTED_SIZE, f"Size changed! Expected {EXPECTED_SIZE}, got {len(data)}")

    def test_02_reference_tflite_path_validity(self):
        """Verify Reference TFLite initializes, runs pure invoke, and produces valid provenance."""
        result = CanonicalBenchmark.benchmark_tflite(
            model_path=ARTIFACT_PATH,
            input_shape=(1, 3, 128, 128),
            num_threads=2,
            warmup_runs=10,
            measured_runs=100,
            use_xnnpack=False
        )
        self.assertGreater(result.pure_invoke_latency_ms, 0)
        self.assertGreater(result.throughput_img_s, 0)
        self.assertAlmostEqual(result.throughput_img_s, 1000.0 / result.pure_invoke_latency_ms, delta=0.1)
        self.assertEqual(result.provenance.delegate, "REFERENCE")
        self.assertFalse(result.provenance.delegate_enabled)
        self.assertEqual(result.provenance.warmup_count, 10)
        self.assertEqual(result.provenance.timed_iterations, 100)
        self.assertEqual(result.provenance.num_threads, 2)
        self.assertEqual(result.provenance.artifact_sha256, EXPECTED_SHA)

    def test_03_xnnpack_path_diagnostics(self):
        """Verify that attempting to benchmark this artifact with XNNPACK reports the exact runtime error."""
        with self.assertRaises(RuntimeError) as ctx:
            CanonicalBenchmark.benchmark_tflite(
                model_path=ARTIFACT_PATH,
                input_shape=(1, 3, 128, 128),
                num_threads=2,
                warmup_runs=10,
                measured_runs=100,
                use_xnnpack=True
            )
        err_str = str(ctx.exception)
        self.assertTrue("XNNPACK" in err_str or "TfLiteXNNPackDelegate" in err_str)

    def test_04_provenance_schema_fields(self):
        """Verify BenchmarkProvenance includes delegate fields."""
        prov = BenchmarkProvenance(
            evaluator="test",
            runtime="test_rt",
            runtime_mode="performance",
            performance_enabled=True,
            num_threads=2,
            batch_size=1,
            input_shape=[1, 3, 128, 128],
            warmup_count=10,
            timed_iterations=100,
            delegate="REFERENCE",
            delegate_enabled=False
        )
        d = prov.to_dict()
        self.assertIn("delegate", d)
        self.assertIn("delegate_enabled", d)
        self.assertEqual(d["delegate"], "REFERENCE")
        self.assertFalse(d["delegate_enabled"])

    def test_05_throughput_formula_strict_enforcement(self):
        """Verify strict mathematical relationship throughput = 1000 / latency."""
        res = CanonicalBenchmarkResult(
            model_path=ARTIFACT_PATH,
            runtime="TensorFlow Lite (BUILTIN_WITHOUT_DEFAULT_DELEGATES)",
            num_threads=2,
            pure_invoke_latency_ms=80.0,
            p50_latency_ms=80.0,
            p95_latency_ms=100.0,
            min_latency_ms=75.0,
            max_latency_ms=110.0,
            throughput_img_s=12.5,
            preprocessing_latency_ms=0.1,
            postprocessing_latency_ms=0.1,
            end_to_end_latency_ms=80.2,
            warmup_runs=10,
            timed_iterations=100,
            provenance=BenchmarkProvenance(
                evaluator="test",
                runtime="test",
                runtime_mode="performance",
                performance_enabled=True,
                num_threads=2,
                batch_size=1,
                input_shape=[1, 3, 128, 128],
                warmup_count=10,
                timed_iterations=100,
                delegate="REFERENCE",
                delegate_enabled=False
            )
        )
        res.validate()  # Passes

        # Should raise if inconsistent
        res_bad = CanonicalBenchmarkResult(
            model_path=ARTIFACT_PATH,
            runtime="TensorFlow Lite (BUILTIN_WITHOUT_DEFAULT_DELEGATES)",
            num_threads=2,
            pure_invoke_latency_ms=80.0,
            p50_latency_ms=80.0,
            p95_latency_ms=100.0,
            min_latency_ms=75.0,
            max_latency_ms=110.0,
            throughput_img_s=50.0,  # Deliberately wrong
            preprocessing_latency_ms=0.1,
            postprocessing_latency_ms=0.1,
            end_to_end_latency_ms=80.2,
            warmup_runs=10,
            timed_iterations=100,
            provenance=res.provenance
        )
        with self.assertRaises(ValueError):
            res_bad.validate()


if __name__ == "__main__":
    unittest.main()
