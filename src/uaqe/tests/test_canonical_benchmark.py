"""Test Suite for UAQE Canonical Pure Inference Benchmark and Job Isolation."""

import unittest
import os
import sys
import json
import hashlib
from typing import Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from uaqe.telemetry.canonical_benchmark import (
    CanonicalBenchmark,
    CanonicalBenchmarkResult,
    BenchmarkProvenance
)
from uaqe.telemetry.runtime_monitor import (
    RuntimeMonitor,
    pause_active_monitors,
    resume_active_monitors,
    _ACTIVE_MONITORS
)


class TestCanonicalBenchmark(unittest.TestCase):
    """Rigorous verification of the canonical pure inference benchmark."""

    @classmethod
    def setUpClass(cls):
        cls.mobilenet_int8 = "output/phase_c2/models/mobilenetv3_sem_9class_qat_int8.tflite"
        if not os.path.exists(cls.mobilenet_int8):
            cls.mobilenet_int8 = "output/phase_c3/models/c3_best_int8.tflite"
        
        cls.mobilenet_fp32 = "output/phase_c1/models/mobilenetv3_sem_9class_fp32.onnx"

    def test_01_mobilenet_artifact_integrity(self):
        """Phase 12: Verified MobileNet artifact SHA256 and size must match baseline exactly."""
        self.assertTrue(os.path.exists(self.mobilenet_int8), f"Artifact missing: {self.mobilenet_int8}")
        
        size = os.path.getsize(self.mobilenet_int8)
        self.assertEqual(size, 1855816, f"Artifact size mismatch: {size} != 1,855,816 bytes")

        h = hashlib.sha256()
        with open(self.mobilenet_int8, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        sha256 = h.hexdigest()
        self.assertEqual(
            sha256,
            "10d51d4f243c554734e7347f9010297fb182a64c4ad89d5551e24830846dab1d",
            f"Artifact SHA256 mismatch: {sha256}"
        )

    def test_02_canonical_tflite_benchmark_protocol(self):
        """Phase 2 & Phase 4: Canonical INT8 benchmark executes strictly 10 warmups, 100 timed invokes, 2 threads."""
        res = CanonicalBenchmark.benchmark_tflite(
            model_path=self.mobilenet_int8,
            input_shape=(1, 3, 128, 128),
            num_threads=2,
            warmup_runs=10,
            measured_runs=100
        )

        self.assertIsInstance(res, CanonicalBenchmarkResult)
        self.assertEqual(res.warmup_runs, 10)
        self.assertEqual(res.timed_iterations, 100)
        self.assertEqual(res.num_threads, 2)
        self.assertEqual(len(res.raw_latencies_ms), 100)

        # Strict positive latency & throughput
        self.assertGreater(res.pure_invoke_latency_ms, 0)
        self.assertGreater(res.p50_latency_ms, 0)
        self.assertGreater(res.p95_latency_ms, 0)
        self.assertGreater(res.throughput_img_s, 0)

        # Strict throughput formula
        expected_tput = 1000.0 / res.pure_invoke_latency_ms
        self.assertAlmostEqual(res.throughput_img_s, expected_tput, delta=0.1)

    def test_03_canonical_onnx_benchmark_protocol(self):
        """Phase 2 & Phase 3: Canonical FP32 benchmark executes strictly 10 warmups, 100 timed invokes, 2 threads."""
        if not os.path.exists(self.mobilenet_fp32):
            self.skipTest(f"FP32 ONNX model missing at {self.mobilenet_fp32}")

        res = CanonicalBenchmark.benchmark_onnx(
            model_path=self.mobilenet_fp32,
            input_shape=(1, 3, 128, 128),
            num_threads=2,
            warmup_runs=10,
            measured_runs=100
        )

        self.assertIsInstance(res, CanonicalBenchmarkResult)
        self.assertEqual(res.warmup_runs, 10)
        self.assertEqual(res.timed_iterations, 100)
        self.assertEqual(res.num_threads, 2)
        self.assertEqual(len(res.raw_latencies_ms), 100)

        # Strict positive latency & throughput
        self.assertGreater(res.pure_invoke_latency_ms, 0)
        self.assertGreater(res.p50_latency_ms, 0)
        self.assertGreater(res.p95_latency_ms, 0)
        self.assertGreater(res.throughput_img_s, 0)

        # Strict throughput formula
        expected_tput = 1000.0 / res.pure_invoke_latency_ms
        self.assertAlmostEqual(res.throughput_img_s, expected_tput, delta=0.1)

    def test_04_benchmark_provenance_completeness(self):
        """Phase 7: Verify all 18 required provenance fields are populated and valid."""
        res = CanonicalBenchmark.benchmark_tflite(
            model_path=self.mobilenet_int8,
            input_shape=(1, 3, 128, 128),
            num_threads=2,
            warmup_runs=10,
            measured_runs=100
        )
        prov = res.provenance.to_dict()

        required_fields = [
            "evaluator",
            "runtime",
            "runtime_mode",
            "performance_enabled",
            "num_threads",
            "batch_size",
            "input_shape",
            "warmup_count",
            "timed_iterations",
            "latency_type",
            "throughput_type",
            "preprocessing_in_timing",
            "postprocessing_in_timing",
            "dataset_io_in_timing",
            "runtime_monitor_active_during_timing",
            "gc_before_timing",
            "benchmark_version",
            "artifact_sha256",
            "artifact_size_bytes"
        ]

        for field_name in required_fields:
            self.assertIn(field_name, prov, f"Missing required provenance field: {field_name}")

        self.assertEqual(prov["warmup_count"], 10)
        self.assertEqual(prov["timed_iterations"], 100)
        self.assertEqual(prov["num_threads"], 2)
        self.assertEqual(prov["batch_size"], 1)
        self.assertFalse(prov["preprocessing_in_timing"])
        self.assertFalse(prov["postprocessing_in_timing"])
        self.assertFalse(prov["dataset_io_in_timing"])
        self.assertFalse(prov["runtime_monitor_active_during_timing"])
        self.assertTrue(prov["gc_before_timing"])
        self.assertEqual(prov["artifact_sha256"], "10d51d4f243c554734e7347f9010297fb182a64c4ad89d5551e24830846dab1d")
        self.assertEqual(prov["artifact_size_bytes"], 1855816)

    def test_05_runtime_monitor_pause_and_isolation(self):
        """Phase 6: RuntimeMonitor is paused during timed sections without psutil contention."""
        monitor = RuntimeMonitor(sample_interval_sec=0.1)
        monitor.start()
        try:
            self.assertIn(monitor, _ACTIVE_MONITORS)
            self.assertFalse(monitor.is_paused())

            pause_active_monitors()
            self.assertTrue(monitor.is_paused())

            resume_active_monitors()
            self.assertFalse(monitor.is_paused())
        finally:
            monitor.stop()
            self.assertNotIn(monitor, _ACTIVE_MONITORS)


if __name__ == "__main__":
    unittest.main()
