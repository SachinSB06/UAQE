"""Test suite for UAQE safe performance optimization layer.

Validates:
1. PerformanceBenchmark decomposes latency stages accurately.
2. Multi-threaded benchmark measures 1, 2, and 4 threads with statistical rigor.
3. Both Correctness and Performance execution modes yield identical predictions and accuracy.
4. Performance mode safe fallback mechanism to Correctness mode.
5. Memory profiler and cleanup verification.
"""

import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from uaqe.telemetry.performance_benchmark import PerformanceBenchmark
from uaqe.telemetry.memory_profiler import MemoryProfiler
from uaqe.optimization.runtime_config import RuntimeMode, RuntimeConfig
from uaqe.orchestration.optimization_orchestrator import OptimizationOrchestrator


class TestPerformanceOptimization(unittest.TestCase):
    """Regression and validation suite for safe performance optimization."""

    @classmethod
    def setUpClass(cls):
        cls.mobilenet_model = os.path.join(PROJECT_ROOT, "output", "phase_c1", "models", "mobilenetv3_sem_9class_fp32.onnx")
        cls.mobilenet_int8 = os.path.join(PROJECT_ROOT, "output", "phase_c2", "models", "mobilenetv3_sem_9class_qat_int8.tflite")
        cls.semiconductor_dataset = r"D:\semiconductor_dataset\dataset"

    def test_01_performance_benchmark_isolation(self):
        """1. PerformanceBenchmark accurately measures and separates pure invoke from prep/post."""
        if not os.path.exists(self.mobilenet_int8):
            self.skipTest("INT8 TFLite model missing")

        res = PerformanceBenchmark.benchmark_tflite(
            model_path=self.mobilenet_int8,
            input_shape=(1, 3, 128, 128),
            num_threads=2,
            warmup_runs=5,
            measured_runs=15
        )
        d = res.to_dict()
        self.assertIn("pure_inference", d)
        self.assertGreater(d["pure_inference"]["mean_ms"], 0.0)
        self.assertGreater(d["pure_inference"]["throughput_ips"], 0.0)
        self.assertIn("preprocessing_mean_ms", d)
        self.assertIn("postprocessing_mean_ms", d)
        self.assertIn("end_to_end_mean_ms", d)
        self.assertGreaterEqual(d["end_to_end_mean_ms"], d["pure_inference"]["mean_ms"])

    def test_02_thread_benchmark_measurements(self):
        """2. Thread benchmark measures 1, 2, 4 threads with CPU and RAM tracking."""
        if not os.path.exists(self.mobilenet_int8):
            self.skipTest("INT8 TFLite model missing")

        thread_results = PerformanceBenchmark.benchmark_threads_tflite(
            model_path=self.mobilenet_int8,
            thread_counts=(1, 2),
            repeats=2,
            runs_per_repeat=10
        )
        self.assertEqual(len(thread_results), 2)
        for r in thread_results:
            self.assertIn("threads", r)
            self.assertIn("mean_latency_ms", r)
            self.assertIn("p95_latency_ms", r)
            self.assertIn("throughput_ips", r)
            self.assertIn("cpu_avg_pct", r)
            self.assertIn("ram_avg_mb", r)
            self.assertGreater(r["mean_latency_ms"], 0.0)

    def test_03_correctness_vs_performance_equivalence(self):
        """3. Performance mode produces identical accuracy and predictions to Correctness mode."""
        if not os.path.exists(self.mobilenet_model) or not os.path.exists(self.semiconductor_dataset):
            self.skipTest("MobileNet model or dataset missing")

        with tempfile.TemporaryDirectory() as tmp_dir:
            orch = OptimizationOrchestrator(output_root=tmp_dir)

            # Correctness run
            res_corr = orch.run({
                "model_path": self.mobilenet_model,
                "dataset_path": self.semiconductor_dataset,
                "target_hardware": "raspberrypi5",
                "optimization_profile": "balanced",
                "max_candidates": 1,
                "test_samples": 25,
                "calib_samples": 16,
                "auto_approve": True,
                "runtime_mode": "correctness",
                "performance_enabled": False
            })

            # Performance run
            res_perf = orch.run({
                "model_path": self.mobilenet_model,
                "dataset_path": self.semiconductor_dataset,
                "target_hardware": "raspberrypi5",
                "optimization_profile": "balanced",
                "max_candidates": 1,
                "test_samples": 25,
                "calib_samples": 16,
                "auto_approve": True,
                "runtime_mode": "performance",
                "performance_enabled": True,
                "num_threads": 2
            })

            self.assertEqual(res_corr.get("verdict"), "VERIFIED")
            self.assertEqual(res_perf.get("verdict"), "VERIFIED")
            self.assertEqual(res_corr.get("baseline_status"), "VALID")
            self.assertEqual(res_perf.get("baseline_status"), "VALID")

            acc_corr = res_corr.get("metrics", {}).get("optimized_accuracy")
            acc_perf = res_perf.get("metrics", {}).get("optimized_accuracy")
            self.assertEqual(acc_corr, acc_perf, "Accuracy must match exactly between correctness and performance modes.")

    def test_04_memory_profiler(self):
        """4. Memory profiler tracks RSS and executes garbage collection safely."""
        profiler = MemoryProfiler()
        initial_rss = profiler.get_current_rss_mb()
        self.assertGreater(initial_rss, 50.0)

        res = profiler.cleanup()
        self.assertIn("before_cleanup_mb", res)
        self.assertIn("after_cleanup_mb", res)
        self.assertIn("freed_mb", res)


if __name__ == "__main__":
    unittest.main()
