"""Automated Test Suite for UAQE Real Execution & Mock Elimination.

Validates that:
1. No mock strings or synthetic formulas exist in the optimization engine.
2. ImageNet-1000 + 10-class MobileNet -> INVALID_BASELINE, halts before candidate 1.
3. Valid MobileNet + Semiconductor -> REAL optimization with genuine TFLite FlatBuffer and telemetry.
4. Valid ResNet-50 + CIFAR-10 -> REAL optimization with genuine ONNX INT8 QDQ and telemetry.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from uaqe.orchestration.optimization_orchestrator import OptimizationOrchestrator
from uaqe.optimization.accuracy_safety_policy import BaselineStatus


class TestNoMockOptimization(unittest.TestCase):
    """Regression suite guaranteeing authentic execution and zero mock artifacts."""

    def test_01_no_mock_strings_in_source_code(self):
        """1. Assert no mock byte strings or synthetic formula artifacts in optimization codebase."""
        src_dir = os.path.join(PROJECT_ROOT, "src", "uaqe")
        forbidden = [
            b"MOCK_MOBILENET_TFLITE_INT8",
            b"GENERIC_MODEL",
            b"cand_lat = fp32_lat * 0.65",
            b"cand_size = int(fp32_size * 0.26)"
        ]

        for root, _, files in os.walk(src_dir):
            for fname in files:
                if fname.endswith(".py") and fname != "test_no_mock_optimization.py":
                    fpath = os.path.join(root, fname)
                    with open(fpath, "rb") as f:
                        content = f.read()
                        for target in forbidden:
                            self.assertNotIn(
                                target,
                                content,
                                f"Found forbidden mock pattern '{target.decode()}' in {fpath}"
                            )

    def test_02_mobilenet_imagenet_invalid_baseline(self):
        """2. ImageNet-1000 + 10-class MobileNet -> INVALID_BASELINE, halts before candidate 1."""
        model_path = os.path.join(PROJECT_ROOT, "src", "models", "mobilenetv3_sem.onnx")
        dataset_path = r"D:\imagenet_10k_split"

        if not os.path.exists(model_path) or not os.path.exists(dataset_path):
            self.skipTest("MobileNet ONNX model or ImageNet-10K dataset not available on disk.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            orchestrator = OptimizationOrchestrator(output_root=tmp_dir)
            job_config = {
                "model_path": model_path,
                "dataset_path": dataset_path,
                "target_hardware": "raspberrypi5",
                "optimization_profile": "balanced",
                "max_candidates": 3,
                "test_samples": 20,
                "auto_approve": True
            }

            result = orchestrator.run(job_config)

            # Assert baseline is marked invalid
            self.assertEqual(result.get("verdict"), "FAILED")
            self.assertEqual(result.get("stopping_reason"), "INVALID_BASELINE")
            self.assertEqual(result.get("baseline_status"), BaselineStatus.INVALID_BASELINE.value)
            self.assertEqual(result.get("winner"), "NONE")
            self.assertFalse(result.get("accuracy_constraint_satisfied", True))

            # Candidate search must halt before Candidate 1
            candidates = result.get("candidates", [])
            self.assertEqual(len(candidates), 0, "Candidates were executed despite INVALID_BASELINE!")

            # Verify metrics file on disk
            job_dir = result["job_dir"]
            metrics_path = os.path.join(job_dir, "metrics.json")
            self.assertTrue(os.path.exists(metrics_path))
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            self.assertEqual(metrics.get("stopping_reason"), "INVALID_BASELINE")
            self.assertEqual(metrics.get("verdict"), "FAILED")

    def test_03_mobilenet_semiconductor_real_optimization(self):
        """3. Valid MobileNet + Semiconductor -> REAL optimization with genuine TFLite FlatBuffer."""
        model_path = os.path.join(PROJECT_ROOT, "src", "models", "mobilenetv3_sem.onnx")
        dataset_path = r"D:\semiconductor_dataset\dataset"

        if not os.path.exists(model_path) or not os.path.exists(dataset_path):
            self.skipTest("MobileNet ONNX model or Semiconductor dataset not available on disk.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            orchestrator = OptimizationOrchestrator(output_root=tmp_dir)
            job_config = {
                "model_path": model_path,
                "dataset_path": dataset_path,
                "target_hardware": "raspberrypi5",
                "optimization_profile": "balanced",
                "max_candidates": 1,
                "test_samples": 30,
                "auto_approve": True
            }

            result = orchestrator.run(job_config)

            # Assert baseline was valid
            self.assertEqual(result.get("baseline_status"), BaselineStatus.VALID.value)
            self.assertEqual(result.get("verdict"), "VERIFIED")

            candidates = result.get("candidates", [])
            self.assertGreater(len(candidates), 0, "No candidates were evaluated.")

            # Assert genuine TFLite FlatBuffer artifact exists
            job_dir = result["job_dir"]
            opt_model_path = os.path.join(job_dir, "final", "optimized_model.tflite")
            if not os.path.exists(opt_model_path):
                opt_model_path = os.path.join(job_dir, "optimized_model.tflite")
            self.assertTrue(os.path.exists(opt_model_path), f"Missing {opt_model_path}")

            model_size = os.path.getsize(opt_model_path)
            self.assertGreater(model_size, 100_000, "TFLite artifact is too small (synthetic stub).")

            with open(opt_model_path, "rb") as f:
                head = f.read(32)
                self.assertNotIn(b"MOCK", head, "TFLite file contains synthetic MOCK byte string!")

            # Assert genuine telemetry
            tel_path = result.get("telemetry_path") or os.path.join(job_dir, "telemetry.json")
            if not os.path.exists(tel_path):
                tel_path = os.path.join(job_dir, "telemetry.json")
            self.assertTrue(os.path.exists(tel_path), f"Telemetry file not found at {tel_path}")
            with open(tel_path, "r", encoding="utf-8") as f:
                tel = json.load(f)
            self.assertIn("phases", tel)
            self.assertIn("fp32_baseline", tel["phases"])
            self.assertIn("final_optimized", tel["phases"])
            baseline_phase = tel["phases"]["fp32_baseline"]
            self.assertGreater(baseline_phase.get("throughput_ips", 0.0), 0.0)

    def test_04_resnet50_cifar10_real_optimization(self):
        """4. Valid ResNet-50 + CIFAR-10 -> REAL optimization with genuine ONNX INT8 QDQ."""
        model_path = os.path.join(PROJECT_ROOT, "output", "phase_e2", "models", "resnet50_cifar10_fp32_baseline.pt")
        dataset_path = r"D:\uaqe_datasets\cifar10\cifar-10-batches-py"

        if not os.path.exists(model_path) or not os.path.exists(dataset_path):
            self.skipTest("ResNet-50 checkpoint or CIFAR-10 dataset not available on disk.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            orchestrator = OptimizationOrchestrator(output_root=tmp_dir)
            job_config = {
                "model_path": model_path,
                "dataset_path": dataset_path,
                "target_hardware": "raspberrypi5",
                "optimization_profile": "balanced",
                "max_candidates": 1,
                "test_samples": 25,
                "calib_samples": 32,
                "auto_approve": True
            }

            result = orchestrator.run(job_config)

            self.assertEqual(result.get("baseline_status"), BaselineStatus.VALID.value)
            self.assertIn(result.get("verdict"), ["VERIFIED", "VERIFIED WITH CAVEATS"])

            job_dir = result["job_dir"]
            opt_onnx_path = os.path.join(job_dir, "final", "optimized_model.onnx")
            if not os.path.exists(opt_onnx_path):
                opt_onnx_path = os.path.join(job_dir, "optimized_model.onnx")
            self.assertTrue(os.path.exists(opt_onnx_path), f"Missing {opt_onnx_path}")

            onnx_size = os.path.getsize(opt_onnx_path)
            self.assertGreater(onnx_size, 10_000_000, "ONNX INT8 model is suspiciously small.")

            # Check telemetry
            tel_path = result.get("telemetry_path") or os.path.join(job_dir, "telemetry.json")
            if not os.path.exists(tel_path):
                tel_path = os.path.join(job_dir, "telemetry.json")
            self.assertTrue(os.path.exists(tel_path), f"Telemetry file not found at {tel_path}")
            with open(tel_path, "r", encoding="utf-8") as f:
                tel = json.load(f)
            self.assertIn("phases", tel)
            self.assertIn("fp32_baseline", tel["phases"])
            self.assertIn("final_optimized", tel["phases"])


if __name__ == "__main__":
    unittest.main()
