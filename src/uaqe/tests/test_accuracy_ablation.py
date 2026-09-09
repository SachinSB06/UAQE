"""Automated test suite for the UAQE Accuracy Ablation Framework."""

from __future__ import annotations

import os
import sys
import json
import copy
import shutil
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from uaqe.common.value_objects import QuantizationConfig, CompressionConfig, OptimizationConfig
from uaqe.common.types import Precision, CompressionType
from uaqe.application.workflow_config import WorkflowConfig
from uaqe.application.stage_factory import StageFactory, StageFactoryDependencies

import accuracy_ablation


class TestAccuracyAblation(unittest.TestCase):
    """Unit and integration tests for the accuracy ablation components."""

    def setUp(self) -> None:
        self.config_dir = os.path.join(PROJECT_ROOT, "src", "config")
        self.model_path = os.path.join(PROJECT_ROOT, "src", "models", "mobilenetv3_sem.onnx")
        self.dataset_path = os.path.join(PROJECT_ROOT, "datasets", "hackathon_test_dataset")
        self.output_dir = os.path.join(PROJECT_ROOT, "reports", "test_ablation_outputs")
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self) -> None:
        if os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
            except Exception:
                pass

    # 1. Config tests
    def test_fp32_baseline_config(self) -> None:
        overrides = accuracy_ablation.get_experiment_overrides("fp32_baseline")
        self.assertEqual(overrides["quantization.default_precision"], "FP32")
        self.assertEqual(overrides["compression.enabled_types"], [])

    def test_int8_only_config(self) -> None:
        overrides = accuracy_ablation.get_experiment_overrides("int8_only")
        self.assertEqual(overrides["quantization.default_precision"], "INT8")
        self.assertEqual(overrides["compression.enabled_types"], [])

    def test_pruning_sweep_overrides(self) -> None:
        """Verify overrides map correctly for all required sweep pruning levels."""
        for level in [5, 10, 15, 20, 30]:
            name = f"int8_pruning_{level}"
            overrides = accuracy_ablation.get_experiment_overrides(name)
            self.assertEqual(overrides["quantization.default_precision"], "INT8")
            self.assertIn("PRUNING", overrides["compression.enabled_types"])
            self.assertAlmostEqual(overrides["compression.pruning_sparsity"], level / 100.0)

    def test_clustering_overrides(self) -> None:
        overrides = accuracy_ablation.get_experiment_overrides("int8_clustering")
        self.assertEqual(overrides["quantization.default_precision"], "INT8")
        self.assertIn("WEIGHT_CLUSTERING", overrides["compression.enabled_types"])
        self.assertEqual(overrides["compression.clustering_clusters"], 256)

    def test_best_pruning_clustering_overrides(self) -> None:
        """Verify dynamic candidate configuration overrides map correctly."""
        overrides = accuracy_ablation.get_experiment_overrides("int8_best_pruning_clustering", best_pruning=0.15)
        self.assertEqual(overrides["quantization.default_precision"], "INT8")
        self.assertIn("PRUNING", overrides["compression.enabled_types"])
        self.assertIn("WEIGHT_CLUSTERING", overrides["compression.enabled_types"])
        self.assertEqual(overrides["compression.pruning_sparsity"], 0.15)
        self.assertEqual(overrides["compression.clustering_clusters"], 256)

    # 2. Stage overrides and config isolation tests
    def test_experiment_config_isolation(self) -> None:
        """Verify that overrides do not leak to the base configs in StageFactory."""
        base_quant = QuantizationConfig(
            default_precision=Precision.INT8,
            per_layer_overrides={},
            calibration_batch_size=16,
            sensitivity_threshold=0.05,
        )
        base_comp = CompressionConfig(
            enabled_types=[CompressionType.PRUNING],
            target_ratio=0.5,
            pruning_sparsity=0.2,
        )
        base_opt = OptimizationConfig()

        deps = StageFactoryDependencies(
            framework_adapters=[],
            hardware_repository=MagicMock(),
            quantization_config=base_quant,
            quantization_strategies={},
            compression_config=base_comp,
            compression_strategies={},
            optimization_config=base_opt,
            exporter_backends=[],
            report_renderers=[],
        )

        factory = StageFactory(MagicMock(), deps)

        wf_config_1 = WorkflowConfig(
            model_path="model.onnx",
            hardware_profile_id="pi",
            config_overrides={
                "quantization.default_precision": "FP32",
                "compression.enabled_types": ["WEIGHT_CLUSTERING"],
                "compression.clustering_clusters": 128,
            },
        )
        
        stage_sa = factory.create_stage("sensitivity_analyzer", wf_config_1)
        self.assertIsNotNone(stage_sa)

        # Confirm base config was NOT mutated
        self.assertEqual(deps.quantization_config.default_precision, Precision.INT8)
        self.assertEqual(deps.compression_config.enabled_types, [CompressionType.PRUNING])

    def test_production_config_verification(self) -> None:
        """Verify production snapshot comparison functions."""
        before = {
            "quantization.json": {"default_precision": "INT8"},
            "compression.json": {"enabled_types": ["PRUNING"]},
            "optimization.json": {"search": "active"},
        }
        after_same = copy.deepcopy(before)
        after_diff = copy.deepcopy(before)
        after_diff["quantization.json"]["default_precision"] = "FP16"

        status_same, details_same = accuracy_ablation.verify_config_snapshots(before, after_same)
        status_diff, details_diff = accuracy_ablation.verify_config_snapshots(before, after_diff)

        self.assertEqual(status_same, "UNCHANGED")
        self.assertEqual(status_diff, "CHANGED — FAILURE")
        self.assertEqual(details_diff["quantization.json"], "CHANGED")

    def test_production_artifacts_verification(self) -> None:
        """Verify production artifacts snapshot comparison functions."""
        before = {
            "raspberrypi5/model.onnx": "hash1",
            "raspberrypi5/model.tflite": "hash2",
        }
        after_same = copy.deepcopy(before)
        after_diff = copy.deepcopy(before)
        after_diff["raspberrypi5/model.tflite"] = "hash3"

        status_same, details_same = accuracy_ablation.verify_artifacts_snapshots(before, after_same)
        status_diff, details_diff = accuracy_ablation.verify_artifacts_snapshots(before, after_diff)

        self.assertEqual(status_same, "UNCHANGED")
        self.assertEqual(status_diff, "CHANGED — FAILURE")
        self.assertEqual(details_diff["raspberrypi5/model.tflite"], "MODIFIED")

    def test_project_root_resolution(self) -> None:
        """Verify PROJECT_ROOT resolves correctly to the absolute directory of accuracy_ablation.py."""
        self.assertTrue(os.path.isdir(accuracy_ablation.PROJECT_ROOT))
        self.assertTrue(os.path.isabs(accuracy_ablation.PROJECT_ROOT))
        self.assertTrue(os.path.exists(os.path.join(accuracy_ablation.PROJECT_ROOT, "accuracy_ablation.py")))

    # 3. Subprocess verification & Evaluation mocks
    @patch("subprocess.run")
    @patch("accuracy_ablation.validate_and_hash_artifact")
    def test_subprocess_failure_detection(
        self, mock_validate, mock_run
    ) -> None:
        """Verify runner fails when main.py returns non-zero code."""
        mock_run.return_value = MagicMock(returncode=1, stdout="Failed logs", stderr="Error message")
        mock_validate.return_value = (True, "VALIDATED", "hash")

        dataset_mock = MagicMock()
        eval_config = {"class_mapping": {}, "preprocessing": "rgb_0_1"}

        with self.assertRaises(RuntimeError) as context:
            accuracy_ablation.run_experiment(
                "fp32_baseline",
                {"quantization.default_precision": "FP32"},
                self.model_path,
                dataset_mock,
                eval_config,
                self.output_dir,
            )
        self.assertIn("failed compiling", str(context.exception))

    @patch("subprocess.run")
    @patch("accuracy_ablation.validate_and_hash_artifact")
    def test_missing_artifact_detection(self, mock_validate, mock_run) -> None:
        """Verify exception raised when output files fail validation."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Success", stderr="")
        mock_validate.return_value = (False, "FILE_NOT_FOUND", "NOT GENERATED")

        dataset_mock = MagicMock()
        eval_config = {"class_mapping": {}, "preprocessing": "rgb_0_1"}

        with self.assertRaises(RuntimeError) as context:
            accuracy_ablation.run_experiment(
                "fp32_baseline",
                {"quantization.default_precision": "FP32"},
                self.model_path,
                dataset_mock,
                eval_config,
                self.output_dir,
            )
        self.assertIn("failed validation", str(context.exception))

    @patch("subprocess.run")
    @patch("accuracy_ablation.validate_and_hash_artifact")
    @patch("accuracy_ablation.RealInferenceEvaluator")
    def test_metrics_generated(
        self, mock_evaluator, mock_validate, mock_run
    ) -> None:
        """Verify metrics collection and nested latency key retrieval."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Success", stderr="")
        mock_validate.return_value = (True, "VALIDATED", "hash_val")

        eval_inst = mock_evaluator.return_value
        eval_inst.evaluate_accuracy_and_numerical_metrics.return_value = (
            {"tflite_accuracy": 0.35},
            {"class_precision": {}},
            {"fp32_vs_tflite": {"cosine_similarity": 0.98}},
        )
        eval_inst.audit_model_size.return_value = {"tflite_size_bytes": 1000}
        eval_inst.benchmark_latency.return_value = {
            "tflite": {
                "benchmark_status": "PASS",
                "mean_ms": 2.5,
                "throughput_ips": 400.0
            }
        }
        eval_inst.benchmark_memory.return_value = {"tflite": {"load_delta_bytes": 500}}

        dataset_mock = MagicMock()
        eval_config = {"class_mapping": {}, "preprocessing": "rgb_0_1"}

        summary = accuracy_ablation.run_experiment(
            "fp32_baseline",
            {"quantization.default_precision": "FP32"},
            self.model_path,
            dataset_mock,
            eval_config,
            self.output_dir,
        )

        self.assertEqual(summary["status"], "SUCCESS")
        self.assertEqual(summary["accuracy"]["tflite_accuracy"], 0.35)
        self.assertEqual(summary["size"]["tflite_size_bytes"], 1000)
        self.assertEqual(summary["performance"]["tflite"]["mean_ms"], 2.5)

    @patch("subprocess.run")
    @patch("accuracy_ablation.validate_and_hash_artifact")
    @patch("accuracy_ablation.RealInferenceEvaluator")
    def test_tflite_not_generated_tolerance(
        self, mock_evaluator, mock_validate, mock_run
    ) -> None:
        """Verify runner tolerates missing TFLite model and reports NOT GENERATED."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Success", stderr="")
        
        # ONNX succeeds, TFLite fails validation/not generated
        def side_effect(path, format_type):
            if format_type == "ONNX":
                return True, "VALIDATED", "onnx_hash"
            return False, "NOT GENERATED", "NOT GENERATED"
        mock_validate.side_effect = side_effect

        eval_inst = mock_evaluator.return_value
        eval_inst.evaluate_accuracy_and_numerical_metrics.return_value = (
            {"tflite_accuracy": "NOT GENERATED"},
            {"class_precision": {}},
            {"fp32_vs_tflite": "NOT GENERATED"},
        )
        eval_inst.audit_model_size.return_value = {
            "tflite_size_bytes": "NOT GENERATED"
        }
        eval_inst.benchmark_latency.return_value = {
            "tflite": {
                "benchmark_status": "FAILED",
                "mean_ms": "NOT GENERATED"
            },
            "onnx": {
                "benchmark_status": "PASS",
                "mean_ms": 1.2,
                "throughput_ips": 833.3
            }
        }
        eval_inst.benchmark_memory.return_value = {
            "tflite": {"load_delta_bytes": "NOT GENERATED"}
        }

        dataset_mock = MagicMock()
        eval_config = {"class_mapping": {}, "preprocessing": "rgb_0_1"}

        summary = accuracy_ablation.run_experiment(
            "fp32_baseline",
            {"quantization.default_precision": "FP32"},
            self.model_path,
            dataset_mock,
            eval_config,
            self.output_dir,
        )

        self.assertEqual(summary["status"], "SUCCESS")
        self.assertEqual(summary["artifacts"]["tflite"]["validation_status"], "NOT GENERATED")
        self.assertEqual(summary["size"]["tflite_size_bytes"], "NOT GENERATED")

    def test_invalid_experiment_name(self) -> None:
        with self.assertRaises(ValueError):
            accuracy_ablation.get_experiment_overrides("invalid_experiment_name")


if __name__ == "__main__":
    unittest.main()
